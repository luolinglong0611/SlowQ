# 变更记录

[返回 README](../README.md)

## 未发布 · 推理加速第 1 轮（2026-10-23）

只减少**重复计算**并新增**显式**复用接口：不改数值路径，不引入 GPU 后端、量化、
推测解码或并发服务。默认运行路径仍是 Python + NumPy、CPU、FP32、batch size 1。
8 个模型、3 类任务与既有公开 API 全部兼容（新增的都是加法）；运行时模块数仍是
**14 个**，`tests/test_dependency_boundary.py` 一行未改，只读 mmap 与调试观测能力保留。

前后对照、命令、原始结果位置与适用条件见
[推理加速第 1 轮实测](推理加速-第1轮-2026-10-23.md)；设计取舍见
[架构说明](architecture.md) 的 §4 / §5 / §6.1 / §7.1 / §8.5 / §9。

### 优化 1：非末 prefill chunk 不再做词表投影

- `CausalLMModel.prefill()`：非末 chunk 走 `encode()`（decoder stack + KV 写入，不碰
  `lm_head`），只有末 chunk 走 `forward()`（`encode()` + 一次投影）。
  512 token / `chunk_size=128` 从 4 次全词表投影降到 1 次。
- **逐位相同，不是近似**：`test_prefill_matches_projecting_every_chunk_exactly` 直接拿
  "每个 chunk 都 `forward` 再取最后一个" 和 `prefill()` 比 `np.array_equal`。
  `forward()` 本来就只为最后一个位置投影（`model.py:398`），所以省掉的那几次
  产出从未被读取。
- 要求观测 `logits` 的 `DebugProbe` 会恢复每 chunk 投影：调试运行可以慢，不可以算出
  另一个结果。`kv_cache.commit` 仍每 chunk 一次、仍在 `encode` 内部，§4 的不变量不受影响。
- `tools/benchmark.py` 同步修正：它的 `time_prefill` 原先**自己手写分块循环**逐块调
  `forward`，与引擎走的不是一条代码；现在改调 `model.prefill()`。
  `ForwardSpy` 同时统计 `forward` 与 `encode`（带深度守卫，避免一个 chunk 被数两次），
  报告里多出 `vocabulary_projections` 一列——**分块数是被观测到的，不是从
  `length // chunk_size` 推出来的**。
- **省掉的工作可观测，加速测不出来**，这两件事分开说。投影次数 4 → 1 由 spy 直接数出；
  但一次投影的净代价用成对 `forward` − `encode` 单独定价后（`tools/perf/projection_probe.py`，
  同树、同批 id、同进程），只占 512-token 分块 prefill 的 **-0.32% ~ 3.02%**。
  这个区间落在本轮观测到过的跨进程离散度量级以内（同一配置两个独立进程：
  Qwen3-0.6B 3.0%、SmolLM2-360M 0.4%），**判读为「尚不能确认收益或回退」**。
  本条早先写作"在实测的约 3% 跨进程噪声底线以内或以此为上界"——那个说法**已撤回**：
  两点数据没有分布，同一个探针在两个模型上的跨进程差就差了一个量级，
  取较大者当所有负载的统一门限没有依据，理由见
  [推理加速第 1 轮实测](推理加速-第1轮-2026-10-23.md) §7.2。代价随
  `vocab_size × hidden_size` 走：SmolLM2-135M 一次 0.86 ms，Qwen3-1.7B 一次 11.04 ms。
  **保留它的理由是免费且逐位相同，不是因为它快**；同一 prefill 上 `sigmoid` 改写
  拿走了绝大部分（两条独立路线交叉核对，相差 1.1% ~ 10.0%）。
  `model.py` 原先那句"被丢掉的三次投影是长 prefill 里最大的可避免开销"**是错的，已按实测重写**。

### 优化 2：decode 与 attention 的冗余工作

- 单 token decode 在"完整因果注意力 + 只读有效 KV 前缀"下，causal mask 恒为全零，
  跳过加 mask 这一步。
- `sigmoid` 改写成 `exp(-|x|)` 的无分支形式：`np.where` 取代两次布尔掩码索引。
  掩码索引要在 MLP 的 `[T, intermediate_size]` 激活上做四次 gather，
  改写前后是**同一组表达式在同一个输入上求值**，含 `-0.0`、`±inf`、`nan`，逐位相同。
  逐位结论是拿**原始数组**比字节得到的，不是从日志打印的聚合误差推出来的——
  聚合误差相同不能证明张量逐位相同。`nan` 只保证**位置**一致，
  位模式不保证（`nan / nan` 与 `exp(nan)` 可以落在不同的 payload 上，两边同样都是 NaN）。
- **本条的归因数字被更正过：旧数字全部撤回，但原始证据保留。**
  原来写的是（`artifacts/perf-round1/sigmoid-check.json`，未删除、未改写，
  该文件早于身份记录改造，本身不含源码哈希）：Qwen3-0.6B 一个 128-token prefill chunk 上，
  旧写法 4.853 ms，等于该层 7 个 matmul 全部预算（3.167 ms）的 **153%**；
  新写法 0.674 ms，即 **7.20×** 快、占 matmul 预算的 21%；
  "真实 gate 激活"均值 +0.064、标准差 0.241、38.8% 为负。三处是错的：

  1. **它测的不是真实激活。** 那个自称"真实 gate 激活"的张量实际是
     `rms_norm(embed_tokens(ids), input_layernorm.weight) @ gate_proj.weight.T`，
     而真实的 decoder layer 在 embedding 与 `gate_proj` 之间还有整个 attention 块、
     一次残差加、以及**另一个** norm（`post_attention_layernorm`，权重与
     `input_layernorm` 不是同一个张量）。旧捷径与真实捕获值的 rel_l2 = **1.09**，
     也就是差值比信号本身还大。
  2. **153% 不是任何真实层的占比。** 分母是 7 个 matmul 在热循环里**各自单独**计时的
     中位数之和，里面没有 softmax、没有 norm、没有残差加、没有任何逐元素运算，
     它不是一次整层执行的时间。
  3. **"出自 RMSNorm，天然居中"是错的。** RMSNorm 除以均方根，**从不减均值**；
     减均值的是 LayerNorm，而 RMSNorm 被写出来的一部分理由正是省掉那一步。
     实测也否证了它：真实捕获的激活均值是 **-0.5462**，不是 0。

  更正后的数字来自**真实前向里捕获的激活**，并且**每一个都是 4 次独立运行的区间，
  不是某一次的读数**（`tools/perf/sigmoid_check.py` →
  `artifacts/perf-round1-rerun/sigmoid/`，四次输出全部保留、未覆盖）：

  | 量 | 旧写法 | 新写法 | 结论（4 次区间） |
  | --- | --- | --- | --- |
  | 算子单独计时（真实捕获激活） | 2.844–3.216 ms | 0.672–0.749 ms | 3.90×–4.35× 快 |
  | 真实 MLP，只替换 `silu` | 5.009–5.297 ms | 2.786–2.841 ms | 省下真实 MLP 的 **43.9%–46.9%** |
  | 真实 decoder layer（含 attention / 两个 norm / 两处残差） | 7.478–7.969 ms | 5.327–5.467 ms | 省下真实整层的 **27.3%–31.4%** |

  真实激活均值 **-0.5462**、标准差 **0.6681**、**84.5%** 为负。
  **这个惩罚只在数值跨越 0 时出现**：全正 / 全负合成输入只有 1.06×–1.16×，
  标准正态合成输入 7.29×–7.61×（比真实激活还高）——所以这个倍数不能推广到任意输入分布。
  合成分布**保留**在探针的 `SYNTHETIC_DISTRIBUTIONS` 里，作用是给算子的输入敏感性划界，
  不是对本模型的测量。方向没变（净收益，而且很大），
  **幅度变了，而且幅度本身也只能给到区间**：算子加速从 7.20× 降到 3.90×–4.35×，
  "占整层 153% 量级"换成"真实整层省下 27.3%–31.4%"。
- **一次运行内的连续样本不是重复实验。** 上面的区间是重测过程中才发现的：
  单次运行内部 15 个样本的 stdev/median 只有 **0.4%–3.7%**，看起来已经收敛；
  但 4 次运行之间，真实整层占比相差 **4.1 个百分点**（27.3%–31.4%）。
  本条早先写作"单个进程的读数，重跑会有约 1% 的移动"（并举 `ops.py` docstring 里
  4.810 ms / 7.15× 与 4.853 ms / 7.20× 为例，说那是同一测量的另一次采样）——
  那个约 1% 只描述了算子比值，**没有覆盖整层归因**，已按实测改写。
  这与 §3 decode 那条是同一个道理，只是这次从算子探针一侧又出现了一遍。
- **四个量互不通用**，引用时必须说清是哪一个：(1) 算子单独计时、(2) 7 个 matmul 之和、
  (3) 真实前向里只替换 `silu`、(4) 端到端收益。能称为"MLP 归因"的只有 (3)；
  (4) 在 A/B 负载数据里，本探针不测，**不能由本节下"测得稳定收益"的结论**。
  `tests/test_perf_tooling.py` 从 `ops.sigmoid` 的 docstring 里正则抽出每个区间，
  断言**所有保留的运行都落在区间内**，并断言最新一次的**代码哈希**仍与当前源码一致
  （哈希把语法树里的 docstring 清空后再算，对文档字符串与行号不敏感，
  否则"把结论写回源码"会让刚记录的身份失效——理由见
  [推理加速第 1 轮实测](推理加速-第1轮-2026-10-23.md) §2.4）。
- `group_query_attention` 返回 pre-softmax scores 的契约保留，`softmax` 明确**不得**
  写入入参——否则调用方会在"pre-softmax"这个名字下拿到概率，而且能过所有数值检查。
  缓冲区所有权在 docstring 里写死。

### 优化 3：显式多轮 KV 复用

- 新增 `Engine.session()` → `Session`，以及 `SessionReuse`、`common_prefix_length`、
  `KVCache.truncate()`、`PrefillResult`。`GenerationMetrics` 增加 `reused_tokens`
  与派生的 `prefill_tokens`（`prompt_tokens - reused_tokens`）。
- 复用边界按**实际 token IDs 的最长公共前缀**决定，不按文本：BPE 会在轮次边界重新合并，
  同一段文字重新分词后可能得到不同的 id 序列。少复用只是慢，多复用会让模型
  条件在一段它没见过的对话上。
- 四个边界情况都有测试：完全命中时仍要跑一个 token（生成采样的是最后一个位置的
  **logits**，不是 hidden state）；失败或取消的请求会让 cache 比 `_committed` 长，
  所以 `_plan` 先 `del self._committed[length:]`；`_check_identity` 在模型或几何
  （层数、KV head 数、head dim、词表、`max_context`）变化时拒绝复用，因为缓冲区是
  按旧几何分配的；图文会话直接拒绝。
- 只支持纯文本会话。**普通独立请求仍然互相隔离**：`stream_ids` 的独立路径照旧 reset，
  两个平铺请求不可能看到彼此的 KV；§4 的不变量 2 因此有了一个**写明的**例外。
- `slowq chat` 默认走会话，`--no-reuse` 关掉，`/clear` 调 `session.reset()`；
  Ctrl-C 取消生成后会话仍可用。开启时会打印第二份 cache 的预留体积。

### 优化 4：图片特征与文档向量复用

- 新增 `ImageFeatureCache`（`vlm.py`，LRU + 字节上限，默认 128 MiB，`0` 关闭，负数报错）。
  `VisionEngine(..., image_cache_bytes=)` 与 `from_directory` 都接受该参数。
  只缓存 `VisionOutput.features`（connector 之后），不缓存整个 `VisionOutput`：
  9 帧 SmolVLM 的 `encoder_output` 是 27.0 MiB 而 features 只有 1.3 MiB，
  缓存前者要花 20 倍内存去省同一次前向。
  `put` 拷入、`get` 拷出，两个请求不会持有同一个缓冲区，调用方改自己拿到的数组
  不会污染缓存。带 `DebugProbe` 时**双向绕过**缓存。
  缓存身份包含像素内容与形状、attention mask、crop 排布、原图/resize 尺寸、
  encoder 尺寸、processor 配置与 `Weights.fingerprint()`。
  **（2026-10-28 更正：这一行当时漏了「同权重、不同配置」那个洞，
  现在 key 的最后一项是 `_vision_identity(model)` 五元组，
  且每次查缓存前还比一次模型对象本身——见下面「二次审查修复 · 问题 2」。）**
- 计时诚实：`vision_encode_seconds` 在命中时是**本次请求自己的**哈希时间，
  不是当初那次前向的时长；`metrics.vision_cache_hit` 标记命中，
  `report()` 输出形如 `16.0 ms (features cached, vision tower skipped)`。
- CLI 的两个图文入口显式传 `image_cache_bytes=0`：一问一答即退出，缓存只会在
  没有任何后续请求会用到它的情况下白付一次像素哈希。
  库路径默认开启，那才是"对同一张图问第二个问题"真正发生的地方。
- **上面那两个耗时都被测过，而且测出旧数字是错的。**
  （下面这几条的毫秒数来自 `artifacts/perf-round1-rerun/identity/`，
  `vlm.py` 是 `77167e65…`；**当前源码对应的那一批**与「哪些还能用」的判定
  见「二次审查修复 · 本轮的身份代价数字换了批次」。）
  哈希一次图片的像素
  （`_image_cache_key`）在 SmolVLM-256M 的两张验证图上是 **5 帧 15.41–16.33 ms、
  9 帧 27.89–29.35 ms**，三个独立进程互差在 6% 以内
  （`tools/perf/identity_cost.py` → `artifacts/perf-round1-rerun/identity/`）。
  本条早先写的是「5 帧 15.6 ms、9 帧 28.2 ms，约 1 GB/s」——毫秒数没错，
  **速率的分子错了**：`blake2b` 吃进去的是 `pixel_values` 与 int64 attention mask
  **两个数组**，而 mask 恰好占被哈希字节的 **40%**，所以真实体积是 25.0 / 45.0 MiB
  而不是 15.0 / 27.0 MiB，真实速率是 **1.49–1.58 GiB/s**。
  只算 `pixel_values` 会把一个 1 GB/s 的哈希器说成慢 40%。
- **模型身份检查（`Weights.fingerprint()`）每次查询缓存前都跑一次，这个代价是测出来的。**
  SmolVLM-256M 的 471 项 manifest 读一次 **0.0286–0.0295 ms**（三个独立进程的中位数），
  Qwen3-1.7B 的 310 项 / 6.4 GiB manifest 读一次 **0.0224 ms**——
  **字节多 6.7 倍反而更便宜**，因为代价跟 manifest 条目数走、不跟张量字节数走，
  这正是它在大模型上付得起的原因。对比它守的那次前向：同一张图的像素哈希是它的
  **538–995 倍**，而被跳过的那次视觉编码是 **2.43–3.52 s**。
- **替代方案（真的去 sha256 全部权重）也计价了，并且否掉了本节早先的一句断言。**
  那句是「哈希权重会比它想省掉的那次前向更贵」——**对 SmolVLM 是错的**：
  全权重哈希实测 **0.43–1.35 s**（区间宽是因为 mmap 页是否已驻留），
  只有该次编码的 **17.0%–38.4%**，即它并不比前向贵，只是仍然贵到每次请求都付不起。
  这句断言**只在 Qwen3-1.7B 上成立**（**7.907 s** 对 6563.5 MiB）。
  结论没变（不选它），**理由换了**：不是"比前向贵"，是"为保护一次前向而每次请求付
  几百毫秒到几秒"。代价与取舍写在 `vlm.py` 的 `_model_identity_changed` docstring 里。
- 新增 `EmbeddingEngine.index_documents()` → `DocumentIndex`：预编码文档后重复查询，
  `index.search(q)` 每次只编码 query。**原有 `search()` 一字未改、仍可用**，
  且因为两者都调 `rank_documents` / `encode_document`，结果 `np.array_equal` 逐位相同
  （不是 `allclose`）。进程内、不落盘、无 ANN、无 RAG。
  失效条件：模型对象、tokenizer 对象、`prefill_chunk_size`、`Weights.fingerprint()`
  中任一变化即拒绝复用。`prefill_chunk_size` 在表里是因为分块只保证 2e-4 等价、
  不保证逐位相同；`max_length` 与 `dimension` 不在表里，因为它们不改变已编码向量的值。
- 新增 `Weights.fingerprint()`：7 元组（解析后的根路径、format_version、model_family、
  task、compute_dtype、source_revision、按名排序的 (张量名, shape, dtype)），
  **只读 manifest 与 spec，不读权重数据**。这里有一个刻意留下的窄缺口并写进了文档：
  在已加载的引擎下覆写 `.npy`，fingerprint 不变——但 mmap 也仍指向旧内容，
  所以缓存与计算看到的是同一份数据。

### 优化 5：线程数与 prefill 分块参数——测了，默认值不动

这一项是**验证任务**，结论是一个有数据支撑的「不改」，不是没做。两个模型
（Qwen3-0.6B、SmolLM2-360M）× 7 个配置，**每个配置一个独立进程**，
优化前后两侧都扫（`artifacts/perf-round1/baseline/` 与 `sweep-opt/`）。

- **线程数**（不设 / 1 / 4 / 8）：opt 侧 512-token 分块 prefill 的极差是 2.4% 与 3.0%，
  decode 的极差是 0.6% 与 2.1%。**尚不能确认收益或回退。**请求值确实在 `import numpy`
  之前进了环境（harness 记 `requested_before_numpy_import: true`），但耗时对它没有反应；
  至于是 Accelerate 忽略了这些变量、还是这些矩阵形状本来就受内存带宽限制，本轮不做判断。
  **有效 BLAS 线程数在 Python 里不可观测**，harness 记为 `null` 而不是猜，
  所以上面任何结论都只针对「请求了这个数」。
- **`prefill_chunk_size`**：`c64`（+14.5% / +11.2%）与 `c512`（+2.3% / +10.0%）
  **明确更慢**；`c256`（-3.8% / -1.2%）按区间重叠判据属于**尚不能确认收益或回退**。
  没有任何一个非默认值落在「明确更快」那一档，所以 **128 不动**。
- **本条早先的「噪声底线」判读法已撤回。** 原来写的是「噪声底线是先测出来再用来判读的」
  「小于 3% 的差这一轮不能判定为收益」，并把线程极差说成「实测跨进程噪声底线的
  0.8 ~ 1.0 倍」。那个 3.0% 只有**两点偶然同配置的运行**（同一棵 opt 树在
  `ab/` 第 1 步与 `sweep-opt/*-t-default` 两个独立进程里的同一个探针）：
  Qwen3-0.6B 714.1 ms vs 693.5 ms（3.0%，进程内区间**互不重叠**），
  SmolLM2-360M 523.0 vs 520.9（0.4%，**重叠**）。两点数据没有分布，
  同一个探针在两个模型上的跨进程差就差了一个量级，取较大者当所有负载的统一门限没有依据；
  「两个模型的区间都不重叠」这句也是错的，数据里只有一个是。
  所以判读规则改成：**不再引用任何统一百分比**，凡是差值落在这个量级以内的一律写
  `尚不能确认收益或回退`。
- **要给出方向性结论需要真正的配对实验**：`bash tools/perf/run_noise_floor.sh <新的输出目录> 4`
  跑多轮独立进程配对、AB／BA 顺序均衡，并把进程内与进程间离散度**分开记录**，
  让每个负载用自己的多轮数据判断稳定性。**本轮没有跑它。**
- `DEFAULT_PREFILL_CHUNK_SIZE = 128`（`embedding.py:54`、`engine.py:45`）未改，
  `config.py` 与 baseline 树**逐字节相同**，`src/` 下没有任何 `os.environ[...] =` 写入
  ——三项都在源码里核对过，不是靠记忆。

### 计时与文档

- `prefill_seconds` 只计**本次请求实际跑过的** chunk；`prefill_tokens_per_second`
  除以 `prefill_tokens` 而不是 `prompt_tokens`，所以一个复用 90% 的轮次
  不会报出 10 倍虚高的吞吐。`report()` 新增 `KV reuse 128/142 cached (90%), 14 prefilled`
  一行，仅在有复用时出现。
- README、`docs/architecture.md`（§4 §5 §6.1 §7.1 §8.5 §9）、`docs/benchmarks.md`
  同步更新。`benchmarks.md` 顶部加了明确横幅：那批数字是**第 1 轮之前**的历史基准，
  不再描述当前代码；并修正了两处会误导的地方——它那个"3 轮取中位数"的图文测法
  现在会命中特征缓存（复现下表必须 `image_cache_bytes=0`），以及分块 vs 整块
  prefill 的比较**成因**已经变了（两边现在都只投影 1 次）。

### 测试

- 快速测试 685 → **826**（+141），收集总数 747 → **888**。逐文件增量：
  `test_session.py` +52（新文件）、`test_model.py` +25、`test_vlm.py` +19、
  `test_ops.py` +15、`test_embedding.py` +11、`test_cli.py` +6、`test_cache.py` +6、
  `test_weights.py` +5、`test_benchmark.py` +2，其余 7 个文件 0。
  新增缓存重点覆盖命中、未命中、失效、取消与边界。
- 8 个模型的真实权重集成测试重跑两次（优化 1+2 后、优化 1~4 后），
  **两次都是 208 passed、0 failed**，逐目录结果一致。
- 8 个模型的官方参考验证**重新抓取 SlowQ 侧再比较**（`ref` 侧未改，不重跑）：
  8 / 8 退出码 0，860 个张量全部 PASS，容差一项没动。
  比"仍然 PASS"更强的是：8 份新 `compare.log` 与备份的 pre-round1 版本
  **逐字节相同**（其中 2 份只差 `[compare] took` 计时行，0 处数值差异），
  即每个张量的实测误差都没有变。最差 rel_L2 2.719e-05（门限 1e-3）。

### 实验后回退 / 否决的改动

按用户要求，没有收益或有回退的实验**回退实现并留下证据**，不留在代码里：

- **softmax 三临时数组合并为单缓冲区**：实测 ±1%，落在同一批数据的运行间波动内，
  即分配不是这个函数的成本。已回退，理由写在 `softmax` 的 docstring 里，
  连同"不得写入入参"这条契约。
- **`Weights.expect` 结果记忆化**：实测无收益，否决。
- **memmap 张量预先 `.T`**：实测无收益，否决。

### 审查修复（第 1 轮审查后，2026-10-26）

审查提出的三个正确性问题与一个状态问题，**每一项都先补一个会失败的测试，再改实现**。
下面每条都列出了那个测试的名字与所在文件，以及**改坏实现后重跑它应当变红**的命令。
这四条的红跑**已在性能采集结束之后补做完毕**（采集期间故意不跑，以免与计时窗口抢核），
日志在 `artifacts/perf-round1-rerun/falsification/`，每条都是「改坏实现 → 对应测试变红 →
还原 → 全套快速测试回到 991 passed」：

| 审查项 | 改坏的点 | 红跑日志 | 变红 / 总数 |
| --- | --- | --- | --- |
| 1a `DocumentIndex` 泄露 | `detached()` 改成恒等（不再拷贝） | `review1a-embedding-detached-identity.txt` | 5 / 6 |
| 1b 换模型后命中旧缓存 | `_model_identity_changed()` 永不作废 | `review1b-vlm-identity-noop.txt` | 3 / 5 |
| 1c `prefill()` 静默忽略 | 去掉整块前置校验，退回逐 chunk | `review1c-prefill-no-upfront-check.txt` | 12 / 13 |
| 2 流式完成状态丢失 | `turns` 自增移回生成器耗尽之后 | `review2-session-count-at-exhaustion.txt` | 3 / 5 |

「变红 / 总数」里没红的那几项是**故意**的：1a 的 `vectors` 走 `np.stack` 本就是新缓冲区、
1b 的探针路径与身份处理正交、2 的「正常耗尽」与「中途抛异常」两条恰好在耗尽后才自增的
错误版本下也对——它们约束的是修复**不得过头**（例如「换模型 ⇒ 永不缓存」），
不是复现 bug 本身，各自的 docstring 里都写明了这一点。
本轮**没有动任何数值容差**，没有引入 GPU 后端、量化、推测解码或新模型支持。

- **`DocumentIndex` 泄露内部可写向量**（`embedding.py`）。`EmbeddingResult` 是 frozen
  dataclass，冻结的是**属性绑定**不是它指向的数组：`result.vector` 不能重新赋值，
  `result.vector[:] = 0` 照样写穿。三条泄露路径（构造时传入的 results、`results`
  属性、`search()` 返回的 batch）共用一个 `detached()` 在两端各拷一次——
  只在一端拷会把另一端留着。失败症状是**分数看起来仍然像余弦相似度，只是不再有意义**，
  同时 `EmbeddingEngine.search()`（重新编码）仍给出正确答案，两边不一致。
  测试：`test_mutating_the_arrays_handed_to_the_constructor_does_not_reach_the_index`、
  `test_mutating_a_vector_read_from_results_does_not_reach_the_index`、
  `test_mutating_a_vector_returned_by_search_does_not_reach_the_index`、
  `test_a_mutated_index_still_agrees_with_a_fresh_search`、
  `test_the_vector_matrix_is_a_copy_the_caller_cannot_write_through`（`test_embedding.py`）。
  代价写在 `detached` 的 docstring 里：每文档 `hidden_size` 个 float32，
  一千个 Qwen3-Embedding-0.6B 文档 4 MiB，且拷贝在索引 A/B 的计时窗口**之内**。
- **换掉模型后仍命中旧图片特征缓存**（`vlm.py`）。缓存身份里原本只有
  `Weights.fingerprint()`，而 fingerprint 在引擎被替换成另一个模型对象时不会自动重算。
  修法是 `VisionEngine._model_identity_changed()`：**每次查缓存前**比对当前模型对象身份，
  变了就整表作废。它**不扫描权重**——审查明确要求不能为了检测身份而每次请求遍历整套
  权重，实测那样做在 SmolVLM-256M 上是 0.43–1.35 s、在 Qwen3-1.7B 上是 7.9 s
  （见优化 4 那两条）。测试用**真实模型对象替换**而不是手改私有 fingerprint：
  `test_swapping_the_model_does_not_serve_the_previous_models_features`、
  `test_a_swapped_engine_rewarms_and_then_hits_again`、
  `test_a_swap_drops_the_unreachable_entries`、
  `test_the_identity_check_reads_the_manifest_and_no_tensor`、
  `test_a_probe_still_bypasses_the_cache_after_a_swap`（`test_vlm.py`）。
- **`prefill()` 静默忽略不匹配的 `inputs_embeds`**（`model.py`）。原来校验发生在
  `_merge_inputs_embeds` 内部，也就是**逐 chunk**、在前几个 chunk 已经把 KV 写进去之后；
  一个形状不对的 features 会让缓存处于半提交状态，报错信息也说不清是哪一侧不对。
  现在 `prefill()` 在**任何 KV 写入之前**用 `_checked_inputs_embeds` 校验整块输入，
  `forward()` 走同一个校验，两条路径对同一个错误块给出**同一个**报错。
  测试：`test_prefill_refuses_fewer_rows_before_the_earlier_chunks_commit`、
  `test_prefill_and_forward_refuse_the_same_block_identically`、
  `test_a_rejected_prefill_leaves_an_existing_cache_exactly_as_it_was`、
  `test_prefill_refuses_features_that_are_not_a_row_matrix`、
  `test_prefill_refuses_rows_nobody_asked_for`（`test_model.py`，后两个按形状参数化）。
  `test_a_rejected_prefill_leaves_an_existing_cache_exactly_as_it_was` 逐字节比对
  拒绝前后的 `k` / `v` 与 `_committed`，把"没有半提交"变成可检验的断言。
- **流式完成状态在调用方提前 close 时丢失**（`engine.py` 的 `Session._run`）。
  `turns` 原来在生成器**耗尽之后**才自增，而 `yield from` 只有在生成器正常结束时才返回：
  调用方在最后一步之后 `close()`（或用完即弃）会让 `GeneratorExit` 抛在 `yield` 上，
  于是一个**已经完成并交付**的请求被报成 `turns=0`，下一轮的前缀复用也就少了它。
  自增现在紧贴在最后一步 `yield` **之前**——交付即计数。
  测试：`test_a_stream_closed_on_its_final_step_still_counts_the_turn`、
  `test_a_stream_closed_after_an_eos_step_still_counts_the_turn`、
  `test_exhausting_a_stream_counts_the_turn_exactly_once`、
  `test_a_decode_that_raises_mid_stream_does_not_count_the_turn`、
  `test_the_completed_turn_count_ignores_cancelled_requests`、
  `test_a_turn_counted_at_delivery_does_not_outrun_the_reuse_it_reports`（`test_session.py`）。
  最后一条守的是反向错误：计数不能跑到它所报告的复用前面去。
- **性能方法学与归因更正**见优化 2、优化 5 与「尚未验证」三节：撤回的数字都写明了
  撤回理由并**保留原始证据**，`artifacts/perf-round1/` 一个字未改。
- **实验工具进版本管理**：本轮用的探针与驱动全部落在 `tools/perf/`（受 Git 跟踪），
  每份 JSON 记录采集时的源码身份，基线树可从 `8968e39` 重建
  （`artifacts/perf-round1/baseline-tree/` 带 `BASELINE-REVISION` 标签）。
  工具本身有回归测试：`tests/test_perf_tooling.py`（135 项），
  包括"按路径 import 不得解析 `sys.argv`"这条——否则 pytest 加载它时会吃掉自己的参数。
- **快速测试 826 → 995（+169），收集总数 888 → 1057。**
  `.venv/bin/python -m pytest -q -m "not integration"` 在最终树上跑出
  **995 passed, 62 deselected**，落盘在 `artifacts/perf-round1-rerun/quick-fixed.log`
  （2026-10-27 06:01 重跑；前两代日志留在同目录 `quick-fixed.log.prev-994` 与
  `quick-fixed.log.prev-991`，10-26 16:17 那次 953 passed 只在终端、未落盘，已被取代）。
  新文件是 `test_perf_tooling.py`（135 项，即上一条），其余增量落在
  `test_embedding.py` / `test_vlm.py` / `test_model.py` / `test_session.py`
  这四处修复各自的测试上。62 项集成测试仍因未设 `SLOWQ_MODEL_DIR` 被 deselect。
  **原「两项待补」均已完成**：逐文件增量已用 `--collect-only -q` 在最终树上重算
  （2026-10-27，故意留到性能采集结束之后跑，避免与计时窗口抢核），
  `docs/validation.md` §快速测试 的增量表逐文件对得上 826 → 995；
  那次输出也已落盘到上面那个日志路径。

### 报告生成器派生化修正（2026-10-27）

`docs/推理加速-第1轮-2026-10-23.md` 整篇由 `tools/perf/make_report.py` 从原始 JSON 生成。
审查发现它有一批**硬编码的旧结论**与**算出来却没读的值**：数字写死在脚本里，
换了数据目录也不变，于是文档印着的判定与它引用的那份原始数据对不上。本轮把它们逐个
改成从数据派生。下面**前七条**是带派生值的修正，每条都按「把派生值改回硬编码、
对应测试必须变红」做了反证，红跑记录在 `artifacts/perf-round1-rerun/falsification/`；
**最后两条**（§4 禁止推断 / §9.4 chat 行有测试，其余描述性口径没有）证据强度不同，
已在条目内写明。新增的回归测试都在 `tests/test_perf_tooling.py`（写这节时 131 项；
现 135 项，多出的 3 项属于下一节的 `run_reference.sh` 修正、第 4 项属于再下一节
§2.2 那句逐字节断言的更正）。

- **§0「先说怎么读」的两个例子原来是手抄的。** 开头写"Qwen3-1.7B 端到端先后跑出
  0.59× 和 1.29×""集成测试先后 115 s 和 3828 s"，都取自已发布那批；从重采目录生成时
  §3 的表里已经没有那一对、§9.4 的耗时也长成三个窗口，文档却仍用旧数字开场。
  现在 §0 的端到端倍率取自 `contradiction_rows` 里最宽的一行、集成耗时取自
  `integration_durations` 的全部窗口，并写明数据来自哪个根目录。
  测试：`test_the_opening_discipline_quotes_what_the_evidence_sections_contain`。
- **集成耗时的「差 N 倍」原来随 `--data` 变。** 最快的 57.17 s 那个窗口只存在于重采根
  `perf-round1-rerun/integration-fixed/logs/`，而 `integration_durations` 只扫 `PERF`，
  于是从已发布根生成报 33×（两个窗口）、从重采根生成报 67×（三个窗口），同一轮两个答案。
  改成默认扫 `(REPRO_ROOT, PERF, DEFAULT_PERF)` 三个根、按路径去重，稳定取到三个窗口。
  测试：`test_the_integration_spread_covers_every_log_on_disk`。
- **§3 跨遍自检列算了三项比值却只印两项、还印成 2 位。** 单元格里印 `base 1.06 / opt 0.82`
  再给一个"1.31×"的判定，读者拿印出来的两个数怎么除都得不到 1.31。现在把原始比值提到
  3 位、并把判据真正用的三项比值（`ab/` 内部、`ab-longctx/` 内部、跨遍 opt）逐项印出来，
  单元格自己的算术能复现自己的判定；汇总段也改成分别报「三项一致 / 只是单批内部对不上 /
  连跨遍都对不上」三类行数，而不是笼统一句。
  测试：`test_the_cross_check_column_reproduces_its_own_verdict`。
- **§1.5 只核对了*被测*的树，没核对*写结论*的脚本。** 文档每个数字都出自 `write_doc`，
  而它在审查期间被反复改过，同一批 JSON 用两版脚本会给出不同措辞，读者无从判断手里这份
  是哪版生成的。现在 §1.5 开头印出 `make_report.py` 自身的 sha256、行数与
  `shasum -a 256` 复核命令（这段放在两个提前 return 之前，否则已发布根那种没有身份块的
  渲染恰好拿不到署名）。测试：`test_the_report_names_the_version_of_the_script_that_wrote_it`。
- **§1.5 的 digest 可复现性原来是一句手写的断言，而且两版都写错过。** 一版说"上表的
  digest 今天一个都复现不出来"——错：基线树是 `artifacts/` 下的 `git archive` 快照、
  `tooling` 那一半为空，本轮的编辑碰不到它，`source_identity.py --digest-only` 仍原样返回
  它的 digest。另一版说"变过的只是把 JSON 写成中文段落的那个文件、不参与计时"——也错：
  有一棵树的改动集里含 `projection_probe.py`，那正是采集它的探针。现在按每棵树重新哈希
  记录的 `source` 与 `tooling` 两半，分成「仍可复现 / 只动了工具 / 动了采集脚本本身 /
  动了被测代码」四类，并且「harness 一个字没变」这句只对*确实*没有 harness 变动的那一类说；
  harness 的检查覆盖该树**全部**采集脚本，不止代表块里那一个。
  测试：`test_the_digest_paragraph_classifies_each_tree_by_what_actually_moved`。
- **`make_report.partial_digest` 是 `source_identity.digest` 的副本，用测试钉住。**
  `tools/perf/` 下的脚本在测试里按路径加载、彼此不能 import，所以 `make_report` 只能自带
  一份 digest 计算。副本没有任何东西拴住原件就会悄悄漂移：§1.5 会继续印一个不再等于
  `--digest-only` 的"digest"。现在对磁盘上每一份身份块都要求
  `partial_digest(block, "source", "tooling")` 同时等于记录的 digest 与原件算出的 digest。
  测试：`test_the_partial_digest_mirrors_the_tool_it_copies`。
- **§10 复现块的「期望」与它自己的命令对不上，且命令缺必填报径。** 期望值手抄自已发布轮
  （"826 passed"），套件早已长大，照着复现的人看到的是被文档判为"错"的正确数字；期望还引用
  `--data` 恰好带到的那个日志目录，与上一行命令实际写入的目录不是同一个。`run_integration.sh`
  与 `run_reference.sh` 都用 `${1:?...}` 要求显式日志目录，文档里的命令却没给，照抄会立刻退出。
  现在期望值从落盘日志派生、命令补齐必填报径、并指向命令真正写入的目录。
  测试：`test_the_reproduction_block_expects_what_its_own_commands_produce`。
  该测试自身有个 `tee` 竞争要两步收敛：§10 第 1 步把套件输出 `tee` 进同一个日志，
  逐字照跑会让测试读到正被 `tee` 截断的文件，所以先跑到临时文件、拷进去、重生成、再跑一遍。
- **§4 那句「补跑更快 ⇒ 不是回退」的禁止推断、以及 §9.4 的 chat 行，用测试钉住。**
  两处描述同一张表，必须对得上：§9.4 原来写"四行读起来像回退（0.85× / 1.04× /
  0.78× / 0.63×）"，是从已发布那批手抄的——重采那批只有一行、连已发布那批也是三行
  （1.04× 不是回退，被误算进去了）；而 §4 又把"补跑更快"当成"旧结果不是回退"的证据，
  这正是审查点名要撤回的推断。现在行数从 `headline_numbers()["chat_cold_slower"]` 派生，
  撤回的推断只允许作为**被引用后反驳**出现、不能作为报告自己的主张留存。
  测试：`test_the_pending_row_quotes_the_chat_table_it_points_at`
  （红跑：`retraction-restated.txt`）。
- 另有几处同源的**描述性口径**一并更正：撤回的 sigmoid 探针路径、sweep 基线缺口注记、
  清单结构描述、sigmoid 探针仅按 mtime 排序的告警。这些是散文表述、没有各自的派生值测试，
  **未做逐条反证**；它们的正确性靠「重新渲染文档 + 全套快速测试通过」保证，
  与上面七条（每条都有把派生值改回硬编码、对应测试变红的红跑记录，
  全部在 `artifacts/perf-round1-rerun/falsification/`）不是同一种证据强度。

### 二次审查修复（第 2 轮审查后，2026-10-28）

四个已确认问题。**没有提交、没有推送**，全部改动留在工作区。

#### 问题 1：复现脚本会覆盖已发布的原始证据

- 第 1 轮的驱动脚本默认输出目录就是 `artifacts/perf-round1-rerun`——**已发布文档
  正在引用的那一份测量**。而它们只保护 `artifacts/perf-round1`，且用的是字符串比较：
  `if [ "$OUT_ROOT" = "$LEGACY" ]`。`artifacts/perf-round1/`（带尾斜杠）不等于那个
  字符串，于是直接走过去；而每个测量程序结尾都是
  `out.parent.mkdir(parents=True, exist_ok=True); out.write_text(...)`，
  **按构造就是静默覆盖**。`run_projection.sh` 是同一个缺陷穿了件外衣：
  它把上一次的尝试 `mv` 进归档目录——字节保住了，但那个本该冲突的命令照样成功。
- 新增 `tools/perf/out_guard.py` + `out_guard.sh`：**一条规则，没有逃生口——
  已存在的路径就是冲突，冲突就是「在任何昂贵测量开始之前」的非零退出**。
  **故意没有 `--force`。** 重测永远可行，只是要测到新目录里去（`fresh_root`），
  而新目录才是让两次运行可比的东西；覆盖只会让其中一次不可验证。
- 两层，因为一层不够：驱动脚本开头调 `reserve`（一小时测量之前发现冲突，
  并且一个驱动自己的全部产物一起检查——半建起来的树是另一种烂摊子）；
  测量程序在**写入那一刻**调 `write_json`，以 `"x"` 打开，
  所以即使中间有别的进程建了那个文件，拒绝仍然成立。
- 路径先过 `realpath` 再比较，所以尾斜杠、`..`、相对路径、绝对路径、符号链接
  都落到同一个答案上。目标位于已发布证据之内的符号链接由**目标**抓住，
  悬空符号链接由 `lexists` 抓住那个被敲进来的名字。
- **编排脚本可以共享一个新建的运行根目录**：`reserve` 接受一个**已存在的空目录**
  （它不含结果、不含状态、不含日志），所以 `mkdir -p` 式的调用继续可用，
  规则没有被削弱。每个驱动只 reserve **自己的子目录**，从不 reserve 交给它的根。
- 三个驱动（`run_integration.sh` / `run_recollection.sh` / `run_reference.sh`）的
  输出目录参数现在都是**可选**的：不给就创建一个唯一的新目录并把路径打到 stderr。
  脚本**不会**通过删除、移动或重命名旧证据来让自己成功。
  所有会把读者引向旧证据目录的复现命令一并改掉（`tools/perf/README.md`、
  本文档、`docs/validation.md`、报告 §10）。
- **必要关联项：报告 §10 的「期望值」原先只从已发布根里读，堵住写入端就等于把它冻住。**
  `quick_suite_expectation()` / `integration_totals()` / `integration_durations()`
  三个函数的默认根都是 `(REPRO_ROOT, PERF, DEFAULT_PERF)`，而 `REPRO_ROOT`
  就是 `artifacts/perf-round1-rerun`——本轮开始拒绝写入的那个根。
  这三处存在的理由恰恰是「§10 印的必须是**当前树**跑出来的数」（它就是为了取代
  手打的 `826 passed` 才加的），可新日志既然不能写进已发布根，
  它们就永远只能读到旧计数：**修好问题 1 反而把这个缺陷变成永久的**。
  改成 `repro_roots()`：`artifacts/perf-run-*` 按名字倒序（名字就是
  `out_guard.fresh_root` 铸的定宽 `%Y%m%d-%H%M%S`，所以字典序即时间序，
  不看 mtime——事后拷进去的日志不该压过真正更新的那次运行）排在最前，
  `REPRO_ROOT` 保留为兜底。`INTEGRATION_LOG_DIRS` 同时补上本轮实际用的
  `integration-round2/`，否则那 8 份日志一份也找不到。
- **§10 的期望行现在各自点名它读的那份文件**，不再统一挂在一个硬编码的根上：
  `quick_suite_expectation()` 的返回值从 `str` 变成 `(summary, source)`，
  集成合计本来就已经带着 `totals[2]`。两个期望可以来自两个不同的运行根，
  而「一个数印在它没读过的目录旁边」正是这三个函数要防的事。
  连带的影响是 §9.4 的窗口数从 3 涨到 4：本轮 Qwen3-1.7B 的 3935.49 s
  落进 `artifacts/perf-run-20261028-185623/integration-round2/`，
  四个窗口 57.17／115.37／3828.08／3935.49 s，最慢与最快差 **69 倍**
  （原先三个窗口是 67 倍）。镜像这条断言的
  `test_the_integration_spread_covers_every_log_on_disk` 跟着改成扫
  `repro_roots()`、要求 4 个窗口，并额外要求其中一个来自 `perf-run-*`——
  少掉的那个窗口恰好就是修复之后落进慢档的那一个。
- **顺带查清了一件事：`pytest` 自己就会重写两个已发布根里的 `report-tables.md`。**
  本轮做证据溯源时发现 `artifacts/perf-round1/report-tables.md` 的 mtime 是当天，
  追下去是 `tests/test_perf_tooling.py` 里 8 个按两个根参数化的测试
  （7 个用 `AB_DIRS`，1 个用 `NOISE_FLOOR_DIRS`，当前两者都是这两个根）
  各自调 `report.main(["--data", root])`：它们把 `report.DOC` 指到 `tmp_path`，
  却没有指 `report.OUT`，而 `main()` 自己会执行 `OUT = PERF / "report-tables.md"`。
  一次完整快速测试把 `perf-round1/` 那份写 8 遍、`perf-round1-rerun/` 那份写 9 遍。
  **不改**：它是派生产物（文件头就写着「由 make_report.py 生成，不要手改」）、
  可以从同一批 JSON 重算、且被 `.gitignore` 忽略，而 `out_guard` 挡的是
  测量结果／历史状态／日志三类；挡住它等于挡住唯一能证明表格没被手改过的命令。
  但 `tools/perf/README.md` 里那句「这是已发布根目录里唯一会被改写的文件」
  原先只点了 `perf-round1-rerun` 一个根、也没说是谁在写，已按上面的事实改正，
  并明确写上**别把这两个文件的 mtime 当作「这轮有没有重新测量」的证据**。
  真正的判据是旁边那些 JSON 与日志：`artifacts/perf-round1-rerun/noise-floor/`
  里 40 份 JSON 最新 mtime 仍是 **2026-10-26 19:06**。
- **同类缺陷还有一处，本轮只改了文档、没有改代码，明确记在这里。**
  `tools/benchmark.py` 结尾是 `output.write_text(...)`，同样**没有接 `out_guard`**，
  而 `docs/benchmarks.md` 原先给的复现命令正是
  `--json artifacts/benchmark/v0.2-<name>-slowq.json`——那 8 份文件是
  `docs/benchmarks.md` 全文引用的已发布测量结果，照着命令重跑就会**原地换掉它们**
  且事后无法察觉。改法：文档里的命令改成先 `out_guard.py fresh-root` 再写进新目录，
  并在旁边写清这个工具仍然会覆盖 `--json` 指到的任何文件。
  **没有给 `tools/benchmark.py` 加防护**：它不属于本轮四个问题中的任何一个，
  加防护要让它跨目录引 `tools/perf/out_guard.py`，属于 一.4 要求避免的范围扩张。
  代价是「不覆盖已发布结果」在这条路径上暂时由命令自己负责，不是由工具保证。

#### 问题 2：视觉缓存遗漏「相同权重、不同配置」的模型替换

- 上一轮加的身份检查只比 `Weights.fingerprint()`，**而 fingerprint 描述的是
  manifest**——哪个目录、哪种格式、哪张张量表。所以在**同一个 `Weights` 对象**上
  用不同配置建出来的两个模型，fingerprint **完全相同**：
  `dataclasses.replace` 改一个合法的计算参数（`vision.layer_norm_eps`）或
  预处理参数（`processor.image_mean`），在同一份权重上重建模型、绑到 `engine.model`，
  旧条目就被交给新塔的调用方。**形状对得上、行数对得上、什么都不报错，
  请求正常返回，而它是自信地错。**
- 新增 `_vision_identity(model)`，一个**五元组**：`Weights.fingerprint()` +
  `config.vision` + `config.processor` + `config.image_token_id` + `config.scale_factor`。
  五项都可哈希、都便宜：fingerprint 只读 manifest，配置对象是标量与 float 元组
  组成的 frozen dataclass，比两个配置是一次字段遍历而不是一次计算。
- `_model_identity_changed()` 比**两样**东西，两样都需要：
  ① **模型对象**，用 weak reference 而不是 `id()`——id 会在对象被回收后复用，
  一个复用来的 id 会和它替换掉的那个模型比较相等，`is` 骗不了，
  而死引用返回 `None`，那也不是当前模型；② 上面那个**值身份**。
  ② 正是 ① 覆盖不到的部分。任一样变了就整表作废、丢掉所有不可达条目。
- **不受支持的替换明确拒绝，不静默返回旧结果**：绑一个没有 `vlm_config` 的对象
  （文本模型）让 `_vision_identity()` 抛 `TypeError`，点名「支持的是
  `SmolVLMModel` 实例」。不是从刷新逻辑深处抛 `AttributeError` 说某个字段不存在，
  也不是留着旧配置、继续用一个新模型根本没有的视觉塔算出来的特征。
  用 `TypeError` 而不是 `slowq.errors` 里的类型：这是调用方绑错了对象的种类，
  不是坏 checkpoint、也不是坏请求。
- **没有**通过每次请求扫描权重内容来解决身份问题：实测整个检查
  （那次 fingerprint、建配置元组、一次弱引用解引用、一次比较）是
  **0.0326–0.0331 ms**，即它包着的裸 fingerprint 的 **1.084–1.091 倍**；
  200 次调用里 `changed_count` 恒为 **0**。正常命中与 probe 双向绕过缓存的行为都保留。
- 覆盖：同权重不同 vision 配置、同权重不同预处理配置、不同权重（上一轮已有）、
  替换后首次重算再命中、未替换时持续命中、命中路径不扫描权重张量、
  不支持的替换被拒。**全部用真实模型对象替换**，不是手动改私有 fingerprint 来模拟；
  「同权重不同 vision 配置」那条还拿 `other.encode_image(image).features`
  与缓存给出的数组做 `assert_array_equal`，即**与当前模型直接计算逐位一致**。
- **反证时发现自己写错了一次，改法记在这里，因为它说的是测试的强度而不是修复的强度。**
  第一次只把五元组回退成 `(model.weights.fingerprint(),)`，
  对象那一条检查留着，结果 **12 项全绿**——那不是「修复不必要」，
  是**回退没有回到修复前的状态**：相关测试全部写 `engine.model = other`（换对象），
  ① 单独就抓住了它们。补了两个对照之后结论是精确的：
  **① 与 ② 各自都足以让这 12 项变红，只有同时回退到「只比指纹」才有 4 项变红**
  （三份日志 `p2a-object-clause-only` / `p2b-tuple-only` / `p2c-both-halves`，
  前两份 12 passed、第三份 4 failed 8 passed，都在
  `artifacts/perf-round2-falsification-20261028-233602/`）。
- **由此暴露一处真实的覆盖缺口，本轮没有补，明确记下来**：
  ② 管的是**对象没换、配置被就地改掉**的情形（`engine.model.vlm_config` 原地改字段），
  ① 看不见它；而**现有测试没有单独钉住 ②**——12 项里没有一项是「同一个对象、
  就地改配置」。要钉住它得构造一个就地改 frozen dataclass 字段的夹具，
  超出四个问题的范围，所以只回退 ② 会变绿这件事被写成限制而不是被藏起来。
  变红的 4 项与另外 8 项的分工也一并写清了：那 8 项在修复前后都绿，
  它们是上一轮「换权重」那批覆盖，不属于本轮问题 2。

#### 问题 3：剔除异常轮次后没有重算 AB/BA 均衡，却计入了「稳定收益」

- 审查者从 `artifacts/perf-round1-rerun/noise-floor/` 的原始 JSON 重算：
  有 **6 行**在污染筛选之后只保住 r2/r3/r4，即 **1 个 AB + 2 个 BA**，
  而报告印的是「顺序均衡：是」，其中 **5 行**还计入了那个「70/73 达到稳定收益判据」。
  根因是均衡只在**全部**采集轮次上算过一次，筛选之后没有重算。
- 现在每一行都从**最终保留下来的轮次**重算有效样本数与顺序均衡，
  并新增两个判不定档位的标签，**都不计入方向、也不计入任何「稳定收益」汇总**：
  `INSUFFICIENT`（`有效轮次不足`，保留轮次 < `MIN_RETAINED_ROUNDS = 2`）、
  `UNORDERED`（`顺序不均衡或未记录`）。**先判轮次够不够，再判均衡**——
  一轮在算术上也是不均衡的（`1 * 2 != 2`），先问均衡会把一次幸存的测量
  报成驱动脚本的 bug，而两个标签指向的是两种不同的补救：加轮次，还是修 harness。
- `_order_balance()` 对**空轮次集**返回 `None`：`0 * 2 == 0` 为真，
  没有这道保护，一行被筛选剥光的行会对着它根本没有的轮次印「顺序均衡：是」。
  表里印 `**无保留轮次**`。位置信息缺失或非法（`None`、非数字）同样归到 `None`，
  印 `**未记录**`——「没记录」既不是「是」也不是「否」，
  从文件名顺序去猜正是会掩盖驱动 bug 的那种做法。
- **「测得稳定收益」这个档位被取消**，只剩「有限轮次方向一致」。
  2–4 轮同向不支持「稳定」；判据在采集前就已声明，表里同时印有效轮次与不确定度。
- **表格、明细、合计、正文与交付报告出自同一组判定**：`PAIRED_SUMMARY` 的
  `verdicts` 与逐行印出来的结论由同一处累加，`tests/test_perf_tooling.py` 里
  `test_the_summary_counts_are_the_verdicts_the_table_printed` 把一份混合伪数据的
  **每一行**从渲染出来的 markdown 里数回去、再和正文引用的汇总比——
  「70/73」之所以能活下来，正是因为这两处曾经各算各的。
- **正负方向对称**：`UNORDERED` 的判定不看比率朝哪边指。
  `test_an_unbalanced_slowdown_is_excluded_the_same_way_a_speedup_is` 用同样的
  轮次结构造一行 opt 更快、一行 opt 更慢，两行都必须被挡；
  否则一条只针对 `gain > 1` 写的规则会**制造出**它所报告的那个回退。
- 被剔除的轮次与剔除理由都保留，原始数据一份未删。
  **没有**按收益方向挑轮次、**没有**放宽异常阈值、**没有**改标签来保住「70」。
- 新增 `--paired-data`：让 §7.3 读**另一个**运行根目录下的 `noise-floor/`，
  而**只**影响这一张表。默认 `None`，所以普通的 `--data <root>` 调用
  照旧读 `<root>/noise-floor/`，其他任何一节的出处都没有动。
  重测单独占一个目录，是因为已发布的那一份是撤回结论背后的原始证据，
  重测只能**加**在它旁边；表里与 §7.1 的出处行都会点名自己读的是哪个根。
- **配对重测跑完了，结果按它自己给的判据判，没有挑。**
  `bash tools/perf/run_noise_floor.sh artifacts/perf-round2-paired 6`：
  5 个模型 × 6 轮 × base/opt = **60 个独立进程**，`NOISE_RC=0`，
  每轮两个进程的 `label` 都带 `-pos1`／`-pos2`，5 个模型各 3 轮 base 先跑、
  交替模式 `B O B O B O`——**采集时的均衡是逐份 JSON 核出来的，不是从驱动设计推的**。
  判读：73 个（模型 × 量）里 **64 个「有限轮次方向一致」**、
  **5 个 `尚不能确认收益或回退`**（逐轮增益跨过 1.00，全在 Qwen3-1.7B）、
  **4 个 `顺序不均衡或未记录`**、**0 个 `有效轮次不足`**。
  **被撤回的「70/73 达到稳定收益判据」由「64/73 达到有限轮次方向一致」取代**——
  分子小 6，档位也换了一个更弱的词，两个方向都不是为了让数字好看。
- **那 4 个 `顺序不均衡` 正是这个 bug 会长出来的样子，只是换了模型。**
  它们采集时 6 轮均衡，被 spike／level 判据剔掉 1–2 轮之后变成 3/5 或 3/4，
  旧代码会对它们印「顺序均衡：是」并把增益计入分子；现在一律不判方向。
  上一轮那 6 行是 Qwen3-1.7B 与 SmolLM2-360M，这一轮落在 Qwen3-1.7B 的
  端到端与三个 prefill 量上——**说明这不是某一次运行的偶然，是筛选与均衡之间的
  结构性交互**，也就是为什么修复必须在筛选之后重算，而不是多跑几轮就完事。
- **轮数区间与那个抛硬币的 p 值改成从数据派生。**
  §9.1 原先硬写「每行只有 2–4 轮独立进程」，而 §7.3 从保留轮次算出来的是 4–6；
  同一份报告里两个数打架，正是问题 3 说的「表格、明细、合计、正文必须出自同一组判定」。
  现在 `PAIRED_SUMMARY["round_span"]` 由 §7.3 那次累加填，§9.1 读它；
  「4 轮全部同向在抛硬币下本就有可能」后面也补上了它自己的 `p = 2^-4 = 0.0625`。
  §7.3 里另一处硬写的「`1.01–1.04×`」例子（真实行已是 `1.02–1.04×`）
  改成指向 `逐轮增益`／`进程间` 两列本身，不再嵌数字——
  嵌进去的数字下一次重测就会再次变成假的。

#### 问题 4：验证失败却返回成功退出码

- `run_integration.sh` 与 `run_recollection.sh` 都是**每个模型都跑完**
  （一趟就知道八个里哪几个失败，这正是它的用处），
  但**累计的失败就是脚本的退出码**。原先印一行 `OVERALL_RC=1` 然后 `exit 0`，
  把结论留在状态行里、把成功留给退出码——编排脚本与 CI 读的是后者。
- **解释器缺失也算失败**：`rc=127`（`.venv/bin/python` 不存在）在 pytest 启动之前
  就退出，原先会被当成「没有模型失败」。现在 `[ $rc -ne 0 ] && FAIL=1`。
- `tools/perf/README.md` 新增一段「**退出码就是结论**」，覆盖三个驱动与 `rc=127`。

#### 本轮的身份代价数字换了批次

`artifacts/probe-rerun-20261028/identity/`（SmolVLM-256M 三个独立进程 +
Qwen3-1.7B 一个）的 `vlm.py` AST 哈希 `e4a22324…` **与当前源码逐字符相同**，
所以本节的数字就是这份代码的代价：fingerprint **0.0301–0.0304 ms**（Qwen3-1.7B
**0.0238 ms**）、全权重 sha256 **0.395–1.62 s**（Qwen3-1.7B **8.489 s**）、
视觉编码 **2.53–4.09 s**、像素哈希 **5 帧 16.25–16.62 ms / 9 帧 28.24–29.65 ms**
（**1.47–1.56 GiB/s**）。

上面「优化 4」那几条引用的 `artifacts/perf-round1-rerun/identity/`
（`vlm.py` 还是 `77167e65…`）给的是 fingerprint 0.0286–0.0295 ms / 0.0224 ms、
全权重 0.43–1.35 s / 7.907 s、视觉编码 2.43–3.52 s、像素哈希 15.41–16.33 ms /
27.89–29.35 ms（1.49–1.58 GiB/s）。**哪些还能用**：`fingerprint()` 住在
`weights.py`，而 `weights.py` 的 AST 哈希在两批之间**没有变**（`b041e6f6…`），
所以那一列仍然描述当前代码；**哪些只能当旁证**：视觉编码与像素哈希走 `vlm.py`，
本轮 `vlm.py` 变了，那一批不能当作当前代码的数值。六个进程合起来的中位数区间是
**0.0286–0.0304 ms**（最宽 **6.11%**），与 Qwen3-1.7B 那两个进程的 **6.13%** 一致。

### 三次审查修复（第 3 轮审查后，2026-10-29）

本轮只动**验证工具与报告一致性**：不新增推理优化，不放宽任何数值容差。
四个问题各有一条先失败、后通过的回归测试；下面每条都写清「修的是什么行为」。

#### 问题 1：状态文件与步骤日志的保护不完整，失败不传播

- **空目录不再被当成可用的输出目标。** `out_guard.py` 原来只有一条规则
  （「路径已存在就是冲突」），但驱动里的状态／日志路径需要的是**文件**：
  预先在 `out/recollection-status.txt` 建一个空目录，通用检查放行，四个步骤照跑，
  状态写入一路报 `Is a directory`，脚本仍然 `exit 0` 并打印完成。
  现在分出 `file_conflict()` / `preflight()` / `reserve(files=…)` 与 CLI 的
  `check-file` 子命令，**空目录也算冲突**；`out_guard.sh` 加了对应的
  `out_guard_check_file()`。
- **写入时防覆盖，不只启动时检查。** 预检通过之后、某一步启动之前出现同名文件，
  `run_recollection.sh` 的 `>` 重定向会截断它——JSON 那条排他写入保护盖不到日志。
  三个驱动（`run_recollection.sh`、`run_integration.sh`、`run_reference.sh`）改成
  `set -uC`：**noclobber 让 `>` 在已存在的路径上直接被拒绝**，命令不执行，
  非零状态和其它失败一样落进累计变量。`>>` 不受影响，追加语义保留。
- **证据写失败必须反映到退出码。** 驱动原来只累计测量子步骤的 rc，所以
  「八项全测成功、但摘要没写成」会报告完整成功。现在 `OVERALL` 之外另有
  `EVIDENCE`，`FINAL=$OVERALL; [ $EVIDENCE -ne 0 ] && FINAL=1`，并且两种失败
  印不同的话：一种是「某一步失败了」，一种是「每一步都成功了，但这一次没能把自己记录下来」。
- 独立子步骤**仍然按设计跑完**（一趟就知道八个里哪几个失败），变的只是最终状态必须真实。
- macOS 自带 bash 3.2 兼容：空数组在 `set -u` 下算未绑定，一律用 `${arr[@]+"${arr[@]}"}`；
  没有关联数组、没有 `mapfile`。

#### 问题 2：显式指定的配对实验目录缺失时静默退回旧实验

- `paired_directory()` 原来是**一串候选按顺序试**：先 `PAIRED_ROOT/noise-floor`，
  再 `PERF/noise-floor`，返回第一个存在的。拼错 `--paired-data` 或新实验还没采集，
  于是照常生成**上一轮**的结论，而报告抬头写的是操作员指定的那个目录。
- 现在**只有一个候选**：`PAIRED_ROOT` 是运行根，`paired_directory()` 只负责补
  `NOISE_DIR` 并做存在性检查，不搜索、不回退。参数解析由新的
  `resolve_paired_root()` 负责，三种结果：未传且数据根下有 → 用数据根自己的；
  未传且没有 → `None`（报告里显示「未运行」）；传了 → **只允许读那一个来源**，
  不可用时在**渲染任何一节之前**打印具体原因并 `SystemExit(1)`。
- 五种「不可用」分别点名：目录不存在 / 那个位置是个文件 / 缺 `noise-floor/` /
  JSON 读不出来（损坏）/ 读不出任何配对 JSON。**先失败、后写文件**，
  所以参数写错不会先覆盖正式文档再报错。
- 报告表头新增两行来源注释：`配对实验（§7.3）根目录：…` 与
  `配对实验来源：--paired-data` 或 `--data 自己的 noise-floor/`。
  光看路径分不出来——`<root>/noise-floor/` 无论是被点名的还是继承的，长得一样。

#### 问题 3：普通测试会改写真实历史报告

- `make_report.py` 新增 `--out` 与 `--doc`：**读哪个数据根**和**写到哪**是两种权限，
  原来是同一个变量。默认 `--out` 跟着 `--data`，而它落在
  `artifacts/perf-round1*/` 里时会被拒绝——那两个 `report-tables.md` 是已发布报告的
  派生证据，和它们旁边的原始 JSON 一起原样保留；要给本轮生成表格必须显式 `--out`。
- 测试侧：新增 autouse fixture `official_products_are_not_test_outputs`，
  它对正式产物取指纹（内容 + mtime），**任何测试改写它们就直接失败**；
  所有 `report.main()` 调用点（9 处，全部改完，不是只改示例那一处）统一走
  `render_report()`，正文与表格都落进 `tmp_path`。
- 于是审查里那个「历史 `report-tables.md` 列 65 项、正文列 64 项」不再是两处数字打架，
  而是**两个来源**：历史表读 `perf-round1-rerun/noise-floor/`（40 份 JSON），
  正文读 `perf-round2-paired/noise-floor/`（60 份 JSON）。本轮实测复现了这一点
  （同一份正文生成器、只换 `--paired-data`）：新根 **64 收益 / 5 尚不能确认 / 4 顺序不均衡**，
  旧根 **65 / 2 / 6**。**不手改数字**——§1.3 现在把本次调用的
  `--data` / `--paired-data` / `--out` / `--doc` 四个路径逐行列出来，
  正文与表格由同一次调用写出，可以互相核对。
- **发布必须排在完整套件之后，而且套件一变就得重发**，这条依赖本轮踩到两次：
  正文 §10 的期望值由 `quick_suite_expectation()` 从盘上最新那份 `quick*.log` 读出来。
  第一遍发布（19:42）执行时本轮日志还没落地，引用了上一轮那份（`1042 passed`）；
  第二遍（20:33）读到 `quick-fixed.log`（`1095 passed`），随后又补了三项守卫测试，
  计数变成 1098，于是第三遍（23:02）才是正式产物。
  那个函数按**名字**排序取第一个能读出摘要的 `quick*.log`，既不按 mtime、
  也不校验日志出自哪一棵树，所以两次都得靠命名驾驭它：
  发布后补跑的那份**不能**叫 `quick-fixed-*.log`（`-` 0x2D 排在 `.` 0x2E 前面，
  会抢在 `quick-fixed.log` 前被读走），叫 `suite-after-release.log`；
  而要让 §10 跟上 1098，新的那份**必须**排在前面，叫 `quick-final.log`
  （`final` < `fixed`，`n` < `x`）。函数本身没有改，两条都只写在文档里
  （交付报告 §9.3／§9.4／§9.8）。
- 发布之后又跑了一整个快速套件（第二遍发布后 `1095 passed, 62 deselected in 32.48s`，
  第三遍发布后 `1098 passed, 62 deselected in 30.99s`），
  两份正式产物的 mtime 一动没动，两个已发布根下 330 份文件的 sha256+mtime
  在十个时点上逐项相同——上面那个 autouse fixture 拦的是测试，这一趟量的是事实。
- `PROTECTED_PRODUCTS` 补上了 `docs/推理加速-第1轮-测量表.md`。清单原先只有三项
  （两个根的 `report-tables.md` 加交付正文），而这张表格是问题 3 要求
  「正文与当前表格同源」之后才出现的正式产物——漏掉它意味着测试改写交付表格
  不会有任何人报警。同时加了三项测试证明这个守卫自己有效：字节相同的重写
  仍然要被发现（`st_mtime_ns` 在指纹里就是为这个）、清单里的文件消失了要报警
  （`product_fingerprint` 对缺失路径返回 `None`，而 `None == None`）、
  `docs/` 下每一份 `推理加速-第1轮*.md` 都必须在清单里（期望值从文件系统取，
  不是把清单再抄一遍）。三项都做了反证，红法各不相同。
  快速套件因此从 1095 变成 **1098 passed, 62 deselected**。

#### 问题 4：直接跑测量程序时，输出冲突要到测完才被拒绝

- 六个接受 `--out` 的测量入口（`core_ab.py`、`vision_cache_ab.py`、
  `document_index_ab.py`、`identity_cost.py`、`projection_probe.py`、`sigmoid_check.py`）
  都把 `out_guard.preflight(ARGS.out)` 放在 `main()` 的**第一条语句**，
  在模型加载与测量之前。原来末尾那次 `write_json` 本来就是排他的，
  所以被占用的 `--out` 从来没有毁掉谁的上一次结果——它毁掉的是**这一次**：
  四套测量跑完，然后拒绝写入。
- **预检不代替最终写入**：末尾的排他写入保留，预检之后才出现的冲突仍然被它拦下。
- 六个入口共用 `out_guard` 一套规则，不是各写各的。已有文件、目录、符号链接、
  已发布证据目录、父路径不可创建，五种都在 `preflight()` 里判掉；**没有 `--force`**。
- 实测（`sigmoid_check.py`）：新路径 7.85 s 跑完并写出 JSON；同一路径第二次
  **0.095 s** 退出、rc=1、原文件 sha256 不变；指向已发布根、指向一个目录，
  都是 rc=1 且原文件不动。

#### 两个自指探针被门禁要求重测

问题 4 改了 `identity_cost.py` 与 `sigmoid_check.py` 自己，探针的 AST 哈希因此变了，
而门禁要求**最新的 artifact 必须出自树里这一份探针**。三条出路里只有重测是合法的：
改写已记录的哈希等于伪造出处，删掉旧运行等于销毁证据，放宽容差被明确禁止。

- 新根 `artifacts/probe-rerun-20261029/`：`sigmoid/sigmoid-check.json` 一份、
  `identity/SmolVLM-256M-Instruct-slowq-run{1,2,3}.json` 三份、
  `identity/Qwen3-1.7B-slowq-run1.json` 一份。配置与上一批逐项相同
  （`calls=200 warmup=20 key_calls=5 path_calls=5 max_context=1024 prefill_chunk_size=128`，
  同样两张验证图）。旧根一份未动。
- 逐项对比见 §7.1；两批的 `vlm.py` / `weights.py` AST 哈希相同（`e4a22324…` / `b041e6f6…`），
  数值同量级，`bit_identical` 全 true、`max_abs_difference` 恒为 0.0。
- **被门禁撑宽的两个区间**（不是手改数字，是覆盖检查要求源码不得低估离散度）：
  `vlm.py` 的身份检查代价从六个进程改成**九个进程**（互差仍在 8% 内，
  5 帧像素哈希跨九个进程仍是 15.41–16.62 ms，即 7.85%，`ceil` 还是 8），
  全权重 sha256 占视觉编码的比例 **15.4%–39.7% → 15.4%–60.3%**（新根三次都没驻留页缓存，
  宿主机 swap 738 MiB 对上一批 674 MiB）；`ops.py` 的标准正态输入加速比
  **7.3–7.6× → 7.3–7.7×**。

#### 交付报告的容差说明按场景分开写

上一版把项目容差统一写成「仍是 2e-4」，不准确。`2e-4` 是
`tests/test_model.py` / `tests/test_session.py` 里 **tiny 合成模型**上
chunk 一致性单元测试的 `rtol`（`atol` 是 `2e-5`），不是跨实现比较的门限。
四套门限各自独立声明，见交付报告 §9.5。

### 尚未验证

每一条都写明「要补什么才能验证」。完整表格见
[推理加速第 1 轮实测](推理加速-第1轮-2026-10-23.md) §9.4。

- **§3 那 25 行端到端负载：第 1 轮那批全部无自检，重采那批已采到，25 行里 2 行仍判 `待重测`。**
  第 1 轮的 `run_round1_longctx.sh` 只跑了 `--suite workloads`，`artifacts/perf-round1/ab-longctx/`
  的 JSON 里 `micro` 为空，所以每行的 `decode 自检` 列印 `—`、25 行全部按 `待重测` 处理。
  毫秒数是真实测量，但**无法判断这些差值里有多少是改动、有多少是那几分钟的机器状态**。
  **自检必须在同一进程、同一时间窗内采集才有意义**：借用几小时前另一次运行的裸 decode
  速率，恰好把它本来要抓的漂移当成了基准。驱动已改成 `--suite all --no-chat`，
  旧的无自检 JSON 保留在原目录，不覆盖。
  **2026-10-26 17:34:16 的重采（`run_round1_longctx rc=0`）落到了这一条**：
  `artifacts/perf-round1-rerun/ab-longctx/` 的 10 份 JSON 每份都带全部 4 个 micro 探针
  （`raw_decode` / `raw_decode_long` / `raw_prefill_full` / `engine_prefill`），
  25 行因此都有了同窗口自检。按 `make_report.py` 自己的判据
  （`suspect_self_check`，两侧之比 `>1.15` 判 `待重测`，**与加速方向无关**）算：
  **23 行可信、2 行待重测**，两行都是 Qwen3-1.7B——`multi-turn` 两侧自检相差
  **1.306×**、`emoji-mixed-script` **1.164×**（同模型 `zh-short` 是 1.131×，
  离门限只差 1%，也就是说这个模型有 3 行贴着判据）。decode 层的
  长/短上下文增长自检 5 个模型全部通过，最差的还是 Qwen3-1.7B 的 **1.142×**。
  **`docs/推理加速-第1轮-2026-10-23.md` 已对重采目录重跑**
  （`make_report.py --data artifacts/perf-round1-rerun`，整篇由数据派生）：§3 现在按
  `suspect_self_check` 判据算出 **23 行可信、2 行待重测**，不再印"25 行全部待重测"；
  §7 汇总段里那两句原先**硬编码**的旧结论也改成从数据派生（见下面
  「make_report.py 派生化修正」一条）。
- **decode 的 10 行是`单窗口观察值`**（5 个模型 × 短/长上下文）：base／opt 两侧各只有
  1 个窗口。**重复单位是整个窗口，同一窗口内的连续 token 不算独立重复实验**——
  一个窗口里 128 个 token 的每 token 耗时高度相关，把它们当 128 次重复会把离散度
  算小一个量级。`core_ab.py` 现支持 `--repeat N` 跑多窗口，第 1 轮那一轮没有用它重跑。
  **2026-10-26 的重采已经补上了这一条**：`--repeat` 的默认值就是 7，
  驱动没有覆盖它，所以 `artifacts/perf-round1-rerun/ab-longctx/*.json` 的
  `raw_decode` / `raw_decode_long` 各带 **7 个独立窗口**（`windows.samples` 7 个、
  `repeat: 7`，另有 `ms_per_token_windows` 的 min/max/mean/stdev 与
  `cache_lengths_observed` 断言每个窗口重建的是同一个状态）。
  旧数据里这两个节点**只有单个 `decode_seconds`、没有 `windows` 键**，
  所以「单窗口观察值」这个标签对旧数据成立、对重采数据不成立。
  多窗口立刻显出了它要显的东西：重采中 Qwen3-1.7B 短上下文 decode 的
  7 个窗口是 **1807.87–2460.70 ms**（极差 36%，取自 `run_round1_longctx.log`
  里 opt 侧那一行，JSON 落地后按同一节点复核），而 Qwen3-0.6B 同探针同侧是
  **685.23–705.79 ms**（极差 3.0%，取自 `ab-longctx/Qwen3-0.6B-slowq-opt.json`
  的 `micro.raw_decode.windows.samples`）——同一个进程、相邻几秒内的 7 个窗口，
  两个模型的极差差了一个量级。**任何小于这个量级的 decode 差值都不能读成收益或回退。**
  这个 36% 不能直接当成 Qwen3-1.7B 的固有离散度：重采窗口内机器**不是空闲的**
  （`recollection-environment.txt` 记 load `2.05 2.16 2.35`、CloudShell 101.8% CPU），
  而它的权重是 6.4 GiB、页缓存装不下几份。**要分清这两者需要在空闲窗口重跑该探针**，
  那正是 `run_noise_floor.sh` 的配对实验要做的事。
- **重采的两批 decode 数据互相矛盾，Qwen3-1.7B 连方向都相反。** 同一棵最终源码，
  `ab/`（step 1）与 `ab-longctx/`（step 2，`--max-context 2048`）相隔约 20 分钟、
  各自 7 个窗口，`raw_decode` 的中位数之比是 Qwen3-1.7B **1.064×**（`ab/`）对
  **0.928×**（`ab-longctx/`）——**符号相反**；`raw_decode_long` 是 **1.501×** 对
  **1.060×**，差 42%。最干净的证据在 base 侧自己身上：同一个探针、同一棵基线树、
  同一天，`ab/` 的中位数 **92.39 ms/token**、`ab-longctx/` **67.74 ms/token**，差 36%。
  其余 4 个模型两批的**方向一致**（都偏向 opt），幅度相差最多 0.07×
  （Qwen2.5-0.5B 的 `raw_decode` 1.147× 对 1.080×、SmolLM2-135M 1.121× 对 1.076×）。
  于是能说的只有：**4/5 个模型的方向在两批独立采集里一致，Qwen3-1.7B 不一致**；
  幅度一律 `尚不能确认收益或回退`。10 行里两批各有 4 行的窗口区间**重叠**，
  重叠的是哪 4 行两批也不同。`make_report.py` 现有的"两次运行不一致"段只比
  workload 的 `total_seconds`，而且只在 `primary != "ab-longctx"` 时触发，
  **读不到 micro decode 的这个矛盾**——要补一条比 decode 中位数的检查。
  **step 3 的配对实验把这个矛盾量出来了，结论是「矛盾本身就是预期结果」。**
  Qwen3-1.7B 的 `raw_decode` 现在有 **6 个独立进程**测同一份代码
  （`ab/`、`ab-longctx/`，加 `noise-floor/` 的 r1–r4）。base 侧中位数是
  **61.33 / 60.33 / 74.69 / 60.74 / 58.24 / 63.64 ms/token**，opt 侧是
  **57.65 / 64.98 / 63.51 / 62.26 / 57.92 / 56.41**；
  也就是**同一棵树的同一个探针，跨进程能差 base 26.9%、opt 15.2%**，
  而那两批报出的倍率之差（1.064× 对 0.928×，13.6%）**小于 opt 侧自己的跨进程极差**。
  所以「符号相反」不是两批里有一批错了，是**这个模型这个探针的被测量小于测量分辨率**。
  **而且这个结论不能靠剔除异常值挽救**——4 轮 × 2 侧共 8 个 `raw_decode` 节点
  **全部通过** spike／level 两条门限（`max/median` 最大只有 **1.79**，
  level 判据 base 最大 **1.20**、opt **1.06**，都远低于 2.0／1.5），
  可 base 四轮中位数的 `max/min` 仍是 **1.28**、opt 是 **1.13**。
  **它的抖动是弥散的（7 个样本一起偏高一点），不是尖峰型的**，
  所以任何离群值规则都清不掉；8 个节点全留下时配对倍率是
  **1.1761 / 0.9756 / 1.0054 / 1.1283**，中位 **1.0668**，区间 **[0.9756, 1.1761] 跨过 1.0**，
  而这个区间**整个落在该模型自己的 B 门限（中位 10.31%）之内**。
  **判定：`尚不能确认收益或回退`。**
  这里也修正上一条里的一处措辞：那 36% 的窗口极差**不是** Qwen3-1.7B 的固有离散度，
  而是**进程内**离散度（`ab-longctx/` 那一个进程的 7 个窗口）；
  配对实验测的是**进程间**离散度，两者不是一回事，都很大，但不能互相替代。
- **窗口间离散度在 base／opt 两侧不对称，但这个不对称分探针、且与运行顺序混淆。**
  重采 `ab-longctx/` 里 10 个节点的 stdev/median：`raw_decode` 上 **base 侧 5/5 更离散**
  （13.2% 对 11.6%、5.8% 对 1.9%、1.6% 对 0.2%、1.4% 对 0.6%、1.1% 对 0.9%，
  依次是 Qwen3-1.7B、SmolLM2-135M、Qwen2.5-0.5B、SmolLM2-360M、Qwen3-0.6B）；
  `raw_decode_long` 上只有 **2/5**（Qwen3-1.7B 60.4% 对 9.3%、SmolLM2-135M 3.5% 对 1.9%），
  另外 3 个反过来是 opt 更离散（Qwen3-0.6B 2.5% 对 1.5%、Qwen2.5-0.5B 2.6% 对 2.2%、
  SmolLM2-360M 1.4% 对 0.3%）。**所以"基线树更抖"不成立**；成立的只有
  `raw_decode` 上那个 5/5，而 `run_round1_longctx.sh` 是 `for side in base opt`——
  **base 一律先跑**，"先跑的一侧更抖"（冷页缓存、mmap 首次触页）与"基线树更抖"
  在这批数据里分不开。分开它要靠 `run_noise_floor.sh` 的 AB/BA 顺序均衡。
  另一个后果是**中位数之比这个统计量对高离散行偏弱**：Qwen3-1.7B base 的
  `raw_decode_long` 7 个窗口是 175.54 / 90.13 / 62.41 / 67.74 / 79.52 / 63.54 / 61.99
  ms/token，最慢那个是它自己中位数的 **2.6 倍**，n=7 的中位数标准误在这种分布下
  不比被测的差值小。
  **AB/BA 均衡把「先跑的一侧更抖」和「基线树更抖」分开了，答案是两个都不是——
  是「Qwen3-1.7B 这一个模型在两侧都会偶发停几十秒」。** Qwen3-1.7B 的 r1
  （base 在 pos1）里，base 那个进程 `wall_seconds = **1225.2 s**`，
  而同参数同树的 r2 base 只要 **261.5 s**（**4.68×**）；r1 base 的 `multi-turn`
  7 个样本是 1.38 / 1.33 / 1.32 / 0.94 / 1.58 / **80.09** / **102.83** s
  （max/median **74.8×**），`raw_decode_long` 窗口是
  59.0 / 67.0 / 177.9 / **469.1** / 150.1 / 85.2 / 176.5 ms/token。
  **但 r1 的 opt 侧同样被污染，只是污染在别的探针上**：
  opt 的 `engine_prefill[32]` 累计 **68.7 s**、`engine_prefill[8]` 累计 **136.3 s**，
  而 base 同两个探针各只要 1.5 s 与 1.1 s（**46× 与 126×**）。
  r2 的 pos1 换成 opt 之后 base 不再爆——可是 **r2 的 opt 侧自己爆了**：
  `raw_prefill_full[512]` 累计 81.6 s 对 base 的 16.6 s（**4.93×**），
  而且**中位数**是 12.51 s 对 r1 opt 的 1.616 s。
  **r4 又轮到 base 爆**（`engine_prefill[32]` 的 `max/median = 2.83`），
  而 r4 是偶数轮、pos1 是 opt——**爆点在 4 轮里出现在 pos1 也出现在 pos2、
  出现在 base 也出现在 opt**，所以「先跑的一侧更抖」和「基线树更抖」**都被排除**，
  剩下的解释是**这个模型在这台机器上会随机停几十秒，与它跑在哪棵树、第几个位置无关**。
  对照：其余 4 个模型 32 个文件里只有 SmolLM2-360M 的 r1 opt 被标记一次（2.13），
  而且差值来自真实的快慢（base ≈100 s、opt ≈80 s 这种系统性差别），不是抖动。
- **污染探测要用「逐节点 max/median」＋「跨轮中位数水平位移」两条，`wall_seconds` 不够；
  而且这两条对两侧同样严格。** 在 `noise-floor/` 40 份 JSON（4 轮 × 5 模型 × 两侧）上：
  - **spike**：同一节点内 `max/median > 2.0`（抓单个爆点）；
  - **level**：该轮中位数 ÷ 同组各轮中位数的中位 `> 1.5`（抓整轮平移——spike 抓不到它，
    因为一轮 7 个样本**一起**变慢时 `max/median` 仍然很小）。
  命中的 13 个节点级标记里 **12 个属于 Qwen3-1.7B**（另 1 个是 SmolLM2-360M 一个中位数
  只有 39 ms 的探针，2.13 相当于 43 ms 的绝对抖动），其中 spike 8 个、level 4 个
  （`engine_prefill[8]` 达 **149.4×**、`raw_prefill_full[512]` **7.96×**、
  `raw_decode_long` **2.28×**、`engine_prefill[32]` **6.26×**）。
  **按文件级**（一个节点坏就整份弃用）被标记的是 Qwen3-1.7B 的 r1 base、r1 opt、
  r2 opt、**r4 base** 与 SmolLM2-360M 的 r1 opt，其余 3 个模型 **0 标记**。
  `wall_seconds` 抓不到这个：r1 opt 的 wall 是 539.1 s 对 r2 opt 的 327.2 s（1.65×），
  r2 opt 对 r2 base 只有 1.25×——**都低于任何合理的整进程门限，可那两个进程里各有一个爆点**。
  **一个自己踩过的坑，值得写下来**：这套探测的第一遍跑出来「只有 1 个节点被标记」，
  看着像数据很干净，其实是**取值函数悄悄漏了两类节点**——`engine_prefill` /
  `raw_prefill_full` 的长度键在 JSON 里是**字符串** `"512"` 而我按整数 `512` 取，
  workload 节点的逐次样本在 `runs[]` 里而**没有** `samples` 键。
  修好后覆盖率是 **584/600 个节点**（缺的 16 个是两个 SmolLM2 上被 harness 主动跳过的
  `long-prompt-short-answer`，见下文），标记数从 1 个变成 13 个。
  **所以这类筛查必须先打印「覆盖节点数 / 期望节点数」再读结论**，
  否则「0 标记」与「根本没看」长得一模一样。
- **对 Qwen3-1.7B，7 个样本的中位数不稳健——这一条现在有了它自己的门限数字。**
  同树同模型同参数，r1 与 r2 的中位数之比：
  `multi-turn` base **1.375**、`engine_prefill[32]` opt **6.25**、`engine_prefill[512]` base **1.445**、
  `raw_prefill_full[512]` opt **0.129**（r2 反而慢 7.7 倍）、`raw_decode_long` base **2.443**。
  **中位数在 1.4×–7.7× 之间移动，而被测的优化效应只有 1.0×–2.1×**，
  所以「取中位数就抗污染」这句话**对这个模型不成立**。
  逐节点过滤后按模型算三个离散度（见下一条定义），**Qwen3-1.7B 的进程内极差 A 中位
  17.28%，是其余 4 个模型（2.52%–4.17%）的 4–7 倍**；它自己的判读门限
  **B 中位 10.31%、p90 33.25%、max 45.32%**，而其余 4 个模型的 B 中位只有 1.32%–3.74%。
  后果很具体：它 `raw_decode` 的配对倍率中位 **1.0668 [0.9756, 1.1761]**，
  **整个区间都落在自己 10.31% 的门限之内**——不是「偏向 opt 一点」，是**测不出来**。
  它的 prefill 行（1.25–1.46）超过 B 中位但没超过 B p90，比其余模型弱一档。
  要补：Qwen3-1.7B 的任何倍率都必须**逐节点先过 spike／level 门限**，
  不合格的轮次整行标 `待重测`，而不是取中位数了事；
  decode 那两行按现有数据只能写 `尚不能确认收益或回退`。
  **为什么只有这个模型会停，仍然是假设不是结论**：它的权重 6.4 GiB，
  是其余模型的 3–11 倍，页缓存装不下几份——但本轮没有采集内存时间序列，
  也没有操纵内存压力做对照，所以不能这么归因。
- **配对实验的判读结果（step 3，40 份 JSON，逐节点过滤）。** 判据只用**符号**：
  每一轮独立进程算一个 `base 中位 / opt 中位`，看它是否 `> 1`。
  **282 个配对样本里 279 个 `> 1`，双侧符号检验 p = 9.6e-79。**
  按模型／行看：**Qwen2.5-0.5B 15/15 行、Qwen3-0.6B 15/15 行、SmolLM2-135M 14/14 行、
  SmolLM2-360M 14/14 行——每一行的每一个可用轮次都是 `base > opt`**（n=3–4 轮）；
  Qwen3-1.7B 15 行里 **12 行全正、3 行混合、0 行全负**（n=2–4 轮）。
  那 3 个负样本是：`raw_prefill_full[8]` r3 比值 **0.8385**（绝对差只有 **29.4 ms**：
  base 152.4 ms 对 opt 181.7 ms，而同组另 3 轮是 1.0115 / 1.0669 / 1.0644）、
  `raw_decode` r2 **0.9756**、`raw_decode_long` r3 **0.9641**——**三个都在该模型自己的
  B 门限之内**，所以读成「尚不能确认收益或回退」，不是读成回退。
  **这两条门限也不是万能的，得写清它漏什么**：上面那个 `raw_prefill_full[8]` r3 base
  节点的**进程内**样本极差是 **78.4%**（opt 侧 52.2%），可它的 `max/median` 没超过 2.0，
  于是通过了 spike 判据。也就是说 spike 抓的是「7 个样本里有 1 个爆」，
  抓不到「7 个样本整体都很散」；level 抓的是「整轮平移」，也抓不到轮内弥散。
  **要补第三条：进程内极差 `(max−min)/median` 超过某个门限（比如 25%）的节点也应弃用。**
  本轮没有加它，因为加了会把 Qwen3-1.7B 剩下的轮次也大量弃掉，
  而该模型的结论已经是「测不出来」，多弃不影响判定；
  但**其余 4 个模型若将来出现同类弥散，这条缺失会让坏数据混进来**。
  **这正是配对设计买到的东西**：若按「文件级弃用整份」（一个节点坏就丢一整轮），
  Qwen3-1.7B 只剩 **1 个干净轮次**，什么都判不了；逐节点弃用把可用轮次恢复到 2–4。
  两种口径都算过，报告用**逐节点**那一个，因为它丢的是被污染的那个测量，
  而不是同一进程里那些干净的测量。
  倍率的**幅度**（各轮比值的中位）：prefill 随长度上升后在 512 回落，
  `engine_prefill` 是 len8 **1.08–1.30×** / len32 **1.27–1.88×** / len128 **1.44–2.18×** /
  len512 **1.29–1.95×**；decode 两行只有 **1.02–1.12×**（Qwen3-1.7B 除外，见上）。
- **三个离散度必须分开报，而且**同尺度才能比**。** 对每个 `(模型, 行, 侧)`
  在过滤后的轮次上定义：**A** = 进程内 7 个样本的 `(max−min)/median`（取各轮中位）；
  **B** = 跨进程**中位数本身**的移动 `(max−min)/median`；
  **C** = 把各轮样本合并后的 `(max−min)/median`。对齐后共 **146 个三元组**：
  **A 中位 3.84%、B 中位 2.77%、C 中位 11.25%**。
  - **A 与 C 同尺度（都是单样本极差），可以直接比：`C > A` 在 146/146 = 100% 成立，
    `C/A` 中位 2.48。** 所以旧文档那句「进程内的 min–max 严重低估跨进程离散度」
    **成立**，而且现在是在同尺度上比出来的。
  - **B 与 A 不同尺度**（一个是中位数的移动、一个是单样本的极差），
    `B < A` 在 79/116（文件级口径）成立——**旧文档把这两个数放在一起比，是错的**。
  - **判读一个「中位数之比」时该用的门限是 B，不是 C。** 因为报告里的倍率就是两个中位数相除，
    它继承的是**中位数**的跨进程移动，不是单样本的极差。
    各模型的 B（中位／p90／max）：Qwen3-0.6B **1.32% / 4.26% / 5.75%**、
    SmolLM2-135M **2.75% / 5.45% / 8.03%**、SmolLM2-360M **2.79% / 5.90% / 12.07%**、
    Qwen2.5-0.5B **3.74% / 11.25% / 13.14%**、Qwen3-1.7B **10.31% / 33.25% / 45.32%**。
    **decode 那两行 1.02–1.12 的倍率，在 Qwen2.5-0.5B 与 SmolLM2-360M 上是落在
    各自 B 的 p90 附近的**，所以「符号全正」与「幅度可判」是两件事：
    符号检验的 p 值极小（279/282），但**单行的幅度只有超过该模型 B 才算测得稳定收益**。
    本报告因此把 decode 行写成「观察到加速」而不是「测得稳定收益」。
- **Qwen3-1.7B 的 `chat` cold-vs-cold 计时**：四行读起来像回退
  （0.85× / 1.04× / 0.78× / 0.63×），但与那批被标记待重测的 workloads 数字来自
  **同一个窗口**（`run_round1_longctx.sh` 只补跑了 workloads，没有补跑 `chat`），
  故**尚不能确认收益或回退**：既没被推翻也没被证实，原始记录保留。
  `输出一致=是` 那一列不受影响、仍然成立。
  要补：在同一窗口内相邻交替重跑 `chat` 套件。
- **Qwen3-1.7B 的 `raw_decode_long`**：报出 0.52×（即变慢），但同进程内长/短上下文之比
  是 1.98 而 base 侧只有 1.00，cache 从约 96 条涨到约 160 条不可能让每 token 成本翻倍，
  故**尚不能确认收益或回退、标记待重测**（本条早先写作「判为不可信」，
  那是拿一个未经证实的物理直觉去否掉一次实测，已改）。
  要补：在空闲窗口用 `--repeat N` 单独重测该探针，得到它自己的离散度。
- **优化 1 与优化 2 现在能分开了，靠的是 harness 记下的 `entry_points`，不是推算。**
  `core_ab.py` 在 `engine_prefill` 每个节点上贴了 spy，把每次进入的入口名逐个记下
  （`tools/perf/core_ab.py:333`：`vocabulary_projections = spy["entry_points"].count("forward")`）。
  实测（Qwen3-0.6B r1，5 个模型全部同构）：

  | prompt 长度 | base `entry_points` | opt `entry_points` | 词表投影次数 |
  | --- | --- | --- | --- |
  | 8 / 32 / 128 | `['forward']` | `['forward']` | **1 对 1，相同** |
  | 512 | `['forward']×4` | `['encode','encode','encode','forward']` | **4 对 1** |

  `model.encode()`（`src/slowq/model.py:281`）是**只跑 decoder stack、不投影词表**，
  `forward()` = `encode()` + 词表投影。两侧都是 4 个 128-token chunk
  （`forward_token_counts` 均为 `[128,128,128,128]`，`prefill_chunk_size` 均为 128），
  所以 opt 恰好**省掉 3 次投影**，而 chunk 数、token 数完全一致。
  **推论：len ≤ 128 时投影次数相同，Δ8／Δ32／Δ128 里可证明不含优化 1。**
  这三档因此就是优化 2（及其余非投影改动）的单独定价：
  `engine_prefill` 的 base/opt 中位数之比是 len8 **1.078–1.296×**、len32 **1.282–1.886×**、
  len128 **1.419–2.184×**；len512（含优化 1）反而回落到 **1.353–1.961×**，
  **512/128 = 0.898–0.969，5 个模型全部小于 1**。
  （口径：逐节点过滤，见上一条；Qwen3-1.7B 每档可用轮次 2–4，其余模型都是 4。）
- **优化 1 单独定价失败，这个「测不出来」本身是结论。**
  用 `Δ512 − 4·Δ128` 消掉优化 2（Δ = base 中位 − opt 中位）得到的残差，
  10 个（5 模型 × 2 探针）是
  **−3.61 / +8.02 / −55.84 / +0.82 / +8.94 / −22.77 / −9.72 / +29.31 / −13.81 / −17.47 ms**，
  即 **−2.75% ~ +1.38% of base(512)，6 负 4 正，符号不稳定**。
  **最有力的反证来自同一个模型的两个探针**：Qwen3-1.7B 的 `engine_prefill` 残差是
  **−55.84 ms**、`raw_prefill_full` 是 **+29.31 ms**——**同一批轮次、同一棵树、
  名义上测同一件事，符号相反、相差 85 ms**，而优化 1 真正省下的 3 次投影
  不可能是负的。所以这个方法给出的不是「优化 1 很小」，是「这个方法没有分辨率」。
  **而且消元的前提本身就不成立**：它假设优化 2 的节省随 token 数线性增长，
  但 attention 是超线性的。512 token 分 4 个 128-token chunk 时，
  第 i 个 chunk 的 query 要对 (i+1)·128 个 key 打分，总量是
  `128×(128+256+384+512) = 128×1280`；而「4 × len128 单 chunk」是 `4×128×128 = 128×512`。
  **两者之比是 1280/512 = 2.5，不是 1**，所以 `4·Δ128` 这个减数从一开始就偏小，
  残差里混着「优化 1 的节省」与「attention 的超线性」，两者分不开。
  **判定：本轮无法用相减法给优化 1 单独定价**，只能给上界
  `|残差| ≤ 56 ms ≤ 2.8% of base(512)`。
- **「要补一个直接开关做配对」这句话是错的——那个探针本来就有，是重采时漏跑了。**
  本条早先写「要补：直接在 opt 树上开／关『跳过非末 chunk 投影』这一个开关做配对」。
  核实后：`tools/perf/projection_probe.py` 做的就是这件事，而且比开关法更干净——
  它不碰任何开关，直接对 `forward` 与 `encode` **成对交替计时**（`forward` 先、
  `encode` 后，差值在**对内**取，7 对取中位），因为 `forward` 在源码里就是
  `encode` + 一次 `output_projection(hidden[-1])`（`src/slowq/model.py:390-398`），
  所以差值就是一次投影的净代价，权重、内存状态、进程全部固定。
  **真正的缺陷是 `run_recollection.sh` 只串了三个 driver，没有这一步**，
  于是 `artifacts/perf-round1-rerun/projection/` 根本不存在，
  而 `make_report.py` 的 `opt1_share_percent` 会在那里静默取到空列表。
  已补：新增 `tools/perf/run_projection.sh`（一模型一进程、严格串行、
  拒绝写入 `artifacts/perf-round1`、要求 `<out-root>/ab` 已存在），
  并作为 step 4 挂进 `run_recollection.sh`。
  同时给 `projection_probe.py` 补了三件它原先没有的东西：
  ① `source_identity.record()`（原先它的 JSON **没有任何源码身份**，
  无法与某一棵树关联，违反「新的结论必须关联新的执行记录」）；
  ② `--ab-dir` 默认从 `--out` 反推 `<out-root>/ab`，
  原先硬编码 `artifacts/perf-round1/ab`——**重采时会拿新一轮的投影代价除以旧一轮的
  prefill，得到一个跨轮混算的「占比」**；③ `validity` 字段与逐行 `invalid` 标记。
- **② 那条自检不是空想，重采两次都撞上了，而且撞得很响。**
  第一次重采 `projection/`（2026-10-26 20:09:04–20:20:03，5 模型串行）：
  四个小模型干净——SmolLM2-135M **0.78 / 0.78 / 0.92 ms**、SmolLM2-360M
  **1.93 / 1.48 / 3.07**、Qwen2.5-0.5B **4.00 / 3.80 / 5.49**、Qwen3-0.6B
  **4.52 / 4.67 / 3.37**（依次是 128／256／512 token），
  跨长度极差只有 **17.9% / 107.4% / 44.5% / 38.6%**（按幸存日志里的中位数重算），
  比旧一轮（**127.4% / 20.8% / 148.8% / 86.6% / 66.5%**，直接读
  `artifacts/perf-round1/projection/*.json`）并不整齐地更好——
  **本条早先写的是「18% / 101% / 74% / 39% 比旧一轮 46.9% / 117.6% / 174.0% /
  197.5% / 1036.5% 干净得多」，那两组数都核不出来，已按盘上数据更正**：
  旧一轮那五个数里最大的 1036.5% 在任何一份 JSON 里都找不到对应，
  而"重采比旧一轮干净"这个方向性说法本身也不再成立——
  两次各有各的脏模型，脏的不是同一批。
  **而 Qwen3-1.7B 三个长度的成对差值中位数全是负的：−8644 / −6250 / −7012 ms。**
  `forward` = `encode` + 一次真实存在的矩阵乘，**负值在算术上不可能**，
  所以这不是「测到一个很小的负数」，是**这次测量无效**。
  **这一段里凡是需要逐样本或 RSS 的细节，现在只有第二次重采的能核**——
  第一次的 JSON 被第二次覆盖了（见下一条），第一次能核的只剩
  `run_projection.log` 里的中位数、占比与 Qwen3-1.7B 那行 `−945.10% of base engine prefill`。
  第二次（20:26:32–20:33:35）在盘上的对应证据是：
  `wall_seconds = 341.9 s`（**已发布那轮同一个探针 66.96 s，慢 5.1 倍**）、
  `rss_after_load_bytes = 6.69 GiB`、
  `encode` 的 7 个样本在 **0.371 s ↔ 21.705 s** 之间跳（**58 倍**）、
  128／256／512 tok 分别有 **4/7、6/7、7/7** 个成对差值为负，
  `validity.verdict` 自己写着 "unusable: … the model was paged out between the two
  halves of the pair"。
  这正是探针自己 docstring 里写着的那条警告
  （"The 6.4 GiB Qwen3-1.7B gets compressed out of memory if anything else is resident"）
  成真了——两次机器上都有一个虚拟机常驻、`load average 2.2–2.5`。
  **两次独立尝试同样失败，所以这是环境限制，不是一次抖动。**
  处置：**不删**，JSON 原样留着，但整行标 `作废`，
  不进中位、不进占比；报告里明确写「下面的占比区间只覆盖采到有效数据的模型」。
  **教训是：一个探针若存在「结果不可能为负」的算术恒等式，就必须把这条恒等式写成自检。**
  原先这个探针没有，所以它把 −945.10% of base prefill 这个数
  当成正常结果打到了 stderr 上，只是碰巧没人照抄。
- **污染不止 Qwen3-1.7B，两批数据里每个模型都有；已发布那轮甚至把负中位数印进了表，
  然后给它编了个理由。** 逐样本数一遍（5 模型 × 3 长度 × 7 对 = 105 对）：

  | 批次 | 负值成对差 | 最差单对 | 有负值的模型 |
  | --- | --- | --- | --- |
  | 已发布 `artifacts/perf-round1/projection/` | **15/105 = 14.3%** | −168.4 ms（Qwen3-1.7B 256 tok） | **5/5**（4、5、1、2、3） |
  | 重采第 2 次 `artifacts/perf-round1-rerun/projection/` | **25/105 = 23.8%** | −13419.4 ms（Qwen3-1.7B 128 tok） | **5/5**（1、4、1、2、17） |

  两处后果，一处比一处严重：
  ① 已发布报告 §2.4 那张「一次词表投影的净代价」表里，SmolLM2-360M 的 512 tok 单元格
  **印的就是 `-0.98 ms`**，紧接着 §2.4 的归因结论第一条还专门解释它：
  「那不是说跳过投影会变快……这正是探针在说『这个量低于它自己的分辨率』」。
  **这个解释是错的，而且错法很典型**：`forward` = `encode` + 一次真实存在的矩阵乘，
  所以成对差为负**不是"小到分辨不出来"，是两半没有在同一个机器状态下测**——
  前者劝读者继续用同一列的其他格子，后者要求这一格退出一切统计。
  报告把一个**测量无效**的信号读成了**测量精度**的信号。
  ② 那个 −0.98 ms 还往下游走了：`用 512 tok 代价` 列算出 **−2.9 ms**、
  占比 **−0.32%**、`优化 1 占省下的` **−0.7%**，于是「占比区间 −0.32% ~ 3.02%」
  的下界是一个负代价**从实测省下里减掉**得来的——负代价乘跳过次数会**抬高**优化 1
  的表面占比。已在 `make_report.py` 里改成：中位数 ≤ 0 的格子标 `作废`、
  不进「三者中位」、不参与任何占比，并且**第二张表在 `direct ≤ 0` 时整行跳过**。
  ③ 顺带核实：已发布报告说的「最差是 Qwen3-1.7B 256 token 的 −168 ms」**是对的**
  （盘上就是 −168.4 ms）。`make_report.py` 现在把这个数改成从 JSON 里取最小的那一对，
  对旧批次会算出同一个 −168.4 ms，对重采批次会算出 −13419.4 ms——
  **派生出来的值两批都对，硬编码的那句只对一批**。
- **探针的 `validity.verdict = "usable"` 只检查中位数，不等于每对都干净。**
  重采第 2 次那四个"可用"模型，各自仍有 1、4、1、2 对负值（合计 8/84 = 9.5%）。
  所以 `usable` 的准确含义是**"三个长度的中位数都为正"**，
  不是"这一格测量可信"。这也是为什么占比只能给区间、而且区间必须连着
  上一条那个跨批次跨度一起读，不能摘出来当一个数用。
- **重采的 `ab/` 里藏着一个 16.52× 的假加速，是这次新加的自检抓出来的，
  原先那张表没有任何检查。** Qwen3-1.7B 的「整块 prefill 512 tok」在重采批次里是
  base **23001.6 ms** 对 opt **1391.9 ms**。而**同一份 base JSON、同一个长度**的
  `engine_prefill` 只有 **2225.8 ms**（7 个样本 2155–2685 ms，很干净）。
  两个探针 prefill 同样多的 token、同样只投影一次词表，真正的差别只有
  「一次全长 attention」对「四次逐渐变长的」，所以在**同一棵树内**它们的比值必然≈1——
  这里是 **10.33**。也就是说 base 侧那 23 s 里绝大部分不是 prefill，
  是权重被换出去再换回来。
  于是加了一条**派生的、对称的**自检：每侧算 `raw_prefill_full ÷ engine_prefill`，
  再取两侧这个比值之间的相差，判据沿用报告里已有的 `SELF_CHECK_TOLERANCE`（1.15×），
  **不看加速方向**——这一行报出来的是个巨大的"加速"，只盯回退的规则会放它过去。
  **区分度是核过的，不是挑出来的**：两个批次 × 两个长度 = **20 行干净数据全部落在
  1.004–1.115**，被污染的那一行是 **11.613**；`noise-floor/` 那 20 个配对里
  19 个落在 1.001–1.115，第 20 个是 **5.746**。中间有一个数量级的空档。
- **这个 16.52× 的真值是多少，有三条独立的路都指向 ≈1.42×。**
  ① 同一份 `ab/` JSON 的 `engine_prefill` 512：2225.8 / 1564.3 = **1.42×**；
  ② `noise-floor/` 里 Qwen3-1.7B「整块 prefill 512 tok」在污染筛选后**只剩 r3、r4 两轮**，
  两轮的增益是 **1.396× 与 1.431×**；
  ③ 已发布那一轮同一格是 **1.43×**。
  四个数（1.42／1.396／1.431／1.43）互相咬合，16.52 是唯一的离群值。
  **这一条也正好演示了配对实验为什么必须存在**：同一个量、同一棵树，
  单进程一次测出 16.52×，四次独立进程测出 1.4×。
- **`noise-floor/` 那一格是 level 判据抓住的，spike 判据抓不住——这次有了具体例子。**
  r2 opt 的 7 个样本是 **1818 → 4096 → 6149 → 12507 → 16966 → 20284 → 19820 ms**，
  **单调爬升**：机器在那一个探针执行期间持续恶化。
  它的进程内 `max/median` 只有 **1.62**，**低于 spike 判据的 2.0**，
  所以 spike 抓不到；但它的轮中位数 12507 ms 对 opt 侧四轮中位数的中位（1570 ms）
  是 **7.97×**，**远超 level 判据的 1.5**，于是被剔掉。
  先前记的「两条判据都抓不住弥散型抖动」这句仍然成立，
  但要补一句：**渐进恶化型**（样本单调爬升、极差不大）恰好是 level 判据的射程之内，
  而 spike 判据对它完全瞎。r1 两侧则是被 spike 抓到的（3.46 与 3.51）。
- **三张表现在剔的是同一批行，且「路线 A/B 交叉核对」的结论改成派生的了。**
  「一次词表投影的净代价」标 `作废`、「整块 prefill」标 `待重测`、
  「两条独立路线」整行跳过，三处指向同一批模型，报告里写明这层对应关系。
  交叉核对的结论原先无条件印「这个一致性足以支撑结论」，
  并把最大相差硬编码成 10.0%；现在按**已有**的 `SELF_CHECK_TOLERANCE` 换算成
  **≤15%** 的判据来判：已发布批次最大相差 **10.0%，通过**；
  重采批次最大相差 **17.9%（Qwen2.5-0.5B），不通过**，
  报告于是明写「**交叉核对没有通过**……『省下的大部分来自优化 2』这一句
  在本批数据上不能由这张表支撑」，并说明它只能由「优化 1 占比」表支撑，
  而那张表给的占比本来就在个位数百分比以下。
  这一处是**结论被数据推翻、而不是数据被结论筛掉**的例子。
- **配对实验的表原先只印了 5 个模型里的 2 个，74 行里只印了 29 行。**
  `noise_floor()` 是 §7 那两张扫描表的噪声来源函数，它顺手把
  `paired_noise_section` 也调了，而传进去的是 `SWEEP_MODELS`
  （只有 `Qwen3-0.6B` 与 `SmolLM2-360M`）。**可 `run_noise_floor.sh` 跑的是全部 5 个模型**：
  `noise-floor/` 里 40 份 JSON = 5 模型 × 4 轮 × 2 侧，一份不少。
  于是那张「用来取代被撤回的 3% 噪声底线」的表，
  **对另外 3 个模型的 prefill／decode／端到端一行判读都没有**，
  而报告正文里 §2–§6 的行恰恰是那 3 个模型的。
  这不是取不到数，是取到了没印——JSON 全在，读者看到的表却像只做了 2 个模型。
  已改：`noise_floor()` 把 `MODELS` 全量传给配对节，
  **但返回值仍然只在 `SWEEP_MODELS` 里取中位**（那两张扫描表只覆盖这两个模型，
  把另外三个的离散度混进来等于又造了一个统一门限）。
  改完 73 行全出，`sweep_opt_between` 的中位仍是 **1.5%**，与改之前逐位相同，
  所以 §7 引用的那个数没有被这次修复动过。
- **改完之后，本轮第一次有了「测得稳定收益」这个档位的行。**

  > **⚠️ 2026-10-28 撤回：本条的「70/73」与「顺序均衡 5/5 全部为「是」」都不成立，
  > 见下面「二次审查修复」的 问题 3。**
  > 剔除异常轮次之后没有重算 AB/BA 均衡：有 **6 行**只保住 r2/r3/r4，
  > 即 **1 个 AB + 2 个 BA**，却被印成「顺序均衡：是」并计入了那个 70。
  > 按保留轮次重算后是 **65 个「有限轮次方向一致」/ 2 个「尚不能确认收益或回退」/
  > 6 个「顺序不均衡或未记录」**，合计仍是 73，但那 6 行不再判方向。
  > 而且「测得稳定收益」这个档位本身已被取消——`make_report.py` 现在只发
  > 「有限轮次方向一致」，因为 2–4 轮同向不足以支持「稳定」。
  > 原始 JSON（`artifacts/perf-round1-rerun/noise-floor/`）**一字未改**，
  > 它是这次撤回背后的证据；重测的配对实验单独放在
  > `artifacts/perf-round2-paired/`，加在它旁边，不替换它。
  > 下面这段保留原样，作为当时怎么得出那个数的记录。

  73 个（模型 × 量）里 **70 个的逐轮增益没有跨过 1.00**、**3 个 `尚不能确认收益或回退`**
  （Qwen3-1.7B 的 decode 短／decode 长／整块 prefill 8 tok）；
  按模型：SmolLM2-135M **14/14**、SmolLM2-360M **14/14**、Qwen2.5-0.5B **15/15**、
  Qwen3-0.6B **15/15**、Qwen3-1.7B **12 收益 / 3 未定 / 0 回退**。
  弃用 10 个（量 × 轮）配对、涉及 11 格；顺序均衡 5/5 全部为「是」。
  这与前面用 `/tmp` 脚本独立算出来的 282 配对 / 279 大于 1 / p = 9.6e-79
  **逐项吻合**——符号检验现在也由 `make_report.py` 自己算
  （新增 `sign_test()`，精确二项、`Fraction` 不过浮点），不再是手算转录。
  **但报告里必须同时写清两件事，否则这 70 行会被读成「70 项都测出了加速大小」**：
  ① 判据给的是**方向**不是**幅度**——`1.01–1.04×` 的意思是「4 次配对都偏快」，
  不是「已经确定快了 2%」；② 配对之所以能在幅度小于进程间离散度时仍然有信息，
  是因为每一轮的 base 与 opt 是**背靠背两个独立进程 + AB／BA 顺序均衡**，
  机器状态那部分噪声在相除时抵消了；**不配对**的比较才需要拿幅度去比 `进程间` 那一列。
  轮数只有 2–4，所以这是「4 次独立进程都同向」，不是「统计上不可能反号」。
- **§9.1 原先那句「本轮没有任何一行达到『测得稳定收益』」现在必须改成有例外。**
  它和它上面两条（「prefill 每侧 7 个样本都在同一个进程窗口内」、
  「decode 那一列是单窗口观察值」）都是 2026-10-23 那轮的事实，
  重采之后 decode 两侧都有 7 个窗口、配对实验也跑了。
  改法是**把两件事分开写**：§9.1 表里的倍数**仍然只来自 `ab/`**（每侧一个进程、
  进程内 7 次重复），所以那一列**仍然只能**支持「观察到加速」；
  「测得稳定收益」由 §7.3 的配对表单独判，两处各自引用自己的数据源，
  并明确写「这两列不能互相替代」。窗口数是**读出来的**
  （新增 `decode_window_counts` / `decode_single_window_rows`，
  复用 §2.3 已经在用的 `decode_median` / `single_window_note`），
  不再是硬编码的「单窗口观察值」。
  顺带修掉审查点 ⑤：`headline_numbers()` 里 decode 的比率原先用 legacy 标量
  `ms_per_token`，现在改用 `decode_median()`——数值 20/20 相同（早先已核对），
  但这才满足 `decode_median` 自己 docstring 里那条
  「表格里印的和比率算的必须是同一个量」。
- **优化 2a 与 2b 没有分别定价**：`raw_decode` 两行含两者之和。要补：加一个只关掉
  mask 跳过的探针。
- **有效 BLAS 线程数不可观测**：Python 里拿不到，harness 记为 `null` 而不是猜。
  在那之前，「请求 N 线程 → 实际用 N 线程 → 耗时变化」这条因果链**没有建立**。
- **Qwen2.5-Coder-0.5B 未进本轮性能 A/B**：只有集成测试（26 passed）与参考验证（PASS）。
  要补：把它加进 `run_round1.sh` 的 `MODELS` 重跑。
- **图文会话的 KV 复用**：优化 3 显式拒绝图文会话（不是静默退化）。要补：把 image token
  的稳定身份纳入前缀比较。
- 全部验证都在 macOS arm64 / Apple M5 Pro 上执行。**Linux x86_64 没有本轮记录**
  （那里只有 v0.1 的 Qwen3-0.6B）。换 OpenBLAS / MKL 会改变 matmul 与 elementwise 的
  相对成本，所以**归因比例不能照搬到别的后端**，虽然两项改动在任何后端上省掉同样的工作。
- Qwen3-1.7B 的集成测试耗时受宿主机状态支配：同一份代码在这台机器上
  既跑出过 115.37 s 也跑出过 3828.08 s。**内存压缩／换页仍是一个*假设*，不是已验证的原因。**
  证据是**不对称**的，必须分开说：3828.08 s 那一个窗口**采到了点样本**
  （`PhysMem: 23G used / 10G compressor / 137M unused`、`vm.compressor_bytes_used` 约 9.3 GiB、
  pytest 逻辑 RSS 5.98 GiB 而 `top` 常驻只有 126 MiB、CPU 时间/墙钟 = 24%），
  115.37 s 那一个窗口**什么系统指标都没采**，只记录了"当时没有其他负载"。
  所以缺的是：快那一侧的对照指标、任何一侧的**时间序列**、swap 用量，
  以及最根本的一条——**变量是被观测的，不是被操纵的**，两次运行也不相邻交替。
  要补：要么在两个窗口内同步采集上述指标再归因，要么**操纵**内存压力做受控实验。
  耗时不能用来判断代码快慢，正确性结论不受影响。详见[测试与数值验证](validation.md)。
- **同一份代码跑两遍，端到端有四行相差超过 0.10×**（Qwen3-1.7B 的
  `long-prompt-short-answer` 在 `ab/` 里是 0.59×、补跑后是 1.29×）。两遍都不是唯一真相；
  报告用补跑那一遍，**仅仅是因为它五个负载都测全了，不是因为它更快**。
  本条早先写作「`ab/` 那一遍报出的 0.59× / 0.71× / 0.84× 不是回退」——
  **那句话撤回**：只凭「补跑更快」就断言旧结果不是回退，而两次运行之间没有留下任何
  可校验的证据。这也是每张端到端表都带 `decode 自检` 列的原因——在这台机器上，
  一个孤立的端到端数字不能单独作为结论。
- **审查修复后的性能数据已全部采齐（4 个驱动、5 行 `rc=0`），但报告尚未据此重生成。**
  修复后的源码已于 2026-10-26 16:20:15 启动重采（`bash tools/perf/run_recollection.sh
  artifacts/perf-round1-rerun`，源码身份摘要
  `32fcc329fa440d9bf39ee34bc560664f74a87414a8211e362e4b768a39e92bc7`，
  严格串行：`run_round1` → `run_round1_longctx` → `run_noise_floor 4` → `run_projection`）。
  `recollection-status.txt` 现在是五行 `rc=0`：step 1 **17:13:31**、step 2 **17:34:16**、
  step 3 **19:06:32**、step 4 两次（**20:20:03** 与 **20:33:35**，见下条）；
  `noise-floor/` **40/40** 份 JSON 已齐，`projection/` **5/5** 份已齐。
  所以**性能数据本身不再欠采**，欠的是下游：
  ① `make_report.py` 里对新数据不成立的硬编码结论（已改完 ⑤⑦⑨⑩⑪⑫ 六处，
  ①②③④⑥⑧ 与三处路径不匹配仍在），
  ② `docs/推理加速-第1轮-2026-10-23.md` 还是从旧数据生成的，
  ③ 桌面交付报告的倍率表也还是旧批次的数字。
  要补：改完剩下的 `make_report.py`，再手动跑
  `.venv/bin/python tools/perf/make_report.py --data artifacts/perf-round1-rerun`
  （驱动**故意不串**这一步：半成品输出根会生成一份满是 `MISSING` 的报告，
  覆盖掉现在这份完整的）。
- **优化 1 的直接定价采了两次，两次不一致，而这个不一致本身就是结论。**
  第 1 次 20:09:04–20:20:03，第 2 次 20:26:32–20:33:35（第 2 次是因为第 1 次之后
  才给探针加上 `validity` 自检，要拿带自检的版本重采一遍）。
  512 token 上「一次投影的净代价」与「占 base 引擎 prefill 的比例」：

  | 模型 | 第 1 次 | 第 2 次 | 占比第 1 次 → 第 2 次 |
  | --- | --- | --- | --- |
  | SmolLM2-135M | 0.92 ms | 1.31 ms | 0.59% → 0.84% |
  | SmolLM2-360M | 3.07 ms | **0.67 ms** | 1.01% → **0.22%**（差 4.6×） |
  | Qwen2.5-0.5B | 5.49 ms | 3.94 ms | 1.57% → 1.13% |
  | Qwen3-0.6B | 3.37 ms | 6.89 ms | 1.00% → 2.04% |
  | Qwen3-1.7B | 负值 | 负值 | 两次都测不出来 |

  **为什么差这么多，看两个组成部分就明白了**：SmolLM2-360M 在 512 tok 上，
  `forward` 中位数两次是 546.24 / 541.24 ms（差 0.9%），`encode` 是 541.37 / 540.55 ms
  （差 0.2%）——**两个量各自都稳到 1% 以内，它们的差却从 4.87 ms 变成 0.69 ms**。
  四个可用模型全都是这个形状（组成部分差 0.1%–2.7%，差值差到 7×）。
  这个探针测的就是那个差，所以它的**分辨率比它两个组成部分的离散度粗一个量级以上**。
  结论：**「优化 1 一次投影值多少毫秒」在这台机器上定不下来**，能说的只有
  「占 512-token base prefill 的 0.2%–2.0%，即**个位数百分比的下半段**」，
  而且要带上两次之间的这个跨度。已发布那轮印的 153% 占比早先已被撤回；
  这一轮的数据支持的是一句**弱得多**的话，报告必须照弱的写。
  **一处必须交代的证据损失**：第 2 次把第 1 次的同名 JSON **覆盖了**，
  那发生在 `run_projection.sh` 加上「已有 JSON 先归档到 `projection-attempt-N`」
  这道保护**之前**，所以第 1 次的逐样本数组已不可恢复。
  第 1 次剩下的证据是它的 stderr（`run_projection.log`，中位数与占比都在里面，
  够复现上表全部数字），已在 `projection/README-ATTEMPT-1-OVERWRITTEN.txt` 里指明。
  这条正撞在审查要求的「保留旧原始证据，不覆盖后再假装旧结果不存在」上：
  第 1 次的**数字**没有假装不存在，但它的**原始样本**确实被覆盖掉了，只能记为已损失。
- **重采那批数据确实只描述一棵树——这一条现在是核过的，不是声明的。**
  三个独立检查，全部只读 JSON 与 git 对象，不跑任何计时：
  ① `noise-floor/` 的 40 份 + `ab/` 10 份 + `ab-longctx/` 10 份 + `vision/` 2 份 +
  `documents/` 2 份，`source_identity.digest` **只有两个值**：
  base 侧一律 `0848028740a00d3c…`（20 份）、opt 侧一律 `32fcc329fa440d9b…`（20 份），
  **后者与本条上面记的采集起始摘要逐字符相同**——两个半小时的采集里源码没动过。
  ② base 侧 15 个文件，`git show 8968e3999954:src/slowq/<file>` 的 sha256
  **15/15** 同时等于磁盘快照与 JSON 里记录的值，所以
  「基线优先从明确的 Git revision 可重建」这条要求**已满足并可复核**：
  `git checkout 8968e399995418e0514f1180767269b6bc46b24d -- src/` 就能重建它，
  不依赖 `artifacts/perf-round1/baseline-tree/` 这唯一一份本地拷贝
  （该目录另有 `BASELINE-REVISION` 写着同一个 SHA）。
  ③ **40 份 noise-floor JSON 的 `source_identity.harness.sha256` 全部是同一个值**
  `6fbf43125d85cbd7fc81e77282388af5dc38c118c412f423e0de05aac98b374a`，
  `harness.path` 一律指向当前树的 `tools/perf/core_ab.py`；
  `vision_cache_ab.py` 是 `8616656ba11a6eff…`、`document_index_ab.py` 是 `f7ae2a94f8db777d…`。
  **一个必须说明的不对称，以及一条被我自己写错又改回来的话。**
  base 侧的 `source_identity.tooling` 是**空字典**，opt 侧记了全部 17 个
  `tools/perf/*.{py,sh}`；原因是基线快照目录**只 checkout 了 `src/`**
  （`ls` 只有 `BASELINE-REVISION` 与 `src`，没有 `tools/`），
  而 `tooling` 是在 `--src` 那棵树里扫出来的。
  本条早先由此写「『采集中途改工具会让摘要不一致』这句话只对 opt 侧成立」——
  **那半句是错的，会低估这套身份记录**：`harness.sha256` 是**独立于 `tooling` 的字段，
  两侧都记**，而两侧跑的确实是**同一个** `tools/perf/core_ab.py`
  （`run_noise_floor.sh` 里 `CORE="tools/perf/core_ab.py"` 写死当前树，只有 `--src` 换成基线树）。
  所以「改测量驱动」在**两侧都会被摘要抓到**。真正没被 base 侧钉住的只是另外 16 个
  `tools/perf` 文件——而 `core_ab.py` 的本地 import 只有 `source_identity`
  （`tools/perf/core_ab.py:116`），`make_report.py` / `run_reference.sh` **都不在测量路径上**。
  于是本轮把这两个文件的修改推迟到采集结束，**是谨慎，不是必要**：
  改它们本来就不会影响已采数据。推迟的真实好处只有一个——
  采集未完时跑 `make_report.py` 会生成一份满是 `MISSING` 的报告并覆盖现有那份完整的。
- **配对实验不覆盖优化 3（多轮 KV 复用）。** `run_noise_floor.sh` 传的是
  `--suite all --no-chat`，落地的 JSON 里**没有 `chat` 键**（两侧都没有）。
  所以 step 3 能判读的是 5 个 workload 的 `total_seconds` 与 4 个 micro 探针，
  **优化 3 的 1.30×／1.14× 那两行仍然只有 `ab/`、`ab-longctx/` 的单窗口证据**。
  要补：给 `run_noise_floor.sh` 加一个跑 `chat` 的变体，或单独写一个会话配对的驱动。
- **`long-prompt-short-answer` 在两个 SmolLM2 上被 harness 主动跳过，不是漏测。**
  JSON 里写的是 `skipped: "prompt (1662 tokens) + max_new_tokens (32) = 1694 exceeds
  max_context=1024. Raise --max-context or lower --max-new-tokens; SlowQ never truncates."`
  —— 那句 **1662 token 是 SmolLM2 tokenizer 的长度**，别的模型同一份文本不是这个数
  （Qwen3-0.6B 是 584）。`run_round1_longctx.sh` 的头注释把它当成通用数字写，要改。
  于是 4 个完整模型的 workload 行是 **4×5−2 = 18** 行而不是 20 行。
- **采集期间机器不是空闲的。** `recollection-environment.txt` 记录了 load average
  2.05 / 2.16 / 2.35，以及 CloudShell（101.8% CPU）、Codex Renderer（92.2%）、
  一个 Virtualization 虚拟机（11.5%）在跑；`vm.swapusage` 是 `total = 0.00M`。
  这批数字因此**只能与同批次的自检列一起读**，不能与已发布那批直接比大小。
- **集成与参考验证的执行记录早于最终源码，已按最终树各重跑一次。**
  两者原先分别在 15:24 与 15:32 采集，而 `src/slowq/vlm.py` 的两处 docstring 最终改写在
  16:14、`tests/test_perf_tooling.py` 的 8 项身份区间测试在 16:13。
  性能采集的启动（16:20:15）晚于全部改动，但集成与参考那两批没有。
  「晚于全部改动」**当时**由 `find src tools tests docs -newermt '2026-10-26 16:20:15'`
  为空可证；**这条 `find` 现在不再为空，别再拿它当证据**——10-27 02:34–02:40 那轮
  就地注入的红跑顶掉了四个 `src/` 文件的 mtime（内容 `cp -p` 还原、`cmp` 逐个确认
  逐字节相同，`cp -p` 还原的是**备份的** mtime），其后 `tools/perf/`、`tests/`、`docs/`
  又因下面那条 §2.2 更正真的改过。**站得住的是哈希不是 mtime**：整树摘要 `32fcc329…`
  同时覆盖 `src/` 与 `tools/perf/` 两半、因此已经复现不出来，但 **`src/` 那 15 个
  逐文件哈希与重采时记下的那份 15/15 相符**——计时数据对应的正是这一半，
  核对命令见 `docs/validation.md`「这两批执行记录早于最终源码，差的是 docstring」那一段。
  差别只是 docstring 与测试文件，**不改变运行时行为**，不过这句判断本身要有执行记录，
  于是采集结束后按最终树重跑了两遍：
  **集成** 8 个模型 **208 passed、0 failed、`OVERALL_RC=0`**（逐模型 25–27 passed，
  日志在 `artifacts/perf-round1-rerun/integration-final/logs/`，时间戳 10-27 02:43）；
  **参考验证** **8/8 `numpy_rc=0 compare_rc=0`、860 个张量、`OVERALL_RC=0`**，
  8 份 `slowq.npz` 相对重采前备份逐字节相同
  （`artifacts/perf-round1-rerun/reference-final/`，详见下一条）。
  **当时同一条里还写了"8 份 `compare.log` 也逐字节相同"，那半句已撤回**：
  那一代的驱动从不写 `$dir/compare.log`，比的是上一代留下的文件与它自己的副本，
  理由与更正后的数字见下面那条与 `docs/validation.md` §参考验证：三进程协议。
  两遍都用独立输出目录（`integration-final/`、`reference-final/`）与新备份后缀
  （`pre-final`），旧的 `integration-fixed/`、`reference-fixed/` 与更早的备份都保留未覆盖，
  所以「重跑前 vs 重跑后是否一致」能从字节比出来，而不是靠记忆。
- **`tools/perf/run_reference.sh` 的 `worst_rel_l2` 列取错了维度——已拆成两列并重跑。**
  它原先 `grep -o 'rel_l2=[0-9.e+-]*'`，只能匹配 `compare.log` 里
  `[internal] SlowQ full prefill vs chunked prefill` 段的逐样本行；
  跨实现的逐张量表在 `[numerics] N expected tensors` 段，是**空白分隔**、没有
  `rel_l2=` 前缀，所以根本没被读到。于是旧 `summary.txt` 里 Qwen2.5-0.5B 印的是
  `worst_rel_l2=0.000e+00`（内部一致性：分块 prefill 与整块逐位相同），
  而该模型的跨实现最差 rel_L2 是 **2.719e-05**——两个量差 4 个数量级，名字却一样。
  现已拆成 `worst_numerics_rel_l2`（`[numerics]` 表第 4 列，`awk` 取科学计数法字段）
  与 `worst_internal_rel_l2`（`[internal]` 段的 `rel_l2=` 值）两列，
  并按最终树重跑，新汇总在 `artifacts/perf-round1-rerun/reference-final/summary.txt`：
  **8/8 `numpy_rc=0 compare_rc=0`、`OVERALL_RC=0`、860 个张量**，跨实现最差 rel_L2
  是 Qwen2.5-0.5B 的 **2.719e-05**（门限 1e-3），内部一致性最差是 Qwen3-1.7B 的
  **1.750e-05**；8 份 `slowq.npz` 相对本轮重采前的备份
  （TAG `pre-final`，即 15:32 那批）**逐字节相同**，即最终树原样复现参考。
  **这一遍汇总表里的 `log=IDENTICAL` 后来被判定为无效证据**，见下一条。
  重跑用独立输出目录 `reference-final/`、新备份后缀 `pre-final`，
  旧的 `reference-fixed/` 与 `pre-review`／`pre-round1` 备份都保留未覆盖。
  （采集期间故意不改这个工具，因为 `source_identity.tree_hashes` 覆盖
  `tools/perf/*.{py,sh}`，采集中途改工具会让各步记录的 tooling 摘要不一致；
  采集结束后才改、才重跑。）
- **`tools/perf/run_reference.sh` 的 `log` 列拿文件与它自己的副本比——已修好并按新格式重跑。**
  上一条把 `worst_rel_l2` 拆成两列之后，汇总表的**四个列**（`tensors=`、两个
  `worst_*_rel_l2=`、`log=`）仍然全部从 `$dir/compare.log` 解析，而驱动重定向的是
  `$OUT/<model>-compare.log`——`$dir/compare.log` 它从来不写
  （`validate_reference.py compare` 对抓取目录只读，全工具 4 处写入都在抓取一侧）。
  于是那四列描述的是**磁盘上碰巧躺着的那一代**，而与它们并排的 `npz=` 描述的是本次运行，
  两列看起来是同一句话。更糟的是 `log=IDENTICAL` 由
  `cmp "$dir/compare.log" "$dir/compare-$TAG.log"` 决定，后者正是脚本几行之前从前者
  `cp` 出来的：**一个拿文件和它自己的副本比的列不可能失败。** 它确实没失败，也确实错了——
  SmolLM2-135M 本次运行的日志与那份陈旧日志差一行（`[compare] took 0.1 s` → `0.0 s`），
  汇总表仍印 `log=IDENTICAL`。三个数值列的取值**碰巧**全对（逐个 diff 过八个模型），
  因为数值没变；那正是这一列不能留下的理由：它报告了自己从未测量过的一致性。
  现在：驱动自己 `cp` 一份到 `$dir/compare.log`（`docs/validation.md` 里
  "最近一次 `compare` 的完整输出"这行因此对每次运行都成立）；四列全部从
  `$OUT/<model>-compare.log` 解析；日志列只滤掉 `[compare] took <n> s` 那行墙钟
  （它测的是机器不是数值），并改名 `log_no_timing=`，让过滤写在读者真正会看的列名上；
  `summary.txt` 头部记下备份后缀与比较规则，`IDENTICAL` 因此可归属、可复算。
  按新格式重跑在 `artifacts/perf-round1-rerun/reference-final2/`（TAG `pre-final2`，
  10-27 04:57）：**8/8 rc=0、860 张量、`OVERALL_RC=0`**；8 份 `slowq.npz` 用 `cmp`
  逐个确认与备份逐字节相同；日志 8 份里 **7 份逐字节相同**，SmolLM2-135M 差
  **恰好一行**计时，滤掉计时行后 **8/8 相同**（这就是 `log_no_timing=IDENTICAL` 的含义）。
  与 `reference-final/summary.txt` 去掉 `log=`／`log_no_timing=` 列后 `diff`，
  **其余每一列逐个相符**。三份 `summary.txt` 用三个独立输出目录、三个新备份后缀，
  被取代的两份**原样保留**，旁边各放一份 `LOG-COLUMN-CORRECTION.txt` 写明哪一列
  当时没在测任何东西、以及更正后的运行在哪里。
  测试（先红后绿）：`test_the_reference_summary_is_derived_from_the_log_this_run_produced`、
  `test_a_tagged_reference_summary_reproduces_from_its_own_logs`（用 Python 从日志
  逐列复算，不经 shell）、`test_a_superseded_reference_summary_is_kept_and_says_why`。
  红跑留档 `falsification/reference-log-self-comparison.log`：缺陷版**复制到**
  `/tmp/falsify/r1/tools/perf/run_reference.sh`（用 `/tmp/falsify/build_r1.py` 从当前文件
  反向重建，`diff` 确认只差那四处修改），由 pytest 插件在 `pytest_collection_finish`
  里换掉 `TOOL_DIR`／`REFERENCE_DIRS`，**3 failed**；仓库那份文件跑前跑后 sha256 都是
  `b3ac8eecc5be2a0e88bbee8aa8d1978de0c6fa395c2ebbee1a2757634b6815d6`。
  这一次没有"`cp` 还原 + `cmp` 确认"那一步，因为工作区那份文件**从头到尾没被改过**。
- **§2.2 那句「新 `compare.log` 与 pre-round1 备份逐字节相同」也是过度断言——已改成从磁盘派生。**
  拆完 `run_reference.sh` 的列之后回头查文档散文，发现同一类错误还有一处，而且是在
  **权威文档**里：`make_report.py` 把这句当字面量印进 §2.2 的表。逐模型重算
  （`cmp` + `diff`，不读任何汇总表）：`slowq.npz` 对 `slowq-pre-round1.npz` 是
  **8/8 逐字节相同**；`compare.log` 对 `compare-pre-round1.log` 是 **7/8**，
  唯一的例外 SmolLM2-360M-Instruct-slowq 差**恰好一行**
  `[compare] took 0.1 s` ↔ `0.0 s`（`diff` 输出 `185c185`，两侧各 188 行）——
  比较器给自己计的那只秒表；滤掉该行后 **8/8**。
  与中间三代（`compare-pre-review` / `-pre-final` / `-pre-final2`）比则是 **6/8**，
  SmolLM2-135M 在那里也不同。**所以这个计数依赖"说的是哪一代"**，
  写死在脚本里必然过时。现在生成器新增 `reference_identity()` /
  `timing_only_difference()` / `reference_identity_cell()`，§2.2 那一格由它们从
  `artifacts/validation/*/` 现算，并按 `；` 分成「npz 的计数」与「`compare.log`
  的原始计数 + 滤掉计时行后的计数 + 把两者分开的那个模型名」两句；
  抓取产物不在树上时（`artifacts/` 是 gitignore 的）改印「**未在本地复算**」而不是印一个
  没人测过的数。列名与 `run_reference.sh` 的 `log_no_timing=` 用同一套口径。
  **这一处不是"撤回"而是"补上计数"**：8 份里 7 份真的逐字节相同，那句话不是空的，
  只是没限定——而一个含秒表的文件上的无限定逐字节断言，正是 `log=` 那一列的散文版。
  测试（先红后绿）：`test_the_pre_round1_identity_row_counts_the_models_it_actually_covers`，
  红跑留档 `falsification/preround1-blanket-byte-identity.log`。
  **这是这批反证里唯一不需要注入的一份**：缺陷就是生成器当时印的那句字面量，
  新测试直接对它变红。`make_report.py` 的 sha256 由
  `8e6d5e28157568c4be7b13f70da494d003c3ce88a025ec896b3fa96a02cbb321`（6017 行）
  变为 `209b4d113565cd9686d8d1ae3a013620fae793c2fa353a5ada6cd62c7f4a672e`（6135 行），
  权威文档 §1.5 的自识别行随之更新，文档重生成两遍字节相同。
  `artifacts/` 下一个字节都没动，所以红跑与绿跑量的是同一批文件。
- **四项审查修复的"改坏应变红"记录——已补完。** 对每一项把实现改回有缺陷的版本、
  重跑上面点名的那个测试、确认它变红、再还原，日志都在
  `artifacts/perf-round1-rerun/falsification/`（`review1a-…`、`review1b-…`、
  `review1c-…`、`review2-…` 四份），逐条的「变红 / 总数」见上面
  「审查修复」一节末尾的表格。还原后都用 `rm -rf src/slowq/__pycache__` 清过字节码：
  CPython 的失效判据是源文件 mtime + size，`cp -p` 会把 mtime 一起还原，
  一次等长替换在同一秒内还原后 `.pyc` 仍被当成有效缓存——本轮核对
  `test_the_identity_docstring_bands_cover_every_run` 时实际踩到过一次，
  详见 `tools/perf/README.md` 的规矩 4。四次还原都以 `cmp` 确认与备份逐字节相同，
  且每次都把全套快速测试重跑回 991 passed。

## v0.2.0

从"只支持 Qwen3-0.6B 的文本生成"扩展到 **8 个模型、3 类任务**（文本生成、文本向量、
图文生成）。本轮改动、验证状态与未完成项见 [v0.2 接入报告](v0.2-多模型接入报告.md)，
逐个模型的状态见[模型支持](models.md)。

### 模型与任务抽象

- 把"架构家族"与"任务类型"拆成两条独立的轴。家族由 `config.json` 的 `model_type`
  决定，规则集中在 `config.py` 的 `TEXT_FAMILIES`（qwen3 / qwen2 / llama / idefics3）；
  任务写进转换后目录的 manifest，**不从 config 猜** ——
  `Qwen/Qwen3-Embedding-0.6B` 公布的 `model_type` 与 `architectures` 和聊天模型取值一致。
- 删除代码里按模型名分支的地方。原来写死"有 Q/K RMSNorm""QKV 有 bias""head_dim 可推导"
  的断言，改成按家族声明：qwen2 的 Q/K/V bias 恒开、O 投影恒关（硬编码在官方
  `Qwen2Attention.__init__`，config 里没有字段）；llama 的三种 bias 都从 config 读；
  qwen3 要求 `head_dim` 必须显式给出，缺失时报 `ConfigError` 而不是回退到
  `hidden_size / heads`（Qwen3-0.6B 的 head_dim 是 128，而 `1024/16 = 64`）。
- sliding window 按**实际启用状态**判断：两个 Qwen2.5-0.5B checkpoint 的 config 里都有
  `sliding_window: 32768` 且 `use_sliding_window: false`，此时跑全注意力；
  既不因为字段存在就拒绝加载，也不会把真正启用的滑动窗口当成全注意力。
- 新增算子：`layer_norm`、`gelu_tanh`、`conv2d_patchify`、`pixel_shuffle`、
  `l2_normalize`；`linear` 增加可选 bias。视觉塔的注意力传 `mask=None`（非因果，
  且单图路径不产生 padding）；真出现部分 padding 的 patch 网格时报错而不是屏蔽。

### manifest 与离线目录

- manifest `format_version` 升到 2，新增 `task` 与 `model_family` 两个键，两者都必须存在。
  版本 1（只有 v0.1 写过）仍可读，缺键时按 v0.1 的唯一可能取值处理为文本生成；
  其他版本号被拒绝，不会静默降级。**v0.1 转好的 Qwen3-0.6B 目录不需要重新转换。**
- 转换器新增 `--task auto|text-generation|text-embedding|image-text-generation`。
  `auto` 读文件而不是读仓库名：sentence-transformers 导出的三个标记文件
  （`modules.json`、`config_sentence_transformers.json`、`1_Pooling/config.json`）
  表示向量模型，嵌套 `vision_config` 加 processor 文件表示图文模型；
  三个标记文件只出现一部分时报错而不是猜。显式指定的 task 仍会与文件证据交叉校验。
- 每个 repo 解析并记录**自己当时**的 commit，不再套用 v0.1 为 Qwen3-0.6B 固定的 revision。
- 图文目录额外保存 `preprocessor_config.json`、`processor_config.json`、`chat_template.json`
  与 `added_tokens.json`，使离线预处理不需要网络。
- 并非所有任务都需要 `generation_config.json` 或 chat template；缺文件时按任务判断是否致命。

### 新任务：文本向量

- 新增 `src/slowq/embedding.py`：`EmbeddingEngine`、`EmbeddingModel`、`EmbeddingResult`、
  `format_query`、`last_token_pool`、`cosine_similarity`、`similarity_matrix`。
- 编码路径与官方 snippet 一致：query 加 `Instruct: …\nQuery:…` 前缀（`Query:` 后无空格），
  document 不加；`add_special_tokens=True`；保留官方因果注意力（不是双向）；
  取最后一个有效 token 的 final hidden state（不是平均池化）；L2 归一化；
  输出官方全维度 1024。
- **不做全词表投影，也不做自回归生成。** KV cache 按当前文本长度分配，不按 `max_length` 预分配。
- 超过 `max_length`（默认 8192）时报错，不静默截断。
- 新增 `slowq embed` 子命令：`--query` / `--document` 可重复，逐条编码并输出两两余弦相似度。

### 新任务：图文生成

- 新增 `src/slowq/vlm.py`：`VisionEngine`、图像预处理、ViT 视觉塔、pixel shuffle、
  connector、图像特征注入、图文 prefill 与复用 KV cache 的文本 decode。
- 完整实现官方单图切分：resize 到最长边 2048 再向上取整到 512 的倍数，切成 512×512 的
  rows×cols 个 crop **加 1 个全局帧**，每帧经 pixel shuffle 后是 64 个 image token，
  crop 之间插入 `<row_r_col_c>` 标签。"只支持一张图"不等于跳过这张图产生的多个 crop。
- 视觉塔是与文本塔不同的一套：LayerNorm（weight + bias）而不是 RMSNorm，
  fc1/fc2 + GELU-tanh 而不是 SwiGLU，q/k/v/out_proj 都带 bias，
  全连接**非因果**注意力，位置来自学习到的 `position_embedding`，没有 RoPE。
- 新增可选 extra `slowq[vision]`（Pillow），只有 `slowq.vlm` 会用到，且只在真正读图时导入。
  **纯文本安装仍然是 numpy + tokenizers + Jinja2 三个运行时依赖。**
- 新增 `slowq vlm` 子命令；`GenerationMetrics` 增加 `vision_encode_seconds`，
  只有图文请求会打印这一行，文本请求的输出不变。
- 图像缺失或读不出、占位符数量对不上、扩展后超过 `max_context`，都给出明确错误，
  不静默截断或忽略图像。一次请求只接受一张图，第二张报错。

### CLI 与工具

- `slowq inspect` 按 manifest 记录的任务分派，三种任务的目录都能查。
- `Engine.from_directory` 拒绝向量与图文目录，`EmbeddingEngine.from_directory` 拒绝
  文本生成目录，两边都在错误信息里指出正确入口。
- `tools/validate_reference.py` 重构为多家族多任务：抽样层按家族与层数选择
  （不再固定第 27 层），抽样权重张量跟着家族走（qwen3 取 q/k norm，qwen2 取 q_proj.bias，
  llama 两者都不取），artifact 目录按模型名分开（`artifacts/validation/<模型目录名>/`），
  artifact 格式升到 3。新增图文任务的逐段验证：官方 processor 的 pixel 张量与 mask、
  视觉塔输出、connector 输出、图文融合位置、最终 logits 与 cached decode。
- 参考侧的图像由 NumPy 侧生成并存成 PNG，参考侧按 sha256 校验，
  两侧证明看到了同一份字节；图像不下载也不入库。

### 测试

- 新增 `tests/real_models.py` 作为真实权重测试的共享管道；集成测试拆成四个模块，
  按 manifest 里的任务过滤：`test_integration_reference.py`（无任务过滤）、
  `test_integration_real_model.py`、`test_integration_embedding.py`、`test_integration_vlm.py`。
- 新增快速测试 `tests/test_embedding.py`、`tests/test_vlm.py`；`tests/conftest.py` 增加
  按家族构造微型 checkpoint 的夹具，覆盖有/无 QKV bias、有/无 Q/K norm、
  head_dim 推导 vs Qwen3 的独立 head_dim、tied/untied 权重、chat template 与停止符、
  缓存位置与上下文边界、full 与 chunked prefill 等价、向量输入格式与归一化、
  视觉 patch 顺序 / mask / connector / image token 注入、错误路径与新模块的依赖边界。
- v0.1 的检查没有被删掉，负向用例保留。
- `tools/benchmark.py` 仍然只测文本生成；向量与图文的耗时通过各自 CLI 的 JSON 输出测量。

### 发布前修复（审查后）

审查发现 5 个问题，全部修掉并各自补了能证明原缺陷的回归测试。

- **向量分块等价测试此前是空转的。** 它构造了带小 `prefill_chunk_size` 的引擎，
  然后调 `model.encode_ids(ids)` —— 分块大小没有传下去，7 个用例全部退化成
  一次 50 token 的整块 prefill。现在每个用例都断言 decoder **实际看到的分块长度**，
  覆盖 `chunk=1`、不能整除的长度和最后一个短块，既比数值也比分块确实发生。
- **manifest 的 `model_family` 此前不与 `config.json` 的 `model_type` 交叉校验。**
  把 SmolLM2 目录的 manifest 改成 `qwen3` 也照样加载。要说清楚的是，**解码规则一直
  来自 `config.json`**：实测该目录的 `config.family` 仍是 `llama`、`qk_norm` 仍是
  `False`，所以权重不会因此被算错，也不会多出 Q/K RMSNorm。真正的缺陷是这个目录
  从此自相矛盾而没有任何一方拒绝它 —— `slowq inspect` 会照 manifest 报出错误的家族，
  而 `config.json` 被换成另一家族时权重确实会被按错误规则解码。现在不一致时抛
  `ConfigError`，说明两边各声称什么、并指出重新转换的命令；
  版本 1 的 manifest 没有这个键，仍然可读，不会被要求补上它当年没写过的字段。
  图文目录**不走这个函数**：`slowq.vlm` 有自己的 `load_config_and_weights`，
  在自己的加载器里分别校验 `config.json` 的 `model_type` 与 manifest 的
  `model_family`，两侧都必须等于 `idefics3`。
- **向量参考侧此前把 SlowQ 拼好的字符串当成输入。** 参考侧对 SlowQ 记录的
  `text` 做分词，于是一个写错的 query 前缀会被两侧同时继承，比较照样 `PASS`。
  现在参考侧从原始 query / document、instruction 和角色**自己重建**输入，
  不调用被测的 `format_query`，也不把 SlowQ 的最终文本当作事实来源；
  官方格式 `Instruct: {instruction}\nQuery:{query}`（`Query:` 后无空格）与
  "document 不加任何前缀"都由参考侧独立断言。抓取配置新增
  `embedding_input_format=2`：**只影响向量任务的证据**，旧向量 artifact 会被
  明确拒绝并提示重抓 `ref`，生成与图文的 artifact 版本不变、无需重抓。
- **删除了 `ops.padding_attention_mask`。** 它对 `np.array([True, False])` 抛
  `IndexError`。先确认了运行时、工具、测试与文档里都没有调用者 ——
  视觉塔走非因果注意力且单图路径不产生 padding，真出现部分 padding 的 patch
  网格时报错而不是屏蔽 —— 所以按"不留无人使用且已知有缺陷的代码"处理，
  而不是修好一个没有调用者的函数。
- **图文请求的校验此前发生在视觉编码之后。** `Engine.stream_ids` 是生成器函数，
  它的校验要到第一次 `next()` 才执行，而 `VisionEngine.stream_prompt` 是普通函数，
  于是"要 0 个新 token"这种非法请求会先把视觉塔跑完一遍才报错。校验逻辑抽成
  `Engine._validate_request`，两条路径共用同一份实现，图文路径在编码图像**之前**调用它。

### 验证工具修复（第二轮审查后）

只改验证工具、测试与文档，没有动前向计算、采样行为、数值容差或支持范围。

- **`compare` 此前会对缺失的输入字段给假 PASS。** 向量 artifact 的比较用
  `ours_entry.get(field) == theirs_entry.get(field)` 判 `raw` / `is_query` / `text`，
  于是两侧都缺同一个字段时两边都是 `None`，`None == None` 成立——
  "两边都没说自己嵌入了什么"被判成"两边说的一样"，打印 `input=ok` 并以退出码 0 通过。
  现在逐侧逐字段判"键在不在 / 是不是 null / 类型对不对"，不合格的记 `input=INVALID`、
  写进 problems、退出码非 0，错误信息带上哪一侧、哪个样例、哪条文本、哪个字段。
  `is_query` 用 `isinstance(value, bool)` 判而不是和 `True` 比相等：Python 里 `bool`
  是 `int` 的子类，JSON 的 `1` 会被当成"是 query"。原有的数值、tokenizer、权重摘要、
  pooling 与格式版本检查一条没减。
- **生产的参考抓取函数此前没有测试真的执行过。** `_reference_embedding_capture`
  是"参考侧自己重建输入"这条修复的核心，但把它的函数体第一句换成 `raise AssertionError`，
  当时那 91 项测试全绿。现在 4 条用例直接调用它：mock 的只是它外围的 torch 接口、
  tokenizer（记录被喂了哪些字符串）与一个返回 `(batch, seq, hidden)` 的假模型，
  没有在测试里另写一份"正确的抓取"；8 条官方字符串逐字写在测试里，
  不调用被测的 `reference_embedding_texts`。

### 结果

- 快速测试：发布前修复轮 **665** 项通过，验证工具修复轮 **685** 项通过
  （无网络、无真实权重；+20 全在 `tests/test_validate_reference.py`，91 → 111）。
- 8 个模型的真实权重集成测试合计 208 项通过、0 失败，逐目录单独设置 `SLOWQ_MODEL_DIR`
  ——**发布前修复轮的结果**，验证工具修复轮没有重跑，8 份原始日志已留档在
  `artifacts/repair-review-20261021/developer-integration-logs/`（sha256 与审查方索引一致）。
  其中 Qwen3-1.7B 该轮耗时 4036.71 s。**完整正确性回归已通过；异常耗时与严重内存压力、
  页面调入调出相符，尚未通过资源充足环境的对照测量排除其他性能因素。**
  权重是 `np.load(..., mmap_mode="r")` 加载的，页面驻留由操作系统管理，
  映射不等于常驻，所以峰值 RSS 6.43 GiB 是观测值而不是结构性推论；
  `sys` 845 s 是 `user` 126 s 的 6.7 倍、CPU 利用率 24% 与之一致。
  62.7 s 那个量级是**接入批次**的实测，不是修复轮的。
- 8 个模型的官方参考数值比较全部 `PASS`：6 个生成模型各 116 项，向量与图文各 82 项，
  合计 860 个张量，greedy argmax 位置全部一致。
  参考环境 transformers 4.51.3 / torch 2.14.1 / eager / FP32 / CPU。
  向量那一侧的证据在发布前修复轮按新的 `embedding_input_format=2` **重新抓取**；
  验证工具修复轮对同一批 artifact **重新比较**，8 / 8 仍 `PASS`、仍是 860 个张量。
- 干净环境安装验证（发布前修复轮）：wheel → 两个新建 venv（无 Pillow 时 660 passed / 5 skipped，
  装上 `[vision]` 后 665 passed / 0 skipped），`torch` / `transformers` 始终不存在，
  `pip check` 干净，三类任务各跑通一条真实离线命令。
- 全部在 macOS arm64 / Apple M5 Pro 上执行；v0.2 新增模型没有 Linux 验证记录。

## v0.1.0

首个实验版本支持 Qwen3-0.6B 的 CPU / FP32 推理。以下为公开提交前完成的主要修复。
详细验证方法见[测试与数值验证](validation.md)。

### 权重转换与下载

- 转换先在暂存目录完成并自检，再安装到目标路径；默认拒绝覆盖非空目录。
- 接受预先创建的空输出目录，安装时拒绝覆盖后来写入其中的文件。
- 下载缓存绑定模型仓库与完整 commit；记录未完成状态，中断后可重跑命令补齐文件。
- 缓存文件按大小和可用的 SHA-256 校验，无来源标记的目录仍会被拒绝。
- 跨分片的 tied weights 分别读取各自分片进行比较。
- 截断的 checkpoint 给出张量、分片和字节范围错误，避免清理 mmap 时掩盖原始异常。

### 生成与数值验证

- 流式生成结束时，将解码器 flush 的尾部文本一并交付，保持流式与完整输出一致。
- `DebugProbe` 按指定位置捕获 logits，支持非末尾位置。
- 数值比较检查预期张量集合、形状和有限性；缺失数据、空集合、NaN / Inf 都会失败。
- tokenizer 样例与权重摘要也检查预期集合，双方同时遗漏同一项时仍会报错。
- 集成测试重新计算当前 tokenizer 输出和权重摘要，不复用历史布尔结论。

### 基准与发布文件

- 分开测量 full / chunked prefill，报告实际前向次数。
- `--threads 4` 与 `--threads=4` 都在导入 NumPy 前设置线程环境变量。
- 补齐 MIT LICENSE，修正转换依赖说明，并配置真实的 GitHub 仓库地址。

数值容差保持不变；测试结果及平台范围见[验证记录](validation.md)。
更完整的早期修复复盘保留在[首次提交的 README](https://github.com/luolinglong0611/SlowQ/blob/b5548dca2faa8e85a078ebdec7d120d4940e3d7c/README.md)中。
