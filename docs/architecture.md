# 慢炖 · SlowQ 架构说明

本文档描述 SlowQ v0.1 的实际实现：模块边界、一次生成的完整数据流、关键张量
shape、KV cache 语义、计时区间定义，以及依赖边界。所有内容对应仓库中真实存在
的代码，不是设计意图。

本文已按 v0.1 的十个缺陷修复更新过。修复本身大多不改变架构，改变的是**工具在
失败时的行为**和**验证结论能不能被采信**，因此它们落在 §3.5（DebugProbe 位置
语义）、§6（两种 prefill 模式与线程报告）、§7（写入侧的安全约束）、§8（流式
flush 不变量）、§11（新增的工具层回归测试）、§12（`compare` 不给假 PASS）和
§14（跨平台复核）里。逐条的问题/修复/回归测试对照表见 README 的
"v0.1 修复记录"。数值路径没有改动，容差一个字都没放宽。

在那十项之后又做了一轮**发布收尾**（2026-10-19），三项：中断的下载可以靠重跑
同一条命令续上、空的 `--out` 不再被前后两次检查互相打脸、`compare` 的元数据
完整性检查不再被"两侧同时缺一条"绕过。它们同样不碰数值路径，落在 §7 和 §12，
本文里标注为"发布收尾 #1/#2/#3"，与前面标注为"v0.1 缺陷 #N"的十项区分开；
对照表见 README 的"v0.1 发布收尾修复记录"。

---

## 1. 模块与依赖方向

依赖是单向的，任何反向引用都是 bug：

```
cli.py ──> engine.py ──> model.py ──> ops.py
                │            │
                │            └───────> cache.py
                │
                ├──> tokenizer.py
                ├──> sampling.py ──> ops.py
                ├──> metrics.py
                └──> config.py <── weights.py
```

| 模块 | 职责 | 行数级别 |
| --- | --- | --- |
| `config.py` | 解析并校验 `config.json` / `generation_config.json`，派生维度 | ~300 |
| `weights.py` | 解析 `manifest.json`，只读 mmap 加载 `.npy`，别名与完整性检查 | ~300 |
| `ops.py` | 无状态数值算子：RMSNorm、RoPE、softmax、SiLU、GQA attention | ~190 |
| `cache.py` | 预分配 KV cache，`length` / `write` / `read` / `commit` / `reset` | ~145 |
| `model.py` | Qwen3 前向：embedding → 28 层 → final norm → logits；`DebugProbe` | ~415 |
| `tokenizer.py` | Jinja2 chat template、BPE 编解码、增量解码器 | ~310 |
| `sampling.py` | temperature / top-k / top-p，独立 RNG | ~155 |
| `engine.py` | 唯一的生成循环：分块 prefill + 逐 token decode | ~370 |
| `metrics.py` | 计时、内存、吞吐统计 | ~200 |
| `cli.py` | `generate` / `chat` / `inspect` 三个子命令 | ~370 |

另有 `errors.py`（异常层级，46 行）、`__init__.py`（公共 API 导出，77 行）、
`__main__.py`（10 行）。合计 13 个模块、2879 行。

**算子不得依赖 CLI，`engine.py` 不得依赖 `cli.py`。** `tools/` 下的脚本可以依赖
`src/slowq`，反之不行。

---

## 2. 一次生成的完整数据流

以 `engine.generate([{"role": "user", "content": "你好"}])` 为例：

```
messages
  │ tokenizer.render_chat()          Jinja2 ImmutableSandboxedEnvironment
  │                                  trim_blocks/lstrip_blocks=True
  ▼
"<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
  │ tokenizer.encode()               byte-level BPE，不加特殊 token
  ▼
prompt_ids: list[int]                长度记为 P（计入 tokenize_seconds）
  │
  │ Engine.stream_ids()
  │   1. 校验 P + max_new_tokens <= max_context      否则 ContextOverflowError
  │   2. kv_cache.reset()                            在请求开始时，不是结束时
  ▼
分块 prefill：for start in range(0, P, prefill_chunk_size)
  │   model.forward(prompt_ids[start:start+C], cache)
  ▼
logits: [vocab_size]                 只保留最后一个位置的 logits
  │ Sampler.sample()                 → 第一个 token（计入 first_token_seconds）
  ▼
decode 循环：
  │   model.forward([token_id], cache)               单 token 前向
  │   Sampler.sample(logits)
  │   IncrementalTextDecoder.push(token_id)          可能返回 ""（字节未凑齐）
  │   直到 token_id ∈ stop_ids 或已生成 max_new_tokens
  ▼
GenerationResult(text, token_ids, prompt_token_ids, stop_reason, metrics)
```

关键点：

- **`stream_ids` 是唯一真正实现循环的地方。** `generate` / `generate_text` /
  `stream` 都只负责把输入解析成 token id 再委托给它，`generate` 与
  `_drain(stream(...))` 等价。因此"prefill 后逐 token decode"只有一份实现。
- **最后一个采样出的 token 不会被喂回模型。** 它交给调用方后循环即结束，所以
  请求结束时 `kv_cache.length == P + generated - 1`。这不是 off-by-one bug，
  `tests/test_engine.py` 对此有专门断言。
- **文本输出永远不包含终止用的特殊 token。** `decode(skip_special_tokens=True)`
  加上增量解码器的 hold-back 共同保证这一点。

---

## 3. 单层前向与张量 shape

记 `T` = 本次前向的 token 数，`S` = cache 中的总 token 数（`offset + T`）。
以 Qwen3-0.6B 的真实维度为例：

```
hidden_size H          = 1024
intermediate_size I    = 3072
num_attention_heads Nh = 16
num_key_value_heads Nk = 8
head_dim D             = 128      # 注意 Nh * D = 2048 != H
vocab_size V           = 151936
num_hidden_layers L    = 28
```

`D` 从 `config.json` **读取**，绝不由 `H / Nh` 推导——Qwen3 的 `q_proj` 输出是
2048 维而残差流是 1024 维，推导会得到 64 并静默算错。

### 3.1 一层 decoder

```
x                      [T, H]

# --- attention 分支 ---
input_layernorm        rms_norm(x, w, eps)                     [T, H]
q_proj                 linear(·, W_q[H·→·Nh*D, H])              [T, 2048]
k_proj                 linear(·, W_k[Nk*D, H])                  [T, 1024]
v_proj                 linear(·, W_v[Nk*D, H])                  [T, 1024]
split_heads            [T, Nh*D] → [Nh, T, D]                   q: [16, T, 128]
                       [T, Nk*D] → [Nk, T, D]                   k,v: [8, T, 128]
q_norm / k_norm        rms_norm 作用在最后一维 D 上              [Nh, T, D] / [Nk, T, D]
apply_rope             x*cos + rotate_half(x)*sin               同上
kv_cache.write         写入 slot [offset, offset+T)
kv_cache.read          读出 [0, S)                              k,v: [8, S, 128]
group_query_attention  见 3.2                                    [16, T, 128]
merge_heads            [Nh, T, D] → [T, Nh*D]                   [T, 2048]
o_proj                 linear(·, W_o[H, Nh*D])                  [T, 1024]
x = x + attention                                               [T, H]

# --- MLP 分支（SwiGLU）---
post_attention_layernorm                                        [T, H]
gate_proj              linear(·, W_gate[I, H])                   [T, 3072]
up_proj                linear(·, W_up[I, H])                     [T, 3072]
silu(gate) * up                                                  [T, 3072]
down_proj              linear(·, W_down[H, I])                   [T, 1024]
x = x + mlp                                                      [T, H]
```

### 3.2 GQA：不物化 `repeat_kv`

HF 的做法是把 K/V 沿 head 轴复制 `Nh/Nk = 2` 份。SlowQ 改为把 query head 折叠
进 batch 轴：

```python
q_flat = q.reshape(Nk, group * T, D)            # [8, 2T, 128]
scores = q_flat @ k.transpose(0, 2, 1)          # [8, 2T, S]
```

query head `kv * group + r` 对应 KV head `kv`，与 `repeat_kv` 展开后的顺序逐元素
一致。`tests/test_ops.py` 用一个独立写出的 `naive_repeat_kv` 参考实现逐 head
对比，覆盖 `(Nh, Nk) ∈ {(16,8), (4,2), (4,4), (8,1), (4,1)}`。

这样做的收益是实打实的：`S = 2048` 时物化 `repeat_kv` 每层每 token 要复制
16 MiB，而折叠方案零复制。

### 3.3 权重布局

所有权重按 HF 的 `[out_features, in_features]` 存储在磁盘上，运行时计算
`x @ W.T`。转置是 numpy view，不产生拷贝，也不需要把 checkpoint 预先转置。

### 3.4 数值细节（都是踩过坑的）

| 细节 | 选择 | 原因 |
| --- | --- | --- |
| 掩码常量 | `np.finfo(np.float32).min` | 与 HF 一致；用 `-inf` 会让整行被掩的 softmax 出 NaN |
| softmax | 先减去行最大值 | FP32 下 `exp(88)` 就溢出 |
| sigmoid | 正负两支分别计算 | `1/(1+exp(-x))` 在 `x < -88` 溢出并触发 `RuntimeWarning`，SiLU 每层都会中 |
| RoPE 配对 | 半分割（`cat([-x2, x1])`） | Qwen3 用半分割，不是交错配对；搞错不会报错，只会算错 |
| RMSNorm | 方差在 FP32 累加 | 与 HF 一致 |
| 全程 dtype | FP32 | 任何算子都不允许出现 `.astype(np.float64)` |

### 3.5 DebugProbe：位置语义

`DebugProbe` 是可选的中间激活捕获（默认关闭：留住一次 2048-token prefill 的全部
激活要几个 GiB）。它有两处容易被写错的地方，都是 v0.1 修过的：

- **位置是相对于本次前向的输入块的**，`0` 是这一块的首 token，`-1` 是末 token，
  对所有捕获点统一。decode 步的块只有 1 个 token，所以那里只有 `-1`/`0` 合法。
  块里没有的位置是**调用方错误**，直接抛 `ValueError`：夹断或取模会安静地报出
  另一个 token 的激活。
- **`ForwardResult.logits` 只覆盖最后一个位置**，所以任何其他位置的 logits 需要
  各自做一次输出投影（`_record_probe_logits`）。旧实现在这里直接复用末位的
  logits，于是"第 5 个位置的 logits"其实是最后一个位置的——键名对，数值错，
  而参考对齐恰恰要比的就是这些位置（v0.1 缺陷 #7）。只算被请求的那些位置：
  全词表投影是整个模型里最贵的单个操作，没人要的位置不该产生开销。末位复用
  生成时已经算好的那一份，因此与 `ForwardResult.logits` 逐 bit 相同。

`tests/test_model.py` 里有一条测试专门构造非末位置的请求，断言捕获到的**不是**
末位那一行。

---

## 4. KV cache 语义

```python
KVCache(max_context, num_layers, num_kv_heads, head_dim)
  k, v: [L, Nk, max_context, D]     一次性预分配，FP32
```

Qwen3-0.6B 在 `max_context = 2048` 下：
`28 × 2 × 8 × 2048 × 128 × 4 B = 448 MiB`。每 token 229,376 B = 224 KiB。

三条不变量，`tests/test_cache.py` 与 `tests/test_model.py` 都有对应断言：

1. **`length` 只统计完整跑完一次前向的 token。** 每层写 `[offset, offset+T)`，
   读 `[0, offset+T)`；`commit(T)` 在**一次前向结束后调用一次**，不是每层一次。
   如果每层 commit，第 2 层就会看到自己刚写的数据被算作"历史"。
2. **cache 在每个请求开始时 reset，而不是结束时。** 中途抛异常的请求因此不会
   污染下一个请求。`reset()` 只把 `length` 归零，不释放也不清零内存——预分配
   的槽位会被下一次写入覆盖。
3. **容量不足必须抛错，绝不静默截断或覆盖。** `check_capacity` 在写入前检查；
   写入失败时 cache 保持原样（`length` 不变）。

存进 cache 的 K 是**已经过 q_norm/k_norm 和 RoPE 的**，所以分块边界不会改变任何
token 的绝对位置。`tests/test_model.py::test_chunk_boundaries_do_not_shift_positions`
直接比较 cache 中的 K 来验证这一点。

位置从 cache 长度推导，`forward` 拒绝与 `kv_cache.length` 不一致的显式
`position_offset`——分块 prefill 和 decode 因此不可能各自漂移。

---

## 5. 分块 prefill

`prefill_chunk_size`（默认 128）把长 prompt 切成若干次前向。第一次写
`[0, C)`，第二次写 `[C, 2C)` 但读 `[0, 2C)`，以此类推。mask 用**绝对位置**构造：

```python
causal_attention_mask(num_queries=T, num_keys=S, position_offset=offset)
# query i 位于绝对位置 offset + i，可以看见 key j 当且仅当 j <= offset + i
```

因此下面三种路径必须在数值上等价，且测试就是这么断言的：

- 一次性 prefill 整个 prompt
- 分块 prefill（chunk ∈ {1, 2, 3, 5, 7}）
- prefill 后逐 token cached decode

唯一允许的差异是 FP32 归约顺序带来的噪声。真实 Qwen3-0.6B 上实测 rel_l2 在
1e-5 量级以内（见 `tests/test_integration_real_model.py` 的
`INTERNAL_REL_L2_TOLERANCE`）。

分块不是性能优化，是**内存上界**：一次 prefill 2048 token 的 attention 分数矩阵
是 `[16, 2048, 2048]` FP32 = 256 MiB，分块后峰值只与 chunk 大小有关。

---

## 6. 计时区间定义

所有时长来自 `time.perf_counter()`，只覆盖模型工作，**不包含下载、转换、也不
包含调用方在 yield 之间打印所花的时间**。

| 名称 | 包含 | 不包含 |
| --- | --- | --- |
| `load_seconds` | 读 `config.json` / `manifest.json`、mmap 权重、编译 chat template | 下载、转换；也**不含把权重 page-in** ——mmap 是惰性的，第一个前向才真正读盘 |
| `tokenize_seconds` | 渲染 chat template + BPE 编码 | — |
| `prefill_seconds` | 所有 prefill 前向（每个 chunk），含 KV 写入 | tokenize、采样 |
| `first_token_seconds` | `tokenize + prefill + 第一次采样`，从请求开始计 | 模型加载 |
| `decode_seconds` | 首 token 之后每一步的 forward + sample | 消费者在两次 yield 之间做的事 |
| `total_seconds` | `tokenize + 从请求开始到结束的墙钟时间` | 模型加载 |

token 计数：

- `prompt_tokens` 是**渲染后 chat template 的 token 数**，不是用户原文的字数。
  prefill 吞吐必须按模型真正吃进去的量算。
- `decode_tokens = max(generated_tokens - 1, 0)`，即去掉第一个 token。
  `decode_tokens_per_second` 因此是稳态速率，不被 prefill 拉低。
- `prefill_chunks` 是实际执行的前向次数。

`tools/benchmark.py` 的微观 prefill 探针把**两种模式分开报**，因为它们是两条不同的
代码路径：`full` 走一次覆盖整段的前向，`chunked` 走引擎真正跑的分块循环。两者的
"前向次数"都来自**实测的调用次数**（探针用包装器数），不是 `长度 // chunk_size`
推算出来的——推算出来的数字曾经被当成实测结果写进文档（v0.1 缺陷 #6），所以这里
只信调用计数。纯 prefill 计时区间里没有采样、没有 decode、没有打印。

线程设置同理要区分**请求值**与**测量值**：报告里 `[threads] requested` 是命令行
给的，`exported before numpy` 说明环境变量是否在 `import numpy` 之前设好，
`detected backend` 是 NumPy 自己报的 BLAS，而**有效线程数报 "not observable"**
——BLAS 不暴露这个量，不能拿请求值冒充。`--threads 4` 与 `--threads=4` 两种写法
等价，都由 `tests/test_benchmark.py` 用子进程守着。

内存三个数字分开报告，因为它们含义不同：

- `weight_logical_bytes`：权重数组的逻辑大小（tied 的 `lm_head` 只算一次）。
- `kv_allocated_bytes`：预分配的 cache。
- `peak_rss_bytes`：`getrusage(RUSAGE_SELF).ru_maxrss`，操作系统实际记账的峰值。
  macOS 返回字节，Linux 返回 KiB，`metrics.py` 做了归一。

**把 mmap 数组的 `nbytes` 相加冒充 RSS 是错的**：没被读过的页根本不常驻。这就是
为什么在 Apple M5 Pro 上实测：映射完权重后峰值 RSS 只有 184.6 MiB，
  跑完一次生成后是 2.6 GiB。

---

## 7. 权重转换与加载

`tools/convert_model.py` 是**唯一**接触 safetensors / BF16 的地方：

```
models/Qwen3-0.6B-hf/            models/Qwen3-0.6B-slowq/
  config.json                      config.json          (原样拷贝)
  generation_config.json           generation_config.json
  tokenizer.json                   tokenizer.json
  tokenizer_config.json            tokenizer_config.json
  vocab.json / merges.txt          vocab.json / merges.txt
  model.safetensors      ────>     manifest.json
                                   tensors/
                                     model.embed_tokens.weight.npy
                                     model.layers.0.self_attn.q_proj.weight.npy
                                     ...                  (每个张量一个文件)
```

BF16 → FP32 是一次位移，不是算术运算：

```python
uint16 → uint32 → << 16 → view(float32)
```

所以转换必须**逐 bit 精确**，`TOLERANCES["max_abs_weights"] = 0.0` 就是这个意思。

`manifest.json` 记录 `format_version`、`compute_dtype`、来源 repo/revision/文件
sha256、每个张量的 `name/file/shape/dtype/source_dtype`，以及 `aliases`
（`lm_head.weight → model.embed_tokens.weight`，tied embedding 只存一份）。

写入侧的安全约束（都是踩过坑之后加的）：

- **默认拒绝非空 `--out`，但接受"不存在"和"存在且为空"。** 转换是"先删后写"式的
  破坏性操作，把一次失败的转换写进已有模型目录，结果是**旧 manifest 仍然有效、但
  `tensors/` 里新旧混杂**——加载时不会报错，只会算出错的答案。要覆盖必须显式给
  `--overwrite`。保护的是**内容**，不是路径本身：一个空目录里没有内容可保护。
- **前后两次检查必须由同一个动作完成。** `check_output_is_installable()` 在转换前
  跑，`install()` 在转换后跑；早先两者用的是不同判据（前者 `not entries`，后者
  `output.exists()`），于是**预先建好的空 `--out` 能通过前检查、却在几十秒转换之后
  被后检查拒绝**，用户看到的是一句"转换过程中出现了东西"。现在后检查不再自己判断，
  直接让 `staging.replace(output)` 这个原子 rename 去判：目录为空则成功，非空则
  `OSError`（macOS/Linux 上是 `ENOTEMPTY`），报错时列出实际条目并说明"里面一个字节
  都没动"。判据只有一份，就不会漂移（发布收尾 #2）。
- **给了 `--overwrite` 也不是就地写。** 整个转换 + 自检在目标旁边的暂存目录里
  完成，全部通过之后才 `os.replace` 换过去；中途任何一步失败都只留下暂存目录，
  旧模型一个字节都没动。逐 `.npy` 原子写**不够**：单个文件原子，整体不原子。
- **tied 权重要读它自己的分片。** `lm_head.weight` 与 `model.embed_tokens.weight`
  逐 bit 相同的自检，必须分别从 `model.safetensors.index.json` 指出的**各自那个
  分片文件**取字节，不能顺手复用当前正在转换的分片（v0.1 缺陷 #8）。
- **下载缓存按 repo + 解析后的完整 commit 分目录**，不按文件名。否则同一文件名
  在不同 revision 下会互相覆盖，而 manifest 里记的 sha256 是"实际用到的那份"——
  缓存命中错了文件，记录就变成假的（v0.1 缺陷 #4）。
- **中断的下载要能靠重跑同一条命令续上，而"完成"必须是真完成。** 快照标记
  `.slowq-snapshot.json` 过去只在**全部文件下完之后**才写，所以首次下载中途失败
  会留下一个没有标记的目录；重跑时 `verify_snapshot()` 先拒绝它（无法证明来源），
  下载器根本没机会执行——缓存目录变成死路，而 SlowQ 又刻意不删别人的目录。
  现在标记在**下载第一个文件之前**就写，并带 `complete: false`；每个文件下完后再
  按 revision 的 listing（先比大小，再比 LFS sha256）复核一遍，全部通过才改写成
  `complete: true` 并记 `completed_at`。于是：短文件/半截文件不会被当成缓存命中，
  永远匹配不上的快照永远不会被标记完成，而没有标记的目录**仍然拒绝**（只是错误
  信息给出三条安全出路：删掉那一个目录后重跑、换 `--download-dir`、用 `--source`），
  既不自动删除也不清空缓存。旧标记没有 `complete` 字段时按"完成"处理——它本来就
  只在成功后才写，这样已有缓存不会被重新下载（发布收尾 #1）。
- **截断的分片要报出自己的错误。** safetensors 头里的字节区间可能超过文件实际长度
  （下载中断）。`memoryview` 切片会像 `bytes` 一样**静默夹断**，短缓冲区一路传到
  NumPy 才炸，而回溯指向切片那一帧，于是 `mmap.close()` 抛 `BufferError` 把真因
  盖掉。`tensor_view()` 先按分片实际长度校验区间，报错里给出张量名、字节区间、
  分片名和文件真实长度（v0.1 缺陷 #10）。

加载侧的安全约束：

- `np.load(..., mmap_mode="r", allow_pickle=False)`。**`allow_pickle=False` 是
  显式的**：`.npy` 可以携带 pickle 载荷，允许它等于允许任意代码执行。
  `tests/test_weights.py` 有一个专门构造 pickle `.npy` 的测试，断言加载被拒绝。
- 数组是只读的 `np.memmap`，运行时无法就地改写权重。
- `check_complete(config)` 在构建模型时校验清单齐全、shape 与磁盘一致、dtype
  是 FP32。缺失或形状错误在**加载时**报错，不会等到前向算到那一层。

---

## 8. Tokenizer 与增量解码

- Jinja2 用 `ImmutableSandboxedEnvironment(trim_blocks=True,
  lstrip_blocks=True)`，并注入 `tojson` 与 `raise_exception`——官方 Qwen3
  template 会用到这两个。
- `enable_thinking=False`（默认）时，assistant 轮末尾会插入空的
  `<think>\n\n</tool_response>` 标记，与官方行为一致。
- 停止 token 集合从 `generation_config.json` 的 `eos_token_id` 读取；Qwen3
  发布**两个**（`<|fim_middle|>` 与 `<|fim_prefix|>`），两个都必须终止生成。

byte-level BPE 的一个字符可能被切成多个 token（`🐢` 在微型 tokenizer 下是 3 个
token），所以流式输出必须用 `IncrementalTextDecoder`：

- 累积原始字节，只在能解出完整字符时输出；
- 解不出的尾巴**扣住不发**，等后续 token 补齐；
- 扣住的 token 数超过 `MAX_HELD_TOKENS = 8` 就强制 flush，避免真遇到坏数据时
  无限缓冲。

朴素地"每个 token 单独 decode 再拼接"会在半个字符上产出 U+FFFD。
`tests/test_tokenizer.py::test_naive_per_token_decoding_would_have_been_wrong`
把这个错误做法实现了一遍并断言它是错的。

**扣住不发就意味着必须有人负责最后吐出来。** 引擎在停止的那一步调用
`decoder.flush()`，并把结果**同时**放进最后一个 `text_delta` 和
`GenerationResult.text`，于是流式消费者拼出来的字符串与阻塞式调用返回的
完全一致：

```
"".join(step.text_delta for step in engine.stream(...)) == engine.generate(...).text
```

EOS 和 `max_new_tokens` 走的是同一个分支，所以两条路径都 flush 过——
v0.1 的缺陷 #3 正是只在其中一条上漏掉，最后一个 token 的尾巴字符（中文和 emoji
最容易触发）就只出现在 `result.text` 里，流式那边凭空少字。
`tests/test_engine.py` 对两条路径都断言了上面那个等式。

---

## 9. 采样

过滤顺序与 `transformers` 的 logits warper 流水线一致：
**temperature → top-k → top-p**。

- `temperature == 0` 表示 greedy（argmax），并**跳过** top-k/top-p，这样 greedy
  模式下这两个参数不会悄悄起作用。
- top-p 保留"累积概率刚好跨过阈值"的那个 token：
  `remove[1:] = remove[:-1]`。少这一步会把 nucleus 少留一个候选。
- argmax 平局取 id 最小的，与 `np.argmax` 一致，保证可复现。
- RNG 是私有的 `np.random.Generator(PCG64(SeedSequence(seed)))`，**绝不碰全局
  `np.random`**，因此固定 seed 可复现，且库的其他部分无法移动它。
- 非法参数、空候选集、非有限 logits 一律抛 `SamplingError`，不会退化成随便挑一个。

---

## 10. 依赖边界

运行时只依赖三个包：

```
numpy >= 1.26
tokenizers >= 0.20
Jinja2 >= 3.1
```

`torch` / `transformers` 只出现在 `pip install "slowq[ref]"` 里，且只被
`tools/validate_reference.py` 的 `ref` 子命令使用（在函数体内惰性 import）。
`tools/convert_model.py` **一个都不需要**：它自己解析 safetensors 容器
（8 字节小端头长度 + JSON 头 + `mmap` 数据段）、自己做 BF16→FP32 位移，
不依赖 `safetensors` 也不依赖 `torch`。

这条边界由 `tests/test_dependency_boundary.py` 自动看守，两种方式：

1. 子进程 import 全部运行时模块，检查 `sys.modules` 里有没有 torch 等；
2. AST 扫描 `src/slowq/**/*.py`，任何 `import torch` / `from transformers ...`
   （包括函数体内的惰性 import）都会失败。

第二种是必要的：第一种只能发现被测试实际调用到的路径。

**一个必须说清楚的例外：** `tokenizers` 自己声明了
`huggingface-hub>=0.16.4` 作为硬依赖（0.21.4 与 0.23.3 都是），所以
`pip install slowq` 会把 `huggingface_hub` 连同它的 HTTP 栈一起装进来（在这个
环境里发行版名叫 `httpx2` / `httpcore2`，`import httpx` 反而不存在）。这不在
SlowQ 的控制范围内。但**装进来不等于被用到**：在一个干净的 venv 里 import
全部 12 个运行时模块后，`sys.modules` 共 264 个模块，其中第三方顶层包只有
`numpy` / `tokenizers` / `jinja2` / `markupsafe`——`torch` / `transformers` /
`safetensors` / `huggingface_hub` / `httpx2` 一个都没有。
依赖边界约束的是 import 图和网络行为，不是 pip 的传递闭包。

转换后的推理**完全离线**，不会自动下载任何东西。
`tests/test_integration_real_model.py::test_loading_never_reaches_the_network`
把 `socket.create_connection` 和 `socket.socket.connect` 换成会抛异常的桩，然后
加载真实模型。

---

## 11. 测试分层

```
默认（快速）   python -m pytest -m "not integration"   480 个测试，本轮实测 4.28 秒
              使用 tests/conftest.py 现场构造的微型 Qwen3 目录（约 40 KiB）
              和一个真正训练过的 byte-level BPE tokenizer，不下载任何权重
              全程离线，不需要任何真实 checkpoint

显式启用      SLOWQ_MODEL_DIR=models/Qwen3-0.6B-slowq \
                  python -m pytest -m integration
              23 个测试，真实 Qwen3-0.6B；需要已转换的 checkpoint
              本轮实测 17.20 s（macOS arm64）；上一轮同一套件实测 18.4 s，
              Linux 容器里 25.8 s——随页缓存冷热在 14–26 s 之间
              参考对齐部分需要事先用 tools/validate_reference.py ref
              抓取过 artifacts/validation/ 下的 ref.npz 与 reference.json
              （pytest 进程内不加载 torch）

合计          503 个测试
```

上面的 480 / 23 / 503 是**本轮（2026-10-19，发布收尾）在 macOS arm64 上实测**的；
上一轮（v0.1 修复轮）结束时同一位置写的是 462 / 23 / 485，差值正是本轮新增的
18 个回归测试（`test_convert_model.py` +10、`test_validate_reference.py` +8）。
**本轮没有连接远程 Linux**，所以 §14 里那份 Linux 容器结果仍然属于上一轮的测量批次，
它跑的是 462 + 23 的套件版本，不是现在这 503 个。

快速套件里有三个文件是**工具层**的回归测试，不是模型层的：
`test_convert_model.py`（27 个）、`test_validate_reference.py`（46 个）、
`test_benchmark.py`（27 个）。它们守的不是"算得对不对"，而是"工具在失败时
会不会说谎、会不会破坏数据"——转换失败留下混杂权重、比较器对 NaN 或缺失张量
判 PASS、基准报告没有真正执行的分块，这三类问题的共同点是**结果看起来正常**。
新增的回归测试都逐个验证过：撤掉对应的修复，它们会失败。

微型 fixture 刻意保留了 Qwen3 的别扭之处，而不是简化掉：

- `head_dim * num_attention_heads != hidden_size`（8 × 4 = 32 ≠ 16），任何从
  `hidden_size` 推导 `head_dim` 的代码在这里都会失败；
- `num_attention_heads / num_key_value_heads == 2`，与 Qwen3-0.6B 相同的 GQA
  分组；
- `tie_word_embeddings = true`，`lm_head.weight` 只作为别名存在；
- norm 权重是 `N(1, 0.02)` 而不是全 1，忽略 norm 权重的 bug 藏不住。

---

## 12. 参考对齐验证

`tools/validate_reference.py` 让两套实现在**独立进程**中运行，通过结果文件交换
数据，这样 2.4 GiB 的 torch FP32 模型和 NumPy 模型不会同时常驻：

```bash
python tools/validate_reference.py numpy   --model models/Qwen3-0.6B-slowq
python tools/validate_reference.py ref     --hf-dir models/Qwen3-0.6B-hf
python tools/validate_reference.py compare
```

参考端设置是**刻意固定**的：`torch_dtype=float32`、`eval()`、
`attn_implementation="eager"`（不用 SDPA/flash kernel）、CPU、batch size 1。

用**完全相同的 token id** 比较：

1. tokenizer：渲染文本与 token id vs `AutoTokenizer`
2. weights：抽样的 6 个张量必须与官方 BF16 checkpoint 的 FP32 上采逐 bit 相同
3. provenance：参考端实际读的 `model.safetensors` 的 sha256，必须与 SlowQ
   `manifest.json` 声称的来源 sha256 相同——否则两边比的根本不是同一份权重
4. full prefill：最后一个位置的 logits + 选定层的 hidden states
5. chunked prefill：同上，但由多次经过 cache 的前向产生
6. cached decode：`--decode-steps` 步强制 decode 后每一步的 logits
7. argmax：每个被比较位置上的 greedy next token 必须一致

捕获层为 `(0, 13, 27)`——首、中、末，既能抓到从第 0 层就存在的错误，也能抓到
只有累积之后才显现的错误。

第 2 项抽样的是**张量名**，不是张量内的元素。存进交换文件的不是张量本身，而是
一份**规范化摘要**（`DIGEST_SPEC_VERSION = "slowq-weight-digest-v1"`）：
sha256 覆盖 spec 字符串 + dtype（`<f4`）+ shape + 按 C 顺序遍历的**全部字节**。
摘要规范本身写进结果文件，所以"逐 bit 相同"这个说法是可核查的——它明确声明了
dtype、shape、字节序和字节表示，且**覆盖整份张量，不做步长抽样、不截断**。

这么改的直接收益：`slowq.npz` 从压缩后约 300 MiB（未压缩 629 MiB，其中
151936×1024 的 embedding 表 FP32 就占 593 MiB）降到约 11 MiB，与只装 logits 和
hidden states 的 `ref.npz` 持平。门限的覆盖面没有变小：摘要覆盖每一个元素，
任何一个 bit 不同都会导致 sha256 不同。集成测试每轮把它们拷进 `tmp_path` 再重算
（而不是覆盖已签收的 `artifacts/`），临时磁盘开销也从 300 MiB 降到 11 MiB。

容差在**测量之前**声明（`TOLERANCES`）：

```
rel_l2_logits   1e-3      max_abs_logits   5e-2
rel_l2_hidden   1e-3      max_abs_hidden   5e-2
rel_l2_weights  0.0       max_abs_weights  0.0
```

理由：FP32 单位舍入是 6e-8，一个 28 层、宽度 1024–3072 的残差网络加上 softmax
会把它放大约 1e-5 相对误差，而且两边用的是不同 BLAS（Accelerate vs torch）与
不同求和顺序。声明值取该估计的约 10 倍。

不要求跨平台逐 bit 相同，也不要求采样生成的文本与 Transformers 完全一致。
**v0.1 之后的两轮修复都没有改动其中任何一个数字。**

### compare 不给假 PASS

比较器的默认倾向必须是"证据不足就失败"，否则它会安静地放行。`compare` 因此在
比数之前先做一轮资格审查：

- **应有的 key 集合是从抓取配置推导出来的**，不是"两个文件里恰好共有的 key"。
  `expected_numeric_keys()` 按 `样例数 × {full, chunked, decode×N} ×
  {logits, 各捕获层 hidden, final_norm} × 各自的位置` 展开，当前配置下是
  **116 个**。少一个报错，多一个也报错。
- **两侧都缺也算失败。** 只看交集的话，一次抓取出错会让两边同时少掉同一批 key，
  比较照样 PASS。
- **0 个比较直接失败。** "没东西可比"不是通过。
- **元数据的应有集合同样从协议推导，不从文件推导。** 数值张量由
  `expected_numeric_keys()` 兜住，但 `check_tokenizer()` / `check_weights()` 过去
  遍历的是**两个文件各自记了什么**的并集：同一条记录在两侧同时缺失时，它根本不
  进入循环，于是"4 个样例的 tokenizer 都对上了"可以在只比了 3 个的情况下打印出来。
  现在应有的样例名来自抓取配置（`capture["sample_names"]`，缺失时退回 `SAMPLES`），
  应有的权重名来自 `WEIGHT_CHECK_SAMPLE`，两侧同时缺失、单侧缺失都失败，错误信息
  指名是哪个样例、哪个权重（发布收尾 #3）。
- **缺元数据即要求重抓，而不是跳过。** 没有 `reference.json`、没有架构指纹、
  没有摘要规范，都判失败。
- **先查这份参考是不是在描述这一轮**：artifact 格式版本、样例名、捕获层、
  捕获位置、chunk 大小、decode 步数、架构指纹、摘要规范，任一项对不上就要求
  重新抓取。这是"不能沿用历史结论"在工具层的落点。
- **展平之前先比 shape**，比之前先查两侧是否有限——`nan > 容差` 恒为 False，
  旧版比较器正是因此把 NaN 判成了通过。参考范数为 0 的情况单独处理，不做除法。
- 失败时非零退出，并**指名是哪个样例的哪个张量**。

`tests/test_validate_reference.py`（46 个）逐条守着上面每一行，其中包括一个真的
`subprocess.run` 起 CLI、断言两侧同时少一条记录时**退出码非零**的测试。

### 实测结果

参考端环境：`transformers 4.51.3`、`torch 2.14.1`、`attn_implementation=eager`、
`dtype=float32`、`threads=4`，在 macOS arm64（Apple M5 Pro）上抓取。

**`PASS: 116 tensors compared within the declared tolerances.`**

| 检查项 | 结果 |
| --- | --- |
| tokenizer：渲染文本 + token id vs `AutoTokenizer` | 4/4 样例逐字符、逐 id 相同（18 / 34 / 66 / 30 tokens） |
| weights：抽样 6 个张量 vs 官方 BF16 的 FP32 上采 | 6/6 **sha256 相同**（整份张量），`max_abs = 0.0` |
| provenance：参考端读的 `model.safetensors` vs manifest 声称的 | sha256 `f47f71177f32bcd1…` 两侧相同 |
| full prefill logits + 第 0/13/27 层 hidden states | 全部通过 |
| chunked prefill（chunk = 8） | 全部通过 |
| 3 步 cached decode | 全部通过 |
| greedy argmax | 20/20 个比较位置全部一致 |

最差的一项是 `zh-single-turn.full_prefill.logits`：rel_l2 **7.986e-06**，
max_abs **1.063e-04**——比声明的门限紧约 100 倍。hidden states 更紧。
这两个数字在 v0.1 交付时、上一轮修复后、以及本轮发布收尾后**三次测量完全相同**
（本轮于 2026-10-19 在本机重跑 `compare`，退出码 0）：两轮改的都是比较器的资格
审查、权重的存储形式和转换工具的写入/下载纪律，没有碰任何数值路径。

四个样例的 greedy 首 token（SlowQ 与参考完全一致）：

| 样例 | prompt tokens | full prefill | chunked | 3 步 decode |
| --- | --- | --- | --- | --- |
| zh-single-turn | 18 | 104198 | 104198 | 15469, 110498, 3837 |
| en-with-system | 34 | 785 | 785 | 12884, 7952, 6303 |
| multi-turn | 66 | 24 | 24 | 21, 488, 220 |
| emoji-and-mixed-script | 30 | 144841 | 144841 | 220, 11622, 46944 |

`artifacts/validation/` 里的 `ref.npz` / `reference.json` / `slowq.npz` /
`samples.json` 就是这次验证的产物，各约 11 MiB。
`tests/test_integration_real_model.py` 会**用当前代码重跑 NumPy 侧**——重新渲染
tokenizer、重新算权重摘要、重新前向——再与已抓取的参考对比，所以运行时代码或
tokenizer 一旦偏离签收时的行为，集成测试就会失败，而 pytest 进程内不加载 torch。
其中 6 个测试专门守着"不能沿用历史结论"：把历史 PASS 原样抄进结果文件不算通过。

**格式变了就要重抓。** 旧的 artifact 里没有 `reference.json`、没有架构指纹、
权重是整份张量而不是摘要，`compare` 会要求重新抓取而不是自动放行。摘要格式是
**上一轮**改的，当时已经用修复后的比较器在 macOS 上重新抓取并复核过参考端
（`ref` 与 `numpy` 是两个独立进程）。

**本轮（发布收尾）没有重新抓取参考端**，也没有改写 `artifacts/validation/`：
只重跑了集成测试（它把已签收的四个文件拷进 `tmp_path`，用当前代码重算 NumPy 侧
再比，23 passed）和一次只读的 `compare`（退出码 0，`PASS: 116`），跑前跑后对
`artifacts/validation/` 四个文件取 sha256，**逐个相同**。也就是说：现有的参考数据
在本轮改动之后依然被判定为对齐，不需要重抓；`tools/validate_reference.py` 本轮只
改了元数据完整性检查，没有改抓取格式，`artifact_format` 仍是 2。

---

## 13. 干净运行环境验证

规范要求在**没有安装 torch 和 transformers** 的环境里用转换后的真实权重完成
一次生成。实际执行过程与结果：

```bash
python3 -m venv /tmp/slowq-clean
/tmp/slowq-clean/bin/pip install "/path/to/slowq"     # 只装运行时依赖
/tmp/slowq-clean/bin/python -c "import torch"          # ModuleNotFoundError ✓

/tmp/slowq-clean/bin/slowq generate \
    --model models/Qwen3-0.6B-slowq \
    --prompt "用一句话解释什么是 KV cache。" \
    --max-new-tokens 64 --max-context 512
```

输出（stdout）：

```
KV cache 是一种用于缓存键值对（Key-Value Pair）的内存结构，用于减少重复访问和提高数据访问效率。
```

统计（stderr），退出码 `0`：

```
loaded in 0.10 s (weights 2.2 GiB logical, KV 112.0 MiB allocated for 512 tokens)
stop reason              eos
prompt tokens            19 (1 chunk(s))
generated tokens         31
prefill                  397.5 ms
first token latency      398.2 ms
decode speed             46.64 tok/s
peak RSS                 2.4 GiB
```

该 venv 里 `pip install .` 装出来的包：`numpy 2.5.4`、`tokenizers 0.23.3`、
`Jinja2 3.1.6`、`slowq 0.1.0`，加上 `huggingface_hub 2.2.0` / `httpx2` /
`httpcore2` / `tqdm` 等 tokenizers 的传递依赖（见 §10 的说明）。
**没有 torch，没有 transformers，没有 safetensors。** 另外为了在同一个 venv 里
跑测试装了 `pytest 9.1.1`，它不是 slowq 的依赖。

快速测试套件在同一个干净 venv 里也全部通过：`372 passed, 17 skipped in 1.34s`
（skipped 的是需要真实模型的集成测试）。

> **本节是 v0.1 修复前在 macOS 上的记录，原样保留（含当时的测试数）。**
> 修复之后重跑同一条 prompt，输出文本的 sha256 是
> `23c4791e233b1262a0ca53f95a75d99db0b34bbd7910ced5c7db25e61ff18319`，与当时
> 逐字相同；那次是冷启动，prefill 7228.8 ms、decode 42.82 tok/s。当时的测试数是
> 372 / 17，v0.1 修复轮之后是 462 / 23，本轮（发布收尾）之后是 §11 里的 480 / 23。

> 注意 `prefill 397.5 ms` 对 19 个 token 而言异常慢——这是冷启动，权重页
> 第一次被读进内存。同一条 prompt 又跑了一次（`--max-context 1024`），输出
> 逐字相同、退出码同样是 `0`，但 prefill 变成 288.8 ms、decode 42.94 tok/s。
> 冷启动的 prefill 每次都不同；decode 在 43–47 tok/s 之间波动，与
> `tools/benchmark.py` 在开发 venv 里测到的 44–47 tok/s 一致。

---

## 14. 跨平台验证

第二平台：Alibaba Cloud Linux 3、内核 5.10.134、2× Intel Xeon Platinum 8575C
（2 socket × 48 core × 2 thread = 192 逻辑核，AVX-512 + AMX）、2 TiB 内存。
**全部操作在 docker 验证容器内完成，宿主机不做任何改动**（复核后确认
宿主机 home 目录下没有任何 slowq 相关文件）。源码用 `COPYFILE_DISABLE=1 tar`
流式送入容器，不落宿主机磁盘。

容器里有**两个环境，必须分开说**，因为它们回答的是不同问题：

- `/opt/miniconda3`：Python 3.14.8，numpy 2.5.4（**scipy-openblas**，与 macOS 的
  Accelerate 不同），tokenizers 0.21.4，jinja2 3.1.6，pytest 9.1.1。
  **这不是干净环境**——早先为了尝试构建 torch 参考端，往里装过 `transformers` /
  `safetensors` / `huggingface_hub`（torch 最终没装上）。本文早先版本把这里写成
  "容器内本就无系统级 torch"，那是**不准确的**，特此更正。它用来跑测试套件和
  转换/基准。
- `/work/clean-venv`：现场 `pip install .` 建的，只有 numpy 2.5.4、
  tokenizers 0.23.3、Jinja2 3.1.6、MarkupSafe 3.0.4、slowq 0.1.0，以及
  tokenizers 自己声明的 `huggingface_hub 2.2.0`。`import torch` /
  `import transformers` / `import safetensors` 三个都是 `ModuleNotFoundError`。
  §13 那种"干净环境端到端生成"必须在这个环境里做。

**上一轮（v0.1 修复轮，2026-10-18）在容器里重跑的结果**——原样保留。本轮发布收尾
**没有连接这台机器**，所以下表的测试数（462 / 23 / 485）对应的是上一轮的套件版本，
不是 §11 里现在的 480 / 23 / 503；本机 macOS 侧的本轮实测见 §11 与 §12。

| 项目 | 结果 |
| --- | --- |
| 快速套件 | **462 passed, 23 skipped**（miniconda 3.63 s；干净 venv 3.60 s） |
| 集成套件 | **23 passed**，19.40 s（干净 venv）；两个环境把全部测试一并选中时都是 **485 passed**（干净 venv 23.06 s / miniconda 22.35 s） |
| NumPy 侧 vs **已固定的** macOS 参考 | **PASS: 116 tensors**；最差 rel_l2 **1.011e-05**（`multi-turn.decode0.logits`）、max_abs 1.159e-04；20/20 argmax 一致 |
| 权重摘要 / provenance | 6/6 与官方 BF16 的 FP32 上采 sha256 相同；`model.safetensors` = `f47f71177f32bcd1…`，与参考端读到的同一份 |
| 权重转换（暂存式转换器） | 310 个张量，6.3 s；310/310 个 `.npy` 与 macOS arm64 的转换结果 **sha256 逐个相同** |
| 覆盖已有模型的守卫 | 非空 `--out` 且未给 `--overwrite` 时拒绝，退出码 1，目标目录一个字节都没动 |
| 截断的 checkpoint | 1.5 GB 的 `model.safetensors` 截到 1.2 GB：在第 202/310 个张量处报出自己的错误（张量名、字节区间、分片名、文件实际长度），退出码 1；已有模型 manifest sha256 不变、310 个张量齐全、仍能生成；无暂存目录残留 |
| 端到端生成（干净 venv） | 输出与 macOS **sha256 逐字节相同**（`23c4791e…`，两个平台各自重跑过一次，仍然相同）；19 prompt / 31 generated、eos、退出码 0 |
| 依赖边界 | 干净 venv 里 `test_dependency_boundary.py` 4 passed |
| `--threads=4`（等号写法） | requested 4、exported before numpy **True**、5 个环境变量全部为 4、后端单独列为 scipy-openblas 0.3.34.106.0、有效线程数报 "not observable" |

测试数与耗时来自**同一次运行**：那一轮把当时的最新源码重新送进容器，干净 venv 里的
`slowq` 包也重新同步过并核对了 13 个 `.py` 的 sha256。原始输出在
`artifacts/linux-recheck/`。

有一处**必须说清楚的复现性细节**：用同一份源码在 Linux 上重跑第二次，结论仍是
`PASS: 116`，但两次输出有一处末位数字不同（`multi-turn.decode2.logits` 的
rel_l2 1.809e-06 vs 1.808e-06，见 `linux-compare.txt` 与 `linux-compare-rerun.txt`）。
原因是 OpenBLAS 在 192 线程下的归约顺序不是逐次确定的。量级远在门限之内、
greedy argmax 也没有变，但**不要把 Linux 上的 rel_l2 当成逐 bit 可复现的常数**。

### 这一行到底证明了什么

Linux 上做的是"**只重算 NumPy 侧**，与 macOS 上抓取并固定下来的 PyTorch 参考张量
对比"。它证明的是 **x86_64 + scipy-openblas 的 NumPy 实现与那份参考一致**；
它**不等于**"在 Linux 上重新执行了 PyTorch 参考实现"——后者没有做。容器里
`import torch` 依然是 `ModuleNotFoundError`。这两件事不能混为一谈。

即便如此，它补上了 §12 之前最大的证据缺口：**"换一种 BLAS 之后 rel_l2 落在哪"
现在有实测值了**，是 1.011e-05，比 macOS 的 7.986e-06 大约 1.3 倍，仍然比声明的
1e-3 门限紧约 100 倍。之前这个数只能靠推测。

转换逐字节相同是设计使然：BF16→FP32 是 `uint32(bits) << 16` 的纯位操作，不经过
任何浮点运算，因此与 BLAS、与指令集都无关。

性能（`tools/benchmark.py`，`--repeat 2`，原始数据在 `artifacts/linux-recheck/`）：

| 设置 | decode | 512-token full prefill | 512-token chunked@128（实测 4 次前向） |
| --- | --- | --- | --- |
| 不设线程（BLAS 默认） | **49.69 tok/s**（20.12 ms/token） | 3322.7 ms | 2130.4 ms |
| `--threads 4` | 15.50 tok/s（64.53 ms/token） | 2774.1 ms | 2487.5 ms |

两行的差别就是缺陷 #9 的意义：线程数被真正设进去了，所以测出来的东西变了。
旧代码里 `--threads=4` 被 argparse 接受但从未导出，报告却照样打印线程信息。

> **跨平台的性能对比不成立，已删除。** 本文早先版本写过"Linux decode 50.51 tok/s
> 高于 macOS，方向与内存带宽 roofline 推算一致"。两台机器的线程设置、BLAS、独占
> 程度都不同（这台是共享机，测量期间 `/proc/loadavg` 从 2.07 升到 16.13），那个
> 对比支撑不了任何结论。macOS 与 Linux 之间**能**成立的对比只有正确性相关的：
> 转换结果逐字节相同、生成文本逐字节相同、与同一份参考的 rel_l2 同量级。

**没有完成的部分，如实记录：** 在 x86_64 上重新执行 PyTorch 参考端**没有做**。
该网络环境拿不到可用的 torch 构建——`registry-1.docker.io` 与 ACR 的 `library`
命名空间均不可达，aliyun 的 pytorch wheel 镜像没有 cp314 的包，conda 的 pytorch
通道装出的 torch 又因缺 `iJIT_NotifyEvent`（Intel ITT 符号）无法 import。所以
"torch 自己在另一种 BLAS / 指令集下会不会算出别的数"仍未验证。这一点在 README
已知限制 #10 中同样记录。

复现 Linux 侧只重算 NumPy 的那一步（不需要 torch）：

```bash
# 先将 SLOWQ_TEST_HOST / SLOWQ_TEST_CONTAINER 设为自己的 SSH 主机和容器名
# 把 macOS 上抓取并固定的两个文件送进容器
COPYFILE_DISABLE=1 tar -cf - -C artifacts/validation ref.npz reference.json | \
  ssh "$SLOWQ_TEST_HOST" "docker exec -i $SLOWQ_TEST_CONTAINER tar -xf - -C /work/slowq/artifacts/validation"
# 容器内
python tools/validate_reference.py numpy --model /work/models/Qwen3-0.6B-slowq \
    --artifacts /work/validation-linux --max-context 512 \
    --prefill-chunk-size 8 --decode-steps 3
python tools/validate_reference.py compare --artifacts /work/validation-linux
```

三个环境陷阱，留给复现的人：

* macOS 的 `tar` 默认写入 AppleDouble 元数据（`._ops.py`），它们以 `.py` 结尾，
  会被 `test_dependency_boundary.py` 的 AST 扫描当源码读并报 UnicodeDecodeError。
  传源码必须 `COPYFILE_DISABLE=1`。
* 该镜像的 conda（26.5.3）默认 libmamba solver 与镜像内老 libstdc++ 不兼容
  （`GLIBCXX_3.4.29 not found`），需要 `CONDA_SOLVER=classic`。
* 想验证"截断的分片报出自己的错误"，截断长度要选在**让 tied 权重自检也能通过**
  的位置（`lm_head.weight` 与 `model.embed_tokens.weight` 都要还在文件里），
  否则先撞上的是 §7 那条跨分片守卫，看不到本次要测的失败路径。
