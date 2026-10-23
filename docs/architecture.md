# 慢炖 · SlowQ 架构说明

本文描述 SlowQ v0.2 的实际实现：模块边界、两条正交的抽象轴（架构家族与任务类型）、
一次请求的完整数据流、关键张量 shape、KV cache 语义、三个任务各自的前向路径、
计时区间定义，以及依赖边界。所有内容对应仓库中真实存在的代码，不是设计意图。

[返回 README](../README.md) · [模型支持](models.md) ·
[测试与数值验证](validation.md) · [性能与内存](benchmarks.md) ·
[变更记录](changelog.md) · [v0.2 接入报告](v0.2-多模型接入报告.md)

文中标注为**历史批次**的测量结果属于当时的执行环境，不代表本轮重新跑过；
本轮（2026-10-20 ~ 10-21）实际执行的数字都在[测试与数值验证](validation.md)、
[性能与内存](benchmarks.md)与[模型支持](models.md)里。

---

## 1. 两条轴：family 与 task

v0.1 只有一个模型，所以"模型是什么样"和"模型拿来做什么"是同一个问题的两个说法。
v0.2 支持 8 个 checkpoint、3 类任务，这两件事必须拆开，否则每加一个模型就要在
前向代码里多一个 `if`：

| 轴 | 回答什么 | 从哪里读 | 决定什么 |
| --- | --- | --- | --- |
| `family` | 这些权重按什么规则计算 | `config.json` 的 `model_type` | 前向结构（Q/K norm、哪些投影带 bias）、权重清单、`head_dim` 能否推导 |
| `task` | 这份目录拿来做什么 | 转换后目录的 `manifest.json` | 用哪个引擎、要不要 `lm_head`、比较器抓哪些张量 |

`family` 有四个取值：`qwen3`、`qwen2`、`llama`、`idefics3`。前三个是纯文本 decoder，
规则集中在 `config.py` 的 `TEXT_FAMILIES`；`idefics3` 的 config 是嵌套结构
（`text_config` + `vision_config`），所以它的加载入口在 `slowq.vlm`。

`task` 有**三个**取值：`text-generation`、`text-embedding`、`image-text-generation`。

**task 不能从 config 猜，这是本轮最重要的一条事实。** `Qwen/Qwen3-Embedding-0.6B`
公布的 `model_type` 是 `qwen3`、`architectures` 是 `["Qwen3ForCausalLM"]`，
这两个字段与聊天模型**取值完全一致**（整个 `config.json` 并不相同：向量模型的
`vocab_size` 是 151669、`eos_token_id` 是 151643、`max_position_embeddings` 是 32768，
聊天模型分别是 151936 / 151645 / 40960——没有一个字段的取值能用来判断任务）。
任何"看架构名字决定入口"的加载器都会把向量模型当成
生成模型建起来，然后在第一次前向时因为缺 `lm_head` 失败——或者更糟，
用一个不该存在的输出投影算出一个看起来正常的向量。所以 task 由转换器写进 manifest，
加载时读 manifest，`slowq.engine.recorded_task()` 是唯一入口。

反过来 `family` 也不能只由 task 决定：图文模型的文本塔是 llama，
`SmolVLMConfig.text.family == "llama"`，而整个 checkpoint 的 family 是 `idefics3`。
`ModelConfig.tensor_prefix` 就是为这种情况存在的——同一个 decoder 实现，
权重前缀从 `model` 换成 `model.text_model`。

```
                       config.json                     manifest.json
                    model_type: qwen3              task: text-embedding
                          │                        model_family: qwen3
                          ▼                                 │
                  family_for() ──> TEXT_FAMILIES            │
                          │                                 ▼
                          ▼                        Engine / EmbeddingEngine
                  ModelConfig(qkv_bias,            / VisionEngine 三选一
                   qk_norm, head_dim, ...)
```

---

## 2. 模块与依赖方向

依赖是单向的，任何反向引用都是 bug：

```
cli.py ──┬─> engine.py ──> model.py ──┬─> ops.py
         │        │                   └─> cache.py
         │        └──> tokenizer.py / sampling.py / metrics.py
         │
         ├─> embedding.py ──> model.py（用 DecoderModel，不用 CausalLMModel）
         │
         └─> vlm.py ──> model.py + engine.py + ops.py
                              （Pillow 只在这里、且只在真正读图时 import）

config.py <── weights.py；两者都不依赖上面任何一层
```

| 模块 | 职责 | 行数 |
| --- | --- | --- |
| `config.py` | 家族规则表 `TEXT_FAMILIES`、`ModelConfig`、`GenerationConfig`、支持范围检查 | 554 |
| `weights.py` | manifest 解析（v1/v2）、只读 mmap、按家族与任务推导的权重清单 | 413 |
| `ops.py` | 无状态算子：RMSNorm、LayerNorm、RoPE、softmax、SiLU、GELU-tanh、GQA attention、conv2d patchify、pixel shuffle、L2 归一化 | 291 |
| `cache.py` | 预分配 KV cache，`length` / `write` / `read` / `commit` / `reset` | 144 |
| `model.py` | 家族通用文本 decoder：`DecoderModel` / `CausalLMModel` / `DebugProbe`；加载时交叉校验 manifest 与 `config.json` 的家族 | 658 |
| `tokenizer.py` | Jinja2 chat template、BPE 编解码、增量解码器 | 370 |
| `sampling.py` | temperature / top-k / top-p，独立 RNG | 155 |
| `engine.py` | 唯一的生成循环：分块 prefill + 逐 token decode；`recorded_task()`；文本与图文共用的 `_validate_request()` | 469 |
| `embedding.py` | 向量任务：`EmbeddingEngine` / `EmbeddingModel` / last-token pooling / 余弦相似度 | 421 |
| `vlm.py` | 图文任务：`SmolVLMConfig`、图像预处理、视觉塔、connector、`VisionEngine` | 1403 |
| `metrics.py` | 计时、内存、吞吐统计 | 220 |
| `cli.py` | `generate` / `chat` / `embed` / `vlm` / `inspect` 五个子命令 | 782 |

另有 `errors.py`（47 行）、`__init__.py`（120 行）、`__main__.py`（10 行）。
合计 **15 个模块、6062 行**（发布前修复轮后是 6057；那一轮 `ops.py` -18、
`model.py` +20、`engine.py` +16、`vlm.py` +8，净 +26。验证工具修复轮只改了
`model.py` 的 docstring，+5 行，运行时代码一行未动）。

三条边界由测试守着，不是靠自觉：

- **算子不得依赖 CLI，`engine.py` 不得依赖 `cli.py`。** `tools/` 下的脚本可以依赖
  `src/slowq`，反之不行。
- **`embedding.py` 只依赖 `DecoderModel`**，不依赖 `CausalLMModel`：向量模型没有
  输出投影，`CausalLMModel` 会在构造时拒绝它。
- **`vlm.py` 是唯一可能 import Pillow 的模块**，且 import 在 `_pillow()` 函数体内。
  `import slowq.vlm` 本身不触发它，纯文本安装不会因此在导入期失败。

---

## 3. 文本 decoder：一份实现，三个家族

`model.py` 的 docstring 里那张表就是全部差异。逐条对照 transformers 4.51.3 的
`Qwen3DecoderLayer` / `Qwen2DecoderLayer` / `LlamaDecoderLayer` 之后，剩下的结构
**完全相同**：

```
residual = x
x = residual + Attention(input_layernorm(x))
residual = x
x = residual + MLP(post_attention_layernorm(x))
```

差异全部是**标志位**，不是分支结构：

| | qwen3 | qwen2 | llama |
| --- | --- | --- | --- |
| Q/K RMSNorm（逐 head） | 有 | 无 | 无 |
| q/k/v bias | `config.attention_bias` | **恒开** | `config.attention_bias` |
| o_proj bias | `config.attention_bias` | **恒关** | `config.attention_bias` |
| MLP bias | 恒关 | 恒关 | `config.mlp_bias` |
| `head_dim` | **必须显式给出** | `hidden // heads` | `hidden // heads` |
| 尊重 `config.sliding_window` | 是 | 是 | 否 |

三条必须解释清楚的规则：

- **qwen2 的 bias 规则不在 config 里。** 它们硬编码在 `Qwen2Attention.__init__`：
  q/k/v 一律 `bias=True`，`o_proj` 一律 `bias=False`。写成 `config:o_proj_bias`
  会让 Qwen2.5 多算一个不存在的 bias；写成 `config:attention_bias` 用于 q/k/v
  则会让它少算——两个 Qwen2.5 checkpoint 根本没有 `attention_bias` 这个字段，
  `.get()` 会返回 `False`。
- **qwen3 拒绝推导 `head_dim`。** transformers 三个家族都写
  `getattr(config, "head_dim", hidden_size // num_attention_heads)`，所以回退本身是
  官方的——但对 Qwen3 是错的：0.6B 的 `hidden_size=1024`、`num_attention_heads=16`，
  推导得 64，而 config 显式写着 `head_dim=128`。Q/K/V 投影的输出维度是
  `16 × 128 = 2048`，**比残差流宽一倍**，O 投影再压回 1024。推导出来的模型能加载、
  能跑、结果全错。所以 `head_dim_must_be_explicit=True`，缺失时报 `ConfigError`。
  qwen2 与 llama 的官方 checkpoint 确实不写这个字段，推导后才校验
  `hidden_size % num_attention_heads == 0`。
- **sliding window 按实际启用状态判断。** 两个 Qwen2.5-0.5B checkpoint 的 config 里
  都有 `sliding_window: 32768` 且 `use_sliding_window: false`。qwen2 / qwen3 的
  **模型级** causal mask 会应用这个窗口，与 `use_sliding_window` 无关，而 SlowQ 只
  实现全注意力。两者在 `max_context <= sliding_window` 时逐元素相同，所以
  `validate_max_context()` 与 `encode()` 各检查一次：超过就报错，并说明
  "参考实现会把这些 key 掩掉而 SlowQ 不会"。既不因为字段存在就拒绝加载，
  也不会把真正启用的滑动窗口当成全注意力算。

`family_for()` 同时检查 `model_type` 与 `architectures`，两者不一致就拒绝——
一个写着 `llama` 却声明 `Qwen3ForCausalLM` 的 config 会被静默地按错误的 bias 规则
解码。llama 家族的 `architectures` 里包含 `VLlama3ForCausalLM`：那是 SmolVLM 给
内嵌文本 config 起的名字，`Idefics3Model` 通过 `AutoModel.from_config` 按
`model_type` 分派，根本不看 `architectures`。

### 3.1 单层前向与张量 shape

记 `T` = 本次前向的 token 数，`S` = cache 中的总 token 数（`offset + T`）。
下面用 Qwen3-0.6B 的真实维度，括号里依次是 Qwen3-1.7B / Qwen2.5-0.5B /
Qwen2.5-Coder / SmolLM2-135M / SmolLM2-360M：

```
hidden_size H          = 1024    (2048 / 896 / 896 / 576 / 960)
intermediate_size I    = 3072    (6144 / 4864 / 4864 / 1536 / 2560)
num_attention_heads Nh = 16      (16 / 14 / 14 / 9 / 15)
num_key_value_heads Nk = 8       (8 / 2 / 2 / 3 / 5)
head_dim D             = 128     (128 / 64* / 64* / 64* / 64*)
vocab_size V           = 151936  (151936 / 151936 / 151936 / 49152 / 49152)
num_hidden_layers L    = 28      (28 / 24 / 24 / 30 / 32)
                              * = 由 hidden_size / Nh 推导
```

```
x                      [T, H]

# --- attention 分支 ---
input_layernorm        rms_norm(x, w, eps)                      [T, H]
q_proj                 linear(·, W_q[Nh*D, H], b_q?)            [T, Nh*D]
k_proj                 linear(·, W_k[Nk*D, H], b_k?)            [T, Nk*D]
v_proj                 linear(·, W_v[Nk*D, H], b_v?)            [T, Nk*D]
split_heads            [T, Nh*D] → [Nh, T, D]                   q: [16, T, 128]
                       [T, Nk*D] → [Nk, T, D]                   k,v: [8, T, 128]
q_norm / k_norm        rms_norm 作用在最后一维 D 上（仅 qwen3）    [Nh, T, D] / [Nk, T, D]
apply_rope             x*cos + rotate_half(x)*sin                同上
kv_cache.write         写入 slot [offset, offset+T)
kv_cache.read          读出 [0, S)                               k,v: [8, S, 128]
group_query_attention  见 3.2                                     [16, T, 128]
merge_heads            [Nh, T, D] → [T, Nh*D]                    [T, 2048]
o_proj                 linear(·, W_o[H, Nh*D], b_o?)             [T, H]
x = x + attention                                                [T, H]

# --- MLP 分支（SwiGLU）---
post_attention_layernorm                                         [T, H]
gate_proj              linear(·, W_gate[I, H], b?)                [T, I]
up_proj                linear(·, W_up[I, H], b?)                  [T, I]
silu(gate) * up                                                   [T, I]
down_proj              linear(·, W_down[H, I], b?)                [T, H]
x = x + mlp                                                       [T, H]
```

`b?` 表示"该家族该投影声明了 bias 才有"，由 `ModelConfig` 的三个布尔字段决定，
不在算子里猜。`linear()` 的 bias 参数是可选的，`None` 就是不加。

### 3.2 GQA：不物化 `repeat_kv`

HF 的做法是把 K/V 沿 head 轴复制 `Nh/Nk` 份。SlowQ 改为把 query head 折叠进
batch 轴：

```python
q_flat = q.reshape(Nk, group * T, D)            # [8, 2T, 128]
scores = q_flat @ k.transpose(0, 2, 1)          # [8, 2T, S]
```

query head `kv * group + r` 对应 KV head `kv`，与 `repeat_kv` 展开后的顺序逐元素
一致。`tests/test_ops.py` 用一个独立写出的 `naive_repeat_kv` 参考实现逐 head 对比，
覆盖 `(Nh, Nk) ∈ {(16,8), (4,2), (4,4), (8,1), (4,1)}`。

这样做的收益是实打实的：`S = 2048` 时物化 `repeat_kv` 每层每 token 要复制 16 MiB，
而折叠方案零复制。

**`Nk == Nh` 时 group size 是 1，这就是普通多头注意力**——SmolVLM 的视觉塔
（12 heads / 12 KV heads）复用同一个函数，不需要第二份实现。视觉塔的差别只是
`mask=None`（非因果），而 `group_query_attention` 本来就接受 `mask=None`。

### 3.3 权重布局与清单

所有权重按 HF 的 `[out_features, in_features]` 存储在磁盘上，运行时计算
`x @ W.T`。转置是 numpy view，不产生拷贝，也不需要把 checkpoint 预先转置。

`required_text_tensors(config)` 按家族与任务推导**必须存在**的张量集合：
`q_norm` / `k_norm` 只在 qwen3 出现，bias 向量只在该家族该投影声明了 bias 时出现，
`lm_head.weight` 只在**非 tied 且非向量任务**时出现。要求一个 checkpoint 合法地
不拥有的张量是**假失败**，静默跳过则是**真失败被藏起来**——两者都不行，
所以清单是算出来的，不是写死的一份 Qwen3 列表。

### 3.4 数值细节（都是踩过坑的）

| 细节 | 选择 | 原因 |
| --- | --- | --- |
| 掩码常量 | `np.finfo(np.float32).min` | 与 HF 一致；用 `-inf` 会让整行被掩的 softmax 出 NaN |
| softmax | 先减去行最大值 | FP32 下 `exp(88)` 就溢出 |
| sigmoid | 正负两支分别计算 | `1/(1+exp(-x))` 在 `x < -88` 溢出并触发 `RuntimeWarning`，SiLU 每层都会中 |
| RoPE 配对 | 半分割（`cat([-x2, x1])`） | 三个文本家族都用半分割，不是交错配对；搞错不会报错，只会算错 |
| RoPE `inv_freq` | 三个家族共用一份 | transformers 里三份 `*RotaryEmbedding` 逐字相同，且都不读 `rope_interleaved` |
| RMSNorm | 方差在 FP32 累加 | 与 HF 一致 |
| LayerNorm | 有偏方差 + weight **和** bias | 视觉塔用 `nn.LayerNorm`，与文本塔的 RMSNorm 是两套，不能复用 |
| 视觉 MLP 激活 | `gelu_tanh`（tanh 近似） | SmolVLM 的 `hidden_act` 是 `gelu_pytorch_tanh`；用 erf 精确版是每层一个小偏差，12 层累积起来会顶到门限 |
| 全程 dtype | FP32 | 任何算子都不允许出现 `.astype(np.float64)` |

唯一**故意**用 FP64 的地方是图像预处理的 `rescale`：官方 `rescale` 在 float64 里算、
存成 float32，`normalize` 全程 float32。pixel 张量是视觉塔逐字节看到的输入，
把两步都放进 float32 会移动舍入位置（见 §8.1）。

### 3.5 DebugProbe：位置语义

`DebugProbe` 是可选的中间激活捕获（默认关闭：留住一次 2048-token prefill 的全部
激活要几个 GiB）。它有两处容易被写错的地方，都是 v0.1 修过的：

- **位置是相对于本次前向的输入块的**，`0` 是这一块的首 token，`-1` 是末 token，
  对所有捕获点统一。decode 步的块只有 1 个 token，所以那里只有 `-1`/`0` 合法。
  块里没有的位置是**调用方错误**，直接抛 `ValueError`：夹断或取模会安静地报出
  另一个 token 的激活。位置在写 cache **之前**校验，不是等到第 28 层。
- **`ForwardResult.logits` 只覆盖最后一个位置**，所以任何其他位置的 logits 需要
  各自做一次输出投影（`_record_probe_logits`）。旧实现在这里直接复用末位的
  logits，于是"第 5 个位置的 logits"其实是最后一个位置的——键名对，数值错，
  而参考对齐恰恰要比的就是这些位置（v0.1 缺陷 #7）。只算被请求的那些位置：
  全词表投影是整个模型里最贵的单个操作，没人要的位置不该产生开销。末位复用
  生成时已经算好的那一份，因此与 `ForwardResult.logits` 逐 bit 相同。

v0.2 增加了 `record_sequence()`，只有 `embed_sequence` 用：图文融合后的
`[T, H]` 嵌入矩阵里，**哪一行拿到了哪个值**正是被测的东西，抽样位置会把大部分
融合结果漏掉。它是 prefill 尺寸的张量，所以按名字单独 opt-in。

`tests/test_model.py` 里有一条测试专门构造非末位置的请求，断言捕获到的**不是**
末位那一行。

---

## 4. KV cache 语义

```python
KVCache(max_context, num_layers, num_kv_heads, head_dim)
  k, v: [L, Nk, max_context, D]     一次性预分配，FP32
```

每 token 的缓存字节数是 `2 × L × Nk × D × 4`，实测值（`max_context=1024` 的总量）：

| 模型 | L | Nk | D | 每 token | @1024 |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B / Coder | 24 | 2 | 64 | 24.0 KiB | 24.0 MiB |
| SmolLM2-135M-Instruct | 30 | 3 | 64 | 45.0 KiB | 45.0 MiB |
| SmolVLM-256M（文本塔） | 30 | 3 | 64 | 45.0 KiB | 45.0 MiB |
| SmolLM2-360M-Instruct | 32 | 5 | 64 | 80.0 KiB | 80.0 MiB |
| Qwen3-0.6B / 1.7B / Embedding | 28 | 8 | 128 | 224.0 KiB | 224.0 MiB |

**两个 Qwen3 的每 token 缓存完全相同，尽管 hidden_size 差一倍**（1024 vs 2048）。
这是 `head_dim` 独立于 `hidden_size` 的直接后果：两者都是 28 层 × 8 个 KV head × 128。
反过来 Qwen2.5-0.5B 的权重比 Qwen3-0.6B 小，缓存却只有它的约 1/9——
KV cache 的大小与参数量没有直接关系。

三条不变量，`tests/test_cache.py` 与 `tests/test_model.py` 都有对应断言：

1. **`length` 只统计完整跑完一次前向的 token。** 每层写 `[offset, offset+T)`，
   读 `[0, offset+T)`；`commit(T)` 在**一次前向结束后调用一次**，不是每层一次。
   如果每层 commit，第 2 层就会看到自己刚写的数据被算作"历史"。
2. **cache 在每个请求开始时 reset，而不是结束时。** 中途抛异常的请求因此不会
   污染下一个请求。`reset()` 只把 `length` 归零，不释放也不清零内存——预分配
   的槽位会被下一次写入覆盖。
3. **容量不足必须抛错，绝不静默截断或覆盖。** `check_capacity` 在写入前检查；
   写入失败时 cache 保持原样（`length` 不变）。

存进 cache 的 K 是**已经过 q_norm/k_norm（如果该家族有）和 RoPE 的**，所以分块
边界不会改变任何 token 的绝对位置。
`tests/test_model.py::test_chunk_boundaries_do_not_shift_positions` 直接比较 cache
中的 K 来验证这一点。

位置从 cache 长度推导，`encode()` 拒绝与 `kv_cache.length` 不一致的显式
`position_offset`——分块 prefill 和 decode 因此不可能各自漂移。

**向量任务不按 `max_length` 预分配。** `EmbeddingModel.encode_ids()` 给每条文本
新建一个 `len(ids)` 长的 cache：一次编码就是一次 prefill，按 8192 的预算分配
会给一句 20 token 的话占掉几百 MiB 永远用不到的 K/V。分块仍然保留，
所以 attention 分数矩阵的峰值只与 chunk 大小有关。

---

## 5. 分块 prefill

`prefill_chunk_size`（默认 128）把长 prompt 切成若干次前向。第一次写 `[0, C)`，
第二次写 `[C, 2C)` 但读 `[0, 2C)`，以此类推。mask 用**绝对位置**构造：

```python
causal_attention_mask(num_queries=T, num_keys=S, position_offset=offset)
# query i 位于绝对位置 offset + i，可以看见 key j 当且仅当 j <= offset + i
```

因此下面三种路径必须在数值上等价，且测试就是这么断言的：

- 一次性 prefill 整个 prompt
- 分块 prefill（chunk ∈ {1, 2, 3, 5, 7}）
- prefill 后逐 token cached decode

唯一允许的差异是 FP32 归约顺序带来的噪声。真实模型上实测 rel_l2 在 1e-5 量级以内
（`tests/real_models.py` 的 `INTERNAL_REL_L2_TOLERANCE`，比跨实现门限紧 100 倍）。

分块不是性能优化，是**内存上界**：一次 prefill 2048 token 的 attention 分数矩阵是
`[16, 2048, 2048]` FP32 = 256 MiB，分块后峰值只与 chunk 大小有关。
实测中分块与整块的吞吐差在几个百分点以内（见[性能与内存](benchmarks.md)）。

**图像特征可以跨 chunk 边界。** `stream_ids()` 维护一个 `feature_cursor`，
每个 chunk 取自己那一段 mask，按 mask 里 `True` 的个数从特征矩阵顺序切片。
一张 9 帧的图产生 576 个占位 token，`prefill_chunk_size=128` 时它们分布在 5 个
chunk 里；假设图像一定落在一个 chunk 内会把特征写错位置，而写错位置**不会报错**
——张量形状是对的，只是行对不上。

---

## 6. 任务一：文本生成

以 `engine.generate([{"role": "user", "content": "你好"}])` 为例：

```
messages
  │ tokenizer.render_chat()          Jinja2 ImmutableSandboxedEnvironment
  │                                  trim_blocks/lstrip_blocks=True
  │                                  模板来自该目录的 tokenizer_config.json
  ▼
"<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"     (Qwen 系)
  │ tokenizer.encode()               byte-level BPE，add_special_tokens=False
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
  `generate_ids` / `stream` 都只负责把输入解析成 token id 再委托给它，
  `generate` 与 `_drain(stream(...))` 等价。因此"prefill 后逐 token decode"
  只有一份实现，图文任务也是继承它（§8）。
- **最后一个采样出的 token 不会被喂回模型。** 它交给调用方后循环即结束，所以
  请求结束时 `kv_cache.length == P + generated - 1`。这不是 off-by-one bug，
  `tests/test_engine.py` 对此有专门断言。
- **文本输出永远不包含终止用的特殊 token。** `decode(skip_special_tokens=True)`
  加上增量解码器的 hold-back 共同保证这一点。
- **停止符集合来自该目录自己的 `generation_config.json`**，不是 Qwen3 的常量。
  Qwen3 发布**两个** eos（`<|fim_middle|>` 与 `<|fim_prefix|>`），两个都必须终止生成；
  Qwen2.5 与 SmolLM2 各一个。少读一个的后果是生成停不下来，一路撞到
  `max_new_tokens`，而这在短回答上看不出来。

`Engine.from_directory()` 按 manifest 的 task 拒绝走错门的调用：向量目录报
"用 `EmbeddingEngine` / `slowq embed`"，图文目录报"用 `VisionEngine` / `slowq vlm`"。
在加载时拒绝并指名正确入口，好过构造出一个 causal LM、然后在第一次前向里因为缺
`lm_head` 失败。`CausalLMModel.__init__` 是第二道同样的检查。

---

## 7. 任务二：文本向量

`embedding.py` 用同一个 `DecoderModel`，但**不做输出投影、不做采样、不做生成**。

```
text
  │ is_query=True → f"Instruct: {instruction}\nQuery:{query}"   （Query: 后无空格）
  │ is_query=False → 原文不加前缀
  ▼
tokenizer.encode(payload, add_special_tokens=True)   ← 与生成路径相反
  │ 超过 max_length（默认 8192）报 ContextOverflowError，不截断
  ▼
input_ids
  │ cache = new_cache(len(ids))                      按文本长度分配，不按 max_length
  │ for chunk in chunks: decoder.encode(chunk, cache)
  ▼
hidden_states: [T_last_chunk, H]
  │ last_token_pool(hidden, len(最后一个 chunk))      不是平均池化
  ▼
pooled: [H]
  │ l2_normalize()                                   F.normalize 的 max(||x||, eps)
  ▼
vector: [1024] FP32，单位长度
```

四个容易被做错、且做错**不会报错**的点：

- **`add_special_tokens=True`。** 官方 recipe 调
  `tokenizer(texts, padding=True, truncation=True, max_length=…)`，那个默认值会跑
  `tokenizer.json` 的 post-processor。`Qwen/Qwen3-Embedding-0.6B` 的 post-processor
  是一个 `TemplateProcessing`，末尾追加 `<|fim_middle|>`（id 151643）——
  **官方的每一个向量都是在那个 token 上 pool 出来的**，少一个 token 就是另一个向量。
  生成路径反而传 `False`，因为它先渲染 chat template，而 `apply_chat_template`
  就是这样 tokenize 自己的输出的。同一个库、两个官方调用点、两个答案；
  `tools/validate_reference.py` 在两侧都比 token id。这是**实测**的：SlowQ 支持的
  三个生成 checkpoint 加不加都一样，这一个恰好加一个。
- **保留因果注意力，不是双向。** 官方 Qwen3-Embedding 就是 causal LM 的
  hidden state，改成双向会得到一个"看起来更像 embedding"但与参考不同的向量。
- **pool 的行号来自最后一个 chunk 的长度，不是 `len(ids)`。** 传 `len(ids)`
  在文本超过一个 chunk 时会去取 decoder 根本没返回的那一行。
- **`last_token_pool` 的有效 token 数由调用方给出**，不从 `hidden_states.shape[0]`
  读。SlowQ 一次编码一条、不 padding，所以两个来源现在一致；但官方 snippet 在
  左 padding 时取 `[:, -1]`、否则取 `attention_mask.sum() - 1`，将来若加 padded
  路径，写死的 `shape[0]` 会静默 pool 到一个 pad token。

输出维度是 `hidden_size`（1024），不是词表维度（这个 checkpoint 是 151669）。
`cosine_similarity()` **故意不重复归一化**：`encode` 已经产出单位向量，
再归一化一次会把"调用方传进来的根本不是 embedding"这件事藏起来。

超过 `max_length` 时报错而不截断，理由与生成路径相同但更硬：一段被截断的文档
嵌入出来的向量与完整文档的**不是同一个向量**，而调用方拿到的仍是一个
形状正确、模长为 1 的数组——这是一个静默的错误答案。

`EmbeddingEngine.from_directory()` 用 `require_chat_template=False` 加载 tokenizer：
官方 recipe tokenize 的是原文，一个不带 chat template 的向量 checkpoint 仍然完全
可用。`render_chat()` 在这种情况下报错，而不是发明一种格式。

`EmbeddingEngine.search()` 是 SlowQ 全部的检索支持：一个 query 对一批 document，
返回向量和按余弦降序的索引。**没有向量库，没有 RAG 管线。**
`encode_many()` 是顺序编码，不是批处理——官方 recipe 的 padding 只是为了把一批
塞进一次前向，逐条编码得到相同的向量而不需要 padding。

---

## 8. 任务三：图文生成

`vlm.py` 是本轮最大的新增模块。它的文本塔**就是** §3 那个 decoder
（`tensor_prefix="model.text_model"`，family `llama`，30 层），
新代码只有预处理、视觉塔、connector 和融合。

### 8.1 预处理：一张图不是一帧

`Idefics3ImageProcessor` 的流水线，SlowQ 逐步复现（顺序不可换）：

```
PIL 图像（RGB / RGBA；其他模式按名字拒绝）
  │ resize：最长边 → size.longest_edge（本 checkpoint 2048）
  │         再 → 不超过 4096 的上界
  │ resize：两边向上取整到 max_image_size.longest_edge（512）的倍数
  │         ← 不保持长宽比，这正是 crop 网格能整除的原因
  ▼
[H, W] 数组
  │ split_image：按行优先切成 rows × cols 个 crop，
  │              **再把整张图 resize 到 512×512 作为最后一个"全局帧"**
  ▼
rows*cols + 1 帧，每帧 512×512
  │ convert_to_rgb（RGBA 在白色背景上做 alpha 合成）
  │ rescale：float64 里乘 1/255，存 float32
  │ normalize：float32 减 mean 除 std（本 checkpoint 两者都是 0.5）
  │ pad：在**归一化后的空间**里补零（0 在那里是中灰，不是黑）
  │ transpose → channels-first
  ▼
pixel_values: [num_frames, 3, 512, 512] FP32
pixel_attention_mask: [num_frames, 512, 512] int64（单图时全 1）
```

两处顺序是**可观测的**，不是风格问题：

- **resize 在 RGB 转换之前**，因为官方就是这个顺序：先插值 RGBA 再合成到白底，
  与先合成再插值不是同一批像素。
- **pad 在 normalize 之后**，所以补的是归一化空间里的 0（中灰），不是黑。

`resize_output_size_rescale_to_max_len()` 里那句看着多余的
`if height % 2 != 0: height += 1` 是官方代码，它改变下一步 resize 的长宽比，
所以必须连浮点除法、`int()` 截断和这个 +1 一起复现，才能落到同样的尺寸上。

实测的两个样例（图片由验证工具生成，不入库；prompt token 数取决于问题文本，
这两行用的是验证抓取里的两条：grid-1x4 配 `Describe the shapes you can see in this image.`
加 system `You are a helpful assistant.`，grid-2x4 配 `What colours appear in this picture?`
且不带 system）：

| 样例 | 原图 → encoder size | crop 网格 | 帧数 | image token | prompt token |
| --- | --- | --- | --- | --- | --- |
| grid-1x4 | 300×1200 → 512×2048 | 1×4 | 5 | 320 | 360 |
| grid-2x4 | 400×800 → 1024×2048 | 2×4 | 9 | 576 | 612 |

"只支持一张图"**不等于**可以跳过这张图产生的多个 crop：跳过它们等于悄悄丢掉
图像的大部分信息，而输出仍然是一句看起来合理的话。

### 8.2 prompt 里的占位符排布

`image_prompt_string()` 生成的字符串按**帧顺序**排布，因为 `inputs_merger` 是按
索引递增填 `<image>` 位置的，所以 prompt 顺序**就是**帧顺序：

```
<fake_token_around_image><row_1_col_1> <image>×64
<fake_token_around_image><row_1_col_2> <image>×64
...
\n<fake_token_around_image><global-img> <image>×64 <fake_token_around_image>
```

`image_seq_len = 64` 来自 `processor_config.json`，并在加载时与几何交叉校验：
`((image_size 512 // patch_size 16) ** 2) / scale_factor**2 = 1024 / 16 = 64`。
两者不一致会在**加载时**报 `ConfigError`，而不是等到第一个请求才发现 prompt 里的
占位符数量与 connector 的输出对不上。`vision_config.image_size` 与
`max_image_size.longest_edge` 也必须相同：位置嵌入是按 `32×32` 的网格学的，
别的尺寸的帧没有合法的 position id。

### 8.3 视觉塔与 connector

```
[3, 512, 512]
  │ conv2d_patchify（k=s=16，valid）→ 不做卷积：切成不重叠的 [3,16,16] 块，
  │   按 PyTorch 的 [out, in, kh, kw] 顺序展平（channel-major），一次 matmul
  ▼
[1024, 768]
  │ + position_embedding[bucketized position ids]      学习到的位置，没有 RoPE
  ▼
12 × Idefics3VisionEncoderLayer:
  │   x + Attention(layer_norm1(x))     LayerNorm（weight+bias）、非因果、
  │                                     q/k/v/out_proj **四个都带 bias**
  │   x + fc2(gelu_tanh(fc1(layer_norm2(x))))
  ▼
[1024, 768]
  │ post_layernorm（LayerNorm）
  ▼
encoder_output: [num_frames, 1024, 768]
  │ pixel_shuffle(scale_factor=4)：[N, h*w, d] → [N, (h/4)*(w/4), d*16]
  ▼
[num_frames, 64, 12288]
  │ modality_projection.proj（nn.Linear，**bias=False**）
  ▼
features: [num_frames * 64, 576]        已按帧顺序展平
```

与文本塔**没有任何数值上的共享**：LayerNorm 而不是 RMSNorm，fc1/fc2 + GELU-tanh
而不是 SwiGLU，四个投影都带 bias，非因果，位置是学出来的。唯一复用的是
`group_query_attention`（12 heads / 12 KV heads，group size 1，`mask=None`）
和 `linear` / `split_heads` / `merge_heads`。

三处实现细节，写错都不报错：

- `pixel_shuffle` 的两组 `permute`/`reshape` **不是**恒等变换：它们决定哪几个相邻
  patch 被拼成一个输出特征，也就决定 connector 的投影看到什么。逐字复现这个序列
  比推理一个"等价"的写法更便宜，也更不容易错。
- `conv2d_patchify` 里的转置是 `(nh, nw, C, kh, kw)` 而不是 `(nh, nw, kh, kw, C)`
  ——PyTorch 的卷积权重按 `[out, in, kh, kw]` 展平，是 channel-major。
  写反了会得到一个形状正确、数值全错的 patch 嵌入。
- `vision_position_ids()` 复现官方的 `bucketize`：`boundaries = arange(1/n, 1, 1/n)`，
  把一帧的 patch 网格映射到位置嵌入训练时用的 `n × n` 网格。在满网格上它恰好退化为
  `arange(n*n)`，也就是 SlowQ 单图路径总是产生的那种情况——但退化是**算出来的**，
  不是假设的。

视觉塔的注意力传 `mask=None`：它是**非因果**的，每个 patch 都能看到其他所有 patch，
唯一需要屏蔽的是 padding，而单图路径不会产生 padding——每帧都被 resize 成正好
`image_size` 见方，所以 pixel mask 全是 1。官方 `Idefics3VisionTransformer` 在
没有 padding 时同样直接丢掉 mask，传 `None` 让全有效快速路径与参考逐 bit 相同，
而不是给每个 score 加一个 0。真出现部分 padding 的 patch 网格时
`VisionTower.encode` **报错**，不做屏蔽：那说明预处理已经偏离了官方规则，
悄悄屏蔽掉会让一张被裁错的图仍然生成出一句看起来合理的话。

### 8.4 融合与生成

```
build_prompt(image, question)
  │ 预处理 → PreparedImage
  │ render_template(messages with {"type": "image"} parts)   ← 结构化 content
  │ expand_image_tokens()：把唯一的 <image> 占位符换成 §8.2 的展开串
  │ tokenizer.encode()
  │ 校验：<image> token 数 == num_frames * image_seq_len     不等就报错
  │ 校验：prompt 长度 <= max_context                         超了就报错
  ▼
VisionPrompt(text, token_ids, image, image_positions)

stream_prompt(prompt)
  │ model.encode_image(prompt.image)          视觉塔 + connector，一次
  │ mask = 在 image_positions 上为 True 的布尔向量
  ▼
Engine.stream_ids(token_ids, prefill_embeds=(features, mask),
                  vision_encode_seconds=…)
  │ 分块 prefill：DecoderModel._merge_inputs_embeds()
  │   hidden = embed(ids); hidden[mask] = features      ← 计数不符就抛错
  ▼
之后就是 §6 的普通 cached decode
```

三点设计选择：

- **`prefill_embeds` 是 `stream_ids` 的参数，不是引擎状态。** 图像特征因此不可能
  泄漏到下一个请求。`vision_encode_seconds` 走同一条路：它属于首 token 延迟，
  不带它就会报出一个悄悄漏掉最大开销的 TTFT（实测占 81%~87%）。
- **decode 阶段不再融合图像特征**，因为官方 `Idefics3Model.forward` 只在
  `past_seen_tokens == 0` 时 merge。所以生成循环一行都不用改。
- **`_merge_inputs_embeds` 拿 mask 的和与特征行数对账，而不是相信调用方。**
  数量不符时报错并说明"两边对不上，没有正确的对齐方式"，而不是丢掉图像特征
  或者把它们写到文本 token 上。

消息用**结构化 content**（`[{"type": "image"}, {"type": "text", ...}]`）渲染，
因为 SmolVLM 的模板会索引 `message['content'][0]['type']`；传纯字符串会在 Jinja
里抛异常，而不是产出一个 prompt。`Tokenizer.render_template()` 就是为这类调用方
留的口子：它跳过 `normalize_messages()` 的纯文本校验（那个校验会拒绝 list 形式的
content），直接渲染调用方准备好的模板变量。

一次请求只接受**一张**图：`expand_image_tokens()` 数占位符，不是恰好一个就报错。
图像缺失、读不出来、模式不支持（调色板 / 灰度 / 16-bit）、占位符数量对不上、
扩展后超过 `max_context`，都给出说明原因的错误，不静默截断或忽略图像。
SlowQ 也不抓 URL：传路径，或者传一个已经加载好的 `PIL.Image.Image`。

---

## 9. 计时区间定义

所有时长来自 `time.perf_counter()`，只覆盖模型工作，**不包含下载、转换、也不
包含调用方在 yield 之间打印所花的时间**。

| 名称 | 包含 | 不包含 |
| --- | --- | --- |
| `load_seconds` | 读 `config.json` / `manifest.json`、mmap 权重、编译 chat template | 下载、转换；也**不含把权重 page-in**——mmap 是惰性的，第一个前向才真正读盘 |
| `tokenize_seconds` | 渲染 chat template + BPE 编码 | — |
| `vision_encode_seconds` | 每一帧过视觉塔 + connector | **图像预处理**（解码、resize、切 crop、prompt 扩展）在 `build_prompt` 里，不计入 |
| `prefill_seconds` | 所有 prefill 前向（每个 chunk），含 KV 写入与图像特征融合 | tokenize、视觉编码、采样 |
| `first_token_seconds` | `tokenize + vision_encode + prefill + 第一次采样`，从请求开始计 | 模型加载 |
| `decode_seconds` | 首 token 之后每一步的 forward + sample | 消费者在两次 yield 之间做的事 |
| `total_seconds` | `tokenize + vision_encode + 从请求开始到结束的墙钟时间` | 模型加载 |

`vision_encode_seconds` 只有图文请求才有值，`GenerationMetrics.report()` 也只在它
`> 0` 时才打印那一行——纯文本请求的输出不变，不会多出一排 `0.0 ms`。
向量任务用的是 `EmbeddingResult.encode_seconds`，它覆盖分块 prefill 与 pooling，
不含 tokenize。

token 计数：

- `prompt_tokens` 是**渲染后 chat template 的 token 数**，不是用户原文的字数。
  图文请求里它包含全部 image token（grid-2x4 是 612，其中 576 个来自图像）。
  prefill 吞吐必须按模型真正吃进去的量算。
- `decode_tokens = max(generated_tokens - 1, 0)`，即去掉第一个 token。
  `decode_tokens_per_second` 因此是稳态速率，不被 prefill 拉低。
- `prefill_chunks` 是实际执行的前向次数。

`tools/benchmark.py` 的微观 prefill 探针把**两种模式分开报**，因为它们是两条不同的
代码路径：`full` 走一次覆盖整段的前向，`chunked` 走引擎真正跑的分块循环。两者的
"前向次数"都来自**实测的调用次数**（探针用包装器数），不是 `长度 // chunk_size`
推算出来的——推算出来的数字曾经被当成实测结果写进文档（v0.1 缺陷 #6），所以这里
只信调用计数。纯 prefill 计时区间里没有采样、没有 decode、没有打印。
`benchmark.py` 用 `Engine.from_directory` 加载，因此**只支持文本生成**；
向量与图文的数字来自各自引擎的计时字段，测法见[性能与内存](benchmarks.md)。

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

**把 mmap 数组的 `nbytes` 相加冒充 RSS 是错的**：没被读过的页根本不常驻。
实测（Apple M5 Pro，`max_context=1024`）：Qwen3-1.7B 映射完 6.4 GiB 权重后峰值
RSS 只有 175.5 MiB，跑完一次生成后是 6.78 GiB；SmolVLM 的峰值 RSS（1471.4 MiB）
反而**超过**权重逻辑体积（978.4 MiB），差额是 pixel 张量与视觉塔的中间激活。
三个数字不能相加，也不能互相替代。

---

## 10. 权重转换与加载

`tools/convert_model.py` 是**唯一**接触 safetensors / BF16 的地方：

```
models/<name>-hf/                    models/<name>-slowq/
  config.json                          config.json          (原样拷贝)
  generation_config.json               generation_config.json   (按任务判断是否必需)
  tokenizer.json                       tokenizer.json
  tokenizer_config.json                tokenizer_config.json
  vocab.json / merges.txt              vocab.json / merges.txt
  added_tokens.json                    added_tokens.json         (有就拷)
  special_tokens_map.json              special_tokens_map.json   (有就拷)
  chat_template.json                   chat_template.json        (有就拷)
  preprocessor_config.json    ────>    preprocessor_config.json  (图文任务)
  processor_config.json                processor_config.json     (图文任务)
  modules.json                         manifest.json  ← task / model_family / source
  config_sentence_transformers.json    tensors/
  1_Pooling/config.json                  model.embed_tokens.weight.npy
  model.safetensors                      model.layers.0.self_attn.q_proj.weight.npy
  model.safetensors.index.json           ...            (每个张量一个文件)
```

BF16 → FP32 是一次位移，不是算术运算：

```python
uint16 → uint32 → << 16 → view(float32)
```

所以转换必须**逐 bit 精确**，`TOLERANCES["max_abs_weights"] = 0.0` 就是这个意思。
也正因为它不经过任何浮点运算，转换结果与 BLAS、与指令集都无关——v0.1 在 Linux 上
重转 Qwen3-0.6B，310 个 `.npy` 与 macOS arm64 的结果 sha256 逐个相同（历史批次，
见 §16）。

### 10.1 task 检测：读文件，不读仓库名

`--task auto|text-generation|text-embedding|image-text-generation`，默认 `auto`。
`auto` 看的是 checkpoint 里的**证据**：

| 任务 | 证据 |
| --- | --- |
| `text-embedding` | sentence-transformers 导出的三个标记文件：`modules.json`、`config_sentence_transformers.json`、`1_Pooling/config.json` |
| `image-text-generation` | config 里嵌套 `vision_config`，顶层 `model_type` 是 wrapper 类型（`idefics3`），且有 processor 文件 |
| `text-generation` | 以上都不是 |

**三个标记文件只出现一部分时报错，不猜。** 少一个可能是导出方式不同，也可能是
目录不完整；两种情况的正确处理都不是"当成生成模型转下去"。显式指定的 task 仍会与
文件证据**交叉校验**：写了 `--task text-embedding` 但目录里没有标记文件，
转换失败并说明看到了什么。

向量模型的张量名还要重命名：`Qwen/Qwen3-Embedding-0.6B` 的导出没有 `model.` 前缀
（`embed_tokens.weight` 而不是 `model.embed_tokens.weight`），manifest 的
`TensorSpec.source_name` 记录原名，运行时用的是规范名。

**并非所有任务都需要同一批文件。** `generation_config.json` 与 chat template 对
向量任务不是必需的，缺了不致命；`preprocessor_config.json` 与 `processor_config.json`
对图文任务是必需的，缺了就无法离线复现预处理。必需与否按任务判断，
不是一刀切的"缺文件就报错"。

### 10.2 manifest

`manifest.json` 记录 `format_version`、`compute_dtype`、`task`、`model_family`、
来源 repo/revision/文件 sha256、每个张量的
`name/file/shape/dtype/source_dtype/source_name`，以及 `aliases`
（`lm_head.weight → model.embed_tokens.weight`，tied embedding 只存一份）。

| `format_version` | 写入者 | `task` / `model_family` | v0.2 行为 |
| --- | --- | --- | --- |
| 1 | 只有 SlowQ v0.1 | 必须**不存在** | 可读；task 按 v0.1 的唯一可能取值处理为 `text-generation` |
| 2 | SlowQ v0.2 | 两者都**必须存在** | 可读，也是唯一会写出的版本 |

版本 1 的默认不是"猜"：v0.1 只转换过 Qwen3 causal LM，所以缺键时的可能取值只有一个。
版本 1 却带着这两个键会被拒绝（那意味着有人手工改过），其他版本号也被拒绝，
`Weights` 不会静默降级。**v0.1 转好的 Qwen3-0.6B 目录不需要重新转换**，
`generate` / `chat` / `inspect` 的行为与 v0.1 一致；本轮验证的 8 个模型里就有这一个
v1 目录，专门用来走读旧格式的路径。

每个 repo 解析并记录**自己当时**的 commit。v0.1 为 Qwen3-0.6B 固定的那个 revision
不会被套到别的 repo 上——那会让缓存目录和 manifest 里的 sha256 描述一份根本没
下载过的权重。

### 10.3 写入侧的安全约束（都是踩过坑之后加的）

- **默认拒绝非空 `--out`，但接受"不存在"和"存在且为空"。** 转换是"先删后写"式的
  破坏性操作，把一次失败的转换写进已有模型目录，结果是**旧 manifest 仍然有效、但
  `tensors/` 里新旧混杂**——加载时不会报错，只会算出错的答案。要覆盖必须显式给
  `--overwrite`。保护的是**内容**，不是路径本身：一个空目录里没有内容可保护。
- **前后两次检查必须由同一个动作完成。** `check_output_is_installable()` 在转换前
  跑，`install()` 在转换后跑；早先两者用的是不同判据（前者 `not entries`，后者
  `output.exists()`），于是**预先建好的空 `--out` 能通过前检查、却在几十秒转换之后
  被后检查拒绝**。现在后检查不再自己判断，直接让 `staging.replace(output)` 这个
  原子 rename 去判：目录为空则成功，非空则 `OSError`（macOS/Linux 上是
  `ENOTEMPTY`），报错时列出实际条目并说明"里面一个字节都没动"。判据只有一份，
  就不会漂移（v0.1 发布收尾 #2）。
- **给了 `--overwrite` 也不是就地写。** 整个转换 + 自检在目标旁边的暂存目录里完成，
  全部通过之后才 `os.replace` 换过去；中途任何一步失败都只留下暂存目录，
  旧模型一个字节都没动。逐 `.npy` 原子写**不够**：单个文件原子，整体不原子。
- **tied 权重要读它自己的分片。** `lm_head.weight` 与 `model.embed_tokens.weight`
  逐 bit 相同的自检，必须分别从 `model.safetensors.index.json` 指出的**各自那个
  分片文件**取字节，不能顺手复用当前正在转换的分片（v0.1 缺陷 #8）。
  Qwen3-1.7B 是 2 个分片、6.4 GiB，用来在更大的模型上验证这条路径。
- **下载缓存按 repo + 解析后的完整 commit 分目录**，不按文件名。否则同一文件名
  在不同 revision 下会互相覆盖，而 manifest 里记的 sha256 是"实际用到的那份"——
  缓存命中错了文件，记录就变成假的（v0.1 缺陷 #4）。
- **中断的下载要能靠重跑同一条命令续上，而"完成"必须是真完成。** 快照标记
  `.slowq-snapshot.json` 在**下载第一个文件之前**就写，带 `complete: false`；
  每个文件下完后再按 revision 的 listing（先比大小，再比 LFS sha256）复核一遍，
  全部通过才改写成 `complete: true`。于是短文件不会被当成缓存命中，永远匹配不上的
  快照永远不会被标记完成，而没有标记的目录**仍然拒绝**（错误信息给出三条安全出路：
  删掉那一个目录后重跑、换 `--download-dir`、用 `--source`），既不自动删除也不清空
  缓存（v0.1 发布收尾 #1）。
- **截断的分片要报出自己的错误。** safetensors 头里的字节区间可能超过文件实际长度。
  `memoryview` 切片会像 `bytes` 一样**静默夹断**，短缓冲区一路传到 NumPy 才炸，
  而回溯指向切片那一帧，于是 `mmap.close()` 抛 `BufferError` 把真因盖掉。
  `tensor_view()` 先按分片实际长度校验区间，报错里给出张量名、字节区间、分片名和
  文件真实长度（v0.1 缺陷 #10）。

### 10.4 加载侧的安全约束

- `np.load(..., mmap_mode="r", allow_pickle=False)`。**`allow_pickle=False` 是
  显式的**：`.npy` 可以携带 pickle 载荷，允许它等于允许任意代码执行。
  `tests/test_weights.py` 有一个专门构造 pickle `.npy` 的测试，断言加载被拒绝。
- 数组是只读的 `np.memmap`，运行时无法就地改写权重。
- `check_names()` / `check_complete()` 在构建模型时校验清单齐全、shape 与磁盘一致、
  dtype 是 FP32。缺失或形状错误在**加载时**报错，不会等到前向算到那一层。
  图文模型检查**两套**清单：`SmolVLMModel` 先按 `config.text` 检查文本塔，
  `VisionModel` 再按 `required_vision_tensors()` 检查视觉塔与 connector。
  两套清单不重叠，所以"视觉塔少了一半张量"不会被文本塔的检查掩盖。
- 转换后的推理**完全离线**。`tests/real_models.py::block_network` 把
  `socket.create_connection` 与 `socket.socket.connect` 换成会抛异常的桩；
  8 个模型各有一条测试在断网状态下跑 `slowq inspect --json` 加对应任务的命令
  （`generate` / `embed` / `vlm`），要求退出码 0 且输出的 `source` 与 manifest
  逐字一致。

---

## 11. Tokenizer 与增量解码

- Jinja2 用 `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`，
  并注入 `tojson` 与 `raise_exception`——官方 Qwen3 template 会用到这两个。
- **模板来自该目录自己的 `tokenizer_config.json`（或 `chat_template.json`）**，
  不是手写的，也不猜 BOS/EOS。四个家族的模板各不相同，用哪一份取决于转换时拷了
  哪一份。
- `enable_thinking=False`（默认）选择 Qwen3 模板的非思考分支，assistant 轮末尾插入
  空的 `<think>\n\n</think>` 标记；其他家族的模板忽略这个变量。
- 停止 token 集合从 `generation_config.json` 的 `eos_token_id` 读取，
  缺失时回退到 `config.json`（HF 就是这么做的）；两边都没有则报错，不猜停止符。
- `add_special_tokens` **两条路径取值不同**，且都是官方行为：生成路径 `False`
  （先渲染模板），向量路径 `True`（跑 post-processor）。见 §7。
- `render_template()` 是不做消息校验的渲染口子，供 `slowq.vlm` 这类需要自己构造
  结构化 content 的调用方使用；`render_chat()` 是带校验的常规入口。
  没有 chat template 时两者都报错，不会发明一种格式。

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
`GenerationResult.text`，于是流式消费者拼出来的字符串与阻塞式调用返回的完全一致：

```
"".join(step.text_delta for step in engine.stream(...)) == engine.generate(...).text
```

EOS 和 `max_new_tokens` 走的是同一个分支，所以两条路径都 flush 过——v0.1 的
缺陷 #3 正是只在其中一条上漏掉，最后一个 token 的尾巴字符（中文和 emoji 最容易
触发）就只出现在 `result.text` 里，流式那边凭空少字。`tests/test_engine.py` 对
两条路径都断言了上面那个等式。SmolLM2 的词表里 emoji 会被切成更多 token，
这条路径因此在四个家族上都被走过。

---

## 12. 采样

过滤顺序与 `transformers` 的 logits warper 流水线一致：
**temperature → top-k → top-p**。

- `temperature == 0` 表示 greedy（argmax），并**跳过** top-k/top-p，这样 greedy
  模式下这两个参数不会悄悄起作用。
- top-p 保留"累积概率刚好跨过阈值"的那个 token：`remove[1:] = remove[:-1]`。
  少这一步会把 nucleus 少留一个候选。
- argmax 平局取 id 最小的，与 `np.argmax` 一致，保证可复现。
- RNG 是私有的 `np.random.Generator(PCG64(SeedSequence(seed)))`，**绝不碰全局
  `np.random`**，因此固定 seed 可复现，且库的其他部分无法移动它。
- 非法参数、空候选集、非有限 logits 一律抛 `SamplingError`，不会退化成随便挑一个。
- 默认采样参数来自该目录的 `generation_config.json`（`temperature` / `top_k` /
  `top_p` / `do_sample`），不是 Qwen3 的常量。

采样只服务两个生成任务。向量任务不采样；图文任务的 decode 阶段复用同一个
`Sampler`，因为融合只发生在 prefill。

---

## 13. 依赖边界

运行时只依赖三个包：

```
numpy >= 1.26
tokenizers >= 0.20
Jinja2 >= 3.1
```

可选 extra 两个，都不是运行时依赖：

```
slowq[vision]  Pillow >= 10.0        只有 slowq.vlm 真正读图时用
slowq[ref]     torch / transformers==4.51.3 / safetensors /
               huggingface_hub / Pillow    只有 validate_reference.py 的 ref 子命令用
```

`torch` / `transformers` 在 `tools/validate_reference.py` 的 `ref` 子命令里
**函数体内惰性 import**。`tools/convert_model.py` **一个都不需要**：它自己解析
safetensors 容器（8 字节小端头长度 + JSON 头 + `mmap` 数据段）、自己做 BF16→FP32
位移，不依赖 `safetensors` 也不依赖 `torch`。Pillow 只负责解码与 resize，
之后的每一个像素都是 NumPy。

这条边界由 `tests/test_dependency_boundary.py`（6 个测试）看守，三种方式：

1. 子进程 import 全部 **14 个**运行时模块，检查 `sys.modules` 里有没有 torch 等
   8 个禁用包；
2. 同一个探针换成检查 `PIL`：`import slowq.vlm` 不得拉进 Pillow，否则纯文本安装
   会在导入期就坏掉；
3. AST 扫描 `src/slowq/**/*.py`，任何 `import torch` / `from transformers ...`
   （包括函数体内的惰性 import）都会失败。

第三种是必要的：第一种只能发现被测试实际调用到的路径。另外还有两个测试直接读
`pyproject.toml`，断言 `dependencies` 恰好是那三个包、`vision` extra 存在且含
Pillow、`dev` extra 不含任何禁用包。

**一个必须说清楚的例外：** `tokenizers` 自己声明了 `huggingface-hub>=0.16.4`
作为硬依赖，所以 `pip install slowq` 会把 `huggingface_hub` 连同它的 HTTP 栈一起
装进来。这不在 SlowQ 的控制范围内。但**装进来不等于被用到**：本轮在开发 venv 里
import 全部 14 个运行时模块后，`sys.modules` 共 268 个模块，其中第三方顶层包只有
`numpy` / `tokenizers` / `jinja2` / `markupsafe`（外加 editable 安装带来的
`_distutils_hack`）——`torch` / `transformers` / `safetensors` / `huggingface_hub` /
`PIL` 一个都没有。依赖边界约束的是 import 图和网络行为，不是 pip 的传递闭包。

转换后的推理**完全离线**，不会自动下载任何东西（§10.4）。

---

## 14. 测试分层

```
默认（快速）   python -m pytest -m "not integration"
              685 passed, 62 deselected，验证工具修复轮实测 6.42 s（macOS arm64；
              发布前修复轮是 665 passed / 5.42 s，当时宿主机 swap 已用 7.4 GiB / 8 GiB，
              空载约 4.1 s——耗时随宿主机压力浮动，通过项数不受影响）
              去掉标记过滤直接 pytest -q 则是 685 passed, 62 skipped：
              那 62 项集成测试因未设 SLOWQ_MODEL_DIR 而跳过
              使用 tests/conftest.py 现场构造的微型 checkpoint 与真正的
              byte-level BPE tokenizer，不下载任何权重，全程离线

显式启用      SLOWQ_MODEL_DIR=models/<name>-slowq python -m pytest -m integration
              每个目录收集 62 项，manifest 里的 task 决定哪些模块运行、
              哪些整模块跳过并打印原因
              发布前修复轮 8 个模型合计 208 passed、0 failed（验证工具修复轮未重跑；
              8 份日志留在本机 artifacts/repair-review-20261021/developer-integration-logs/，
              该目录被 .gitignore 忽略，不随仓库分发）；
              其中 Qwen3-1.7B 单独复跑耗时 4036.71 s，正确性回归已通过，
              异常耗时与宿主机内存压力、页面调入调出相符但尚未做对照测量，
              见 validation.md「Qwen3-1.7B 的异常耗时与宿主机内存压力相符」

合计          747 个测试被收集（685 快速 + 62 集成）
```

集成测试按任务分成四个模块，过滤条件只有一个来源——目录自己的 manifest：

| 模块 | 项数 | 任务过滤 |
| --- | --- | --- |
| `test_integration_reference.py` | 8 | 无（任何目录都跑）：与固定参考比较、离线 `inspect` |
| `test_integration_real_model.py` | 19 | `text-generation` |
| `test_integration_embedding.py` | 17 | `text-embedding` |
| `test_integration_vlm.py` | 18 | `image-text-generation` |

**"每个目录都收集 62 项、通过数不同"是设计而非缺漏**，但它要求读 skipped 列表：
一个文本生成目录跑出来的 `26 passed, 36 skipped` 说明另外两个任务的模块根本没运行，
不能当成完整集成验证。8 个模型的逐项结果见[测试与数值验证](validation.md)。

快速套件里四个文件是**工具层**的回归测试，不是模型层的：
`test_validate_reference.py`（111）、`test_convert_model.py`（30）、
`test_benchmark.py`（27）、`test_dependency_boundary.py`（6）。它们守的不是
"算得对不对"，而是"工具在失败时会不会说谎、会不会破坏数据"——转换失败留下混杂
权重、比较器对 NaN 或缺失张量判 PASS、基准报告没有真正执行的分块，这三类问题的
共同点是**结果看起来正常**。v0.1 新增的回归测试都逐个验证过：撤掉对应的修复，
它们会失败。v0.2 发布前修复那一轮新增的 19 项、验证工具修复那一轮新增的 20 项
同样逐个负向验证过，逐条记录在[v0.2 报告 §6.6 与 §6.8](v0.2-多模型接入报告.md)。

那 20 项里有 4 项专门守"测试有没有真的执行被测函数"这件事：
把 `_reference_embedding_capture` 的函数体第一句换成 `raise AssertionError`，
加它们之前 91 项全绿，加之后 4 项失败。工具层的测试最容易写成
"在 fixture 里重述一遍正确行为"，那样它对生产代码的退化是盲的。

微型 fixture 刻意保留了各家族的别扭之处，而不是简化掉：

- `head_dim * num_attention_heads != hidden_size`（qwen3 夹具），任何从
  `hidden_size` 推导 `head_dim` 的代码在这里都会失败；
- 按家族分别构造**有 / 无 QKV bias**、**有 / 无 Q/K norm** 的 checkpoint，
  所以"qwen2 少算 bias""llama 多算 q_norm"这类错误在快速套件里就会暴露；
- tied 与 untied 两种输出投影都有夹具（SmolVLM 的文本塔是 untied）；
- `num_attention_heads / num_key_value_heads` 覆盖 2 与 3 两种 GQA 分组；
- norm 权重是 `N(1, 0.02)` 而不是全 1，忽略 norm 权重的 bug 藏不住；
- 视觉侧有独立的微型 ViT 夹具，覆盖 patch 顺序、mask、connector 与 image token
  注入的错误路径。

---

## 15. 参考对齐验证

`tools/validate_reference.py`（3620 行）让两套实现在**独立进程**中运行，
通过结果文件交换数据，这样几 GiB 的 torch FP32 模型和 NumPy 模型不会同时常驻：

```bash
M=models/<name>-slowq
H=<官方 checkpoint 快照目录>
python tools/validate_reference.py numpy   --model "$M"
python tools/validate_reference.py ref     --model "$M" --hf-dir "$H"
python tools/validate_reference.py compare --model "$M"
```

`--artifacts` 缺省时由 `--model` 推导为 `artifacts/validation/<模型目录名>`，
所以验证第二个模型不会覆盖第一个的抓取。artifact 格式本轮升到 **3**：
捕获层与摘要权重都从 checkpoint 推导，不再是一份固定的 Qwen3 列表，
并且抓取记录里写明它覆盖的是哪个 task。格式对不上就要求重抓，不自动放行。

参考端设置是**刻意固定**的：`torch_dtype=float32`、`eval()`、
`attn_implementation="eager"`（不用 SDPA/flash kernel）、CPU、batch size 1、
线程 4（SmolVLM 为 8）、`transformers==4.51.3`。

### 15.1 抓取点跟着家族与任务走

| | 文本生成 | 文本向量 | 图文生成 |
| --- | --- | --- | --- |
| 样例 | 4 个 chat prompt | 2 组 × 4 条文本 | 2 张图（生成的 PNG） |
| 捕获层 | `(0, L//2, L-1)` | 同左 | 文本 `(0, L//2, L-1)`，视觉另算 |
| 位置 | `(0, -1)` | `(0, -1)` | `(0, -1)` |
| 比较量 | logits + 各层 hidden + final_norm | 向量 + 相似度 + 各层 hidden | pixel 张量 / mask、视觉塔输出、connector 输出、融合后的 `embed_sequence`、logits |
| 比较项数 | 116 | 82 | 82 |

`capture_layers(L)` 取首、中、末，1~2 层的模型会去重（层数少于 3 时不该把同一层
capture 两次）。**抽样层不再固定在第 27 层**——那是 Qwen3-0.6B 的最后一层，
对 24 层的 Qwen2.5 根本不存在。

`weight_check_sample(arch)` 同理按家族选张量：qwen3 取 `q_norm` / `k_norm`，
qwen2 取 `q_proj.bias`，llama 两者都不取，向量任务不取 `lm_head`，
图文任务额外取视觉塔的首/中/末层（含 bias，因为 `Idefics3VisionAttention` 的四个
投影**都**带 bias，而它旁边的 llama decoder 一个都不带——转换器丢掉视觉 bias
否则不会被发现）。实测各模型的摘要张量数是 7~9 个，SmolVLM 是 22 个。

**"缺失的参数"与"合法的架构默认值"在比较前分开处理。** `ARCHITECTURE_FIELDS`
（19 项）与 `VISION_ARCHITECTURE_FIELDS`（15 项）里的每一项，都由参考侧用官方
modelling 代码自己的表达式从原始 `config.json` 复述一遍；两边一致才开始比数值。
所以架构层面的分歧不会被数值门限掩盖——一个 `head_dim` 读法不同的实现，
会在比任何张量之前就失败。

### 15.2 图文任务的图像怎么保证两侧看到同一份

参考侧不能自己下载或生成图像：那样两边比的可能是不同的像素。做法是
**NumPy 侧生成 PNG 并存进 artifact 目录，参考侧读取时按 sha256 校验**，
两侧的记录里都有这个摘要。图像文件不入库（`artifacts/` 被 Git 忽略）。
`compare` 另外检查 pixel 张量的形状、crop 网格、resize 前后的尺寸、
image token 排布与融合位置——**"生成了一句看起来合理的话"不算数值验证**。

### 15.3 权重摘要

第 2 项抽样的是**张量名**，不是张量内的元素。存进交换文件的不是张量本身，
而是一份**规范化摘要**（`DIGEST_SPEC_VERSION = "slowq-weight-digest-v1"`）：
sha256 覆盖 spec 字符串 + dtype（`<f4`）+ shape + 按 C 顺序遍历的**全部字节**。
摘要规范本身写进结果文件，所以"逐 bit 相同"这个说法是可核查的——它明确声明了
dtype、shape、字节序和字节表示，且**覆盖整份张量，不做步长抽样、不截断**。

这么改的直接收益（v0.1 上一轮）：`slowq.npz` 从压缩后约 300 MiB 降到约 11 MiB。
门限的覆盖面没有变小：摘要覆盖每一个元素，任何一个 bit 不同都会导致 sha256 不同。

### 15.4 容差

容差在**测量之前**声明（`TOLERANCES`）：

```
rel_l2_logits   1e-3      max_abs_logits   5e-2
rel_l2_hidden   1e-3      max_abs_hidden   5e-2
rel_l2_vector   1e-3      max_abs_vector   1e-3
rel_l2_weights  0.0       max_abs_weights  0.0
```

logits 与 hidden 的理由：FP32 单位舍入是 6e-8，一个 24~32 层、宽度 576~6144 的
残差网络加上 softmax 会把它放大约 1e-5 相对误差，而且两边用的是不同 BLAS
（Accelerate vs torch）与不同求和顺序。声明值取该估计的约 10 倍。

`vector` 是 L2 归一化后的向量或余弦相似度，模长恰为 1，所以 `rel_l2` **就是**两个
单位向量之间的距离，`rel_l2 <= 1e-3` 意味着 `cos >= 1 - 5e-7`；摊到 1024 个分量上，
单个分量的误差在 3e-5 量级，`max_abs` 声明在它上面约 30 倍——与 logits 用的余量
一致。

**v0.1 之后没有任何一个数字被放宽过**，本轮 8 个模型全部在原有门限内通过。
实测最差值逐模型列在[模型支持 · 参考数值验证结果](models.md#参考数值验证结果)：
最差 rel_l2 落在门限的 1/37 ~ 1/394，最差 max_abs 落在门限的 1/3 ~ 1/202。
最差 max_abs 八个模型都在**深层 hidden states** 上；最差 rel_l2 有 3 个模型在 logits 上，
SmolVLM 在视觉塔输出上，其余在 hidden states 上。

不要求跨平台逐 bit 相同，也不要求采样生成的文本与 Transformers 完全一致
（greedy argmax 必须一致，且实测全部一致）。

### 15.5 compare 不给假 PASS

比较器的默认倾向必须是"证据不足就失败"，否则它会安静地放行。`compare` 因此在
比数之前先做一轮资格审查：

- **应有的 key 集合是从抓取配置推导出来的**，不是"两个文件里恰好共有的 key"。
  `expected_numeric_keys()` 按任务分派到三个展开函数，少一个报错，多一个也报错。
- **两侧都缺也算失败。** 只看交集的话，一次抓取出错会让两边同时少掉同一批 key，
  比较照样 PASS。
- **0 个比较直接失败。** "没东西可比"不是通过。
- **元数据的应有集合同样从协议推导，不从文件推导。** 应有的样例名来自抓取配置
  （`capture["sample_names"]`，缺失时退回该任务的 `SAMPLES`），应有的权重名来自
  `weight_check_sample(arch)`；两侧同时缺失、单侧缺失都失败，错误信息指名是哪个
  样例、哪个权重（v0.1 发布收尾 #3）。
- **缺元数据即要求重抓，而不是跳过。** 没有 `reference.json`、没有架构指纹、
  没有摘要规范，都判失败。`reference.json` 只由 `ref` 侧写，NumPy 侧永不产生它
  ——所以没有可以从旧运行里"抄过来"的 PASS。
- **先查这份参考是不是在描述这一轮**：artifact 格式版本、task、样例名、捕获层、
  捕获位置、chunk 大小、decode 步数、架构指纹、摘要规范，任一项对不上就要求重抓。
  这是"不能沿用历史结论"在工具层的落点。
- **展平之前先比 shape**，比之前先查两侧是否有限——`nan > 容差` 恒为 False，
  旧版比较器正是因此把 NaN 判成了通过。参考范数为 0 的情况单独处理，不做除法。
- 失败时非零退出，并**指名是哪个样例的哪个张量**。

`tests/test_validate_reference.py`（84 个）逐条守着上面每一行，其中包括真的
`subprocess.run` 起 CLI、断言两侧同时少一条记录时**退出码非零**的测试。

集成测试每轮把已签收的 artifact 拷进 `tmp_path`，**用当前代码重跑 NumPy 侧**
（重新渲染 tokenizer、重新算权重摘要、重新前向）再与固定参考对比，
所以运行时代码或 tokenizer 一旦偏离签收时的行为，集成测试就会失败，
而 pytest 进程内不加载 torch。其中若干测试专门守着"不能沿用历史结论"：
把历史 PASS 原样抄进结果文件不算通过。

---

## 16. 历史批次记录

下面两节是 **v0.1 时期**的执行记录，原样保留，**本轮没有重跑**。
保留它们是因为它们回答的问题（干净环境能不能跑、换一种 BLAS 会怎样）
与本轮的问题不同，而重跑需要当时的环境。本轮的实际执行见
[测试与数值验证](validation.md)与[性能与内存](benchmarks.md)。

### 16.1 干净运行环境（v0.1，macOS）

规范要求在**没有安装 torch 和 transformers** 的环境里用转换后的真实权重完成一次
生成。v0.1 修复前的实际执行：

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

该 venv 里装出来的包：`numpy 2.5.4`、`tokenizers 0.23.3`、`Jinja2 3.1.6`、
`slowq 0.1.0`，加上 `huggingface_hub` / `tqdm` 等 tokenizers 的传递依赖（§13）。
**没有 torch，没有 transformers，没有 safetensors。** 快速测试套件在同一个干净
venv 里也全部通过：`372 passed, 17 skipped in 1.34s`（当时的测试数）。

> 修复之后重跑同一条 prompt，输出文本的 sha256 是
> `23c4791e233b1262a0ca53f95a75d99db0b34bbd7910ced5c7db25e61ff18319`，与当时
> 逐字相同；那次是冷启动，prefill 7228.8 ms、decode 42.82 tok/s。

> 注意 `prefill 397.5 ms` 对 19 个 token 而言异常慢——这是冷启动，权重页第一次被
> 读进内存。同一条 prompt 又跑了一次（`--max-context 1024`），输出逐字相同、
> 退出码同样是 `0`，但 prefill 变成 288.8 ms、decode 42.94 tok/s。
> 冷启动的 prefill 每次都不同；decode 在 43~47 tok/s 之间波动。
> 本轮 Qwen3-0.6B 的 decode 探针是 43.41 tok/s，与这个区间一致。

**v0.1 那一轮没有在干净 venv 里重跑端到端命令**；v0.2 发布前修复轮补上了，见 §16.2。

### 16.2 干净环境安装验证（v0.2 发布前修复轮，macOS arm64）

从仓库源码打 wheel，在**新建的 venv** 里安装并验证，全程不碰开发用的 `.venv`：

```bash
python -m pip wheel . --no-deps --no-build-isolation -w /tmp/slowq_dist
# -> slowq-0.2.0-py3-none-any.whl, 79165 bytes

python3 -m venv /tmp/slowq_clean    # 跑快速套件的那个，先只装 [dev]
/tmp/slowq_clean/bin/python -m pip install "/tmp/slowq_dist/slowq-0.2.0-py3-none-any.whl[dev]"

python3 -m venv /tmp/slowq_text     # 另建一个，始终不装 Pillow，用来跑真实离线向量命令
/tmp/slowq_text/bin/python -m pip install "/tmp/slowq_dist/slowq-0.2.0-py3-none-any.whl"
```

wheel 里 `slowq/` 下 15 个 `.py` 齐全（§2 的 14 个运行时模块加 `__main__.py`，
含新增的 `embedding.py`、`vlm.py`），
`slowq/ops.py` 中 `padding_attention_mask` 出现 **0 次** —— 打 wheel 前先移走了
`build/` 里那份陈旧拷贝，否则会把它重新打进去。

| 项目 | 结果 |
| --- | --- |
| 声明的运行时依赖 | 只有 `numpy>=1.26`、`tokenizers>=0.20`、`Jinja2>=3.1`（wheel METADATA 逐条核对） |
| 无 Pillow 的快速套件 | **660 passed, 5 skipped, 62 deselected**；5 条 skip 全是 `reading an image needs Pillow` |
| 装上 `[vision]` 后同一 venv | **665 passed, 62 deselected, 0 skipped**；`torch` / `transformers` 仍然 `find_spec` 为 `None` |
| `pip check` | 两个 venv 都是 `No broken requirements found.` |
| CLI | `slowq --version` → `slowq 0.2.0`；`generate` / `chat` / `embed` / `vlm` / `inspect` 五个子命令都在 |
| 依赖边界测试 | `tests/test_dependency_boundary.py` 在干净 venv 里 **6 passed** |
| 真实离线文本生成（无 Pillow） | SmolLM2-135M-Instruct，`generate --max-new-tokens 8`，退出码 0，`peak_rss_bytes` 598867968 |
| 真实离线向量（无 Pillow） | Qwen3-Embedding-0.6B，`dimension` 1024，相关文档 0.7006 vs 无关文档 0.0906 |
| 真实离线图文（`[vision]`） | SmolVLM-256M-Instruct，9 帧 / 576 image token，`vision_encode_seconds` 5.535，输出连贯 |

干净 venv 里的 `tokenizers` 是 **0.23.3**，开发 `.venv` 是 **0.21.4**，
所以这一轮顺带覆盖了更新的 tokenizers；`huggingface_hub` 是它的传递依赖，
不是 SlowQ 声明的（`Required-by: tokenizers`）。

图文请求的校验时机也在这个 venv 里从 CLI 侧确认了一次：
`--max-new-tokens 0` 退出码 **2**、报 `max_new_tokens must be >= 1, got 0`，
整个进程 **0.20 s**；同一张图片的合法请求要 **8.25 s**，其中 `vision_encode_seconds` 5.535 s。
差值就是没有跑的那座视觉塔。stderr 里仍会打印 `image prep 59.8 ms` ——
那是 `build_prompt` 的 Pillow 解码与 resize，**必须先做**才知道扩展后的 prompt 是 609 token，
上下文预算检查才有输入；它不是视觉编码。

### 16.3 跨平台（v0.1 修复轮，Linux x86_64）

第二平台：Alibaba Cloud Linux 3、内核 5.10.134、2× Intel Xeon Platinum 8575C
（192 逻辑核，AVX-512 + AMX）、2 TiB 内存。**全部操作在 docker 验证容器内完成，
宿主机不做任何改动**。源码用 `COPYFILE_DISABLE=1 tar` 流式送入容器，不落宿主机磁盘。

容器里有**两个环境，必须分开说**：`/opt/miniconda3`（Python 3.14.8、
numpy 2.5.4 **scipy-openblas**、tokenizers 0.21.4）不是干净环境——早先为了尝试构建
torch 参考端往里装过 `transformers` / `safetensors` / `huggingface_hub`（torch 最终
没装上）；`/work/clean-venv` 是现场 `pip install .` 建的，`import torch` /
`transformers` / `safetensors` 三个都是 `ModuleNotFoundError`。

| 项目 | 结果 |
| --- | --- |
| 快速套件 | **462 passed, 23 skipped**（miniconda 3.63 s；干净 venv 3.60 s） |
| 集成套件 | **23 passed**，19.40 s（干净 venv）；两个环境全选时都是 **485 passed** |
| NumPy 侧 vs **已固定的** macOS 参考 | **PASS: 116 tensors**；最差 rel_l2 **1.011e-05**、max_abs 1.159e-04；20/20 argmax 一致 |
| 权重转换（暂存式转换器） | 310 个张量，6.3 s；310/310 个 `.npy` 与 macOS arm64 的结果 **sha256 逐个相同** |
| 端到端生成（干净 venv） | 输出与 macOS **sha256 逐字节相同**（`23c4791e…`）；19 prompt / 31 generated、eos、退出码 0 |
| 截断的 checkpoint | 1.5 GB 截到 1.2 GB：在第 202/310 个张量处报出自己的错误，退出码 1；已有模型不受影响；无暂存目录残留 |
| `--threads=4`（等号写法） | requested 4、exported before numpy **True**、后端 scipy-openblas、有效线程数报 "not observable" |

**这一行到底证明了什么。** Linux 上做的是"只重算 NumPy 侧，与 macOS 上抓取并固定
下来的 PyTorch 参考张量对比"。它证明的是 **x86_64 + scipy-openblas 的 NumPy 实现
与那份参考一致**；它**不等于**"在 Linux 上重新执行了 PyTorch 参考实现"——后者没有做，
容器里 `import torch` 依然是 `ModuleNotFoundError`。即便如此，它补上了当时最大的
证据缺口：**换一种 BLAS 之后 rel_l2 落在哪**，实测 1.011e-05，比 macOS 的
7.986e-06 大约 1.3 倍，仍比声明门限紧约 100 倍。

**一处复现性细节：** 用同一份源码在 Linux 上重跑第二次，结论仍是 `PASS: 116`，
但两次输出有一处末位数字不同（`multi-turn.decode2.logits` 的 rel_l2 1.809e-06 vs
1.808e-06）。原因是 OpenBLAS 在 192 线程下的归约顺序不是逐次确定的。
量级远在门限之内、greedy argmax 也没变，但**不要把 Linux 上的 rel_l2 当成
逐 bit 可复现的常数**。

> **跨平台的性能对比不成立，已删除。** 本文早先版本写过"Linux decode 50.51 tok/s
> 高于 macOS"。两台机器的线程设置、BLAS、独占程度都不同（这台是共享机，
> 测量期间 `/proc/loadavg` 从 2.07 升到 16.13），那个对比支撑不了任何结论。
> macOS 与 Linux 之间**能**成立的对比只有正确性相关的：转换结果逐字节相同、
> 生成文本逐字节相同、与同一份参考的 rel_l2 同量级。

**没有完成的部分，如实记录：** 在 x86_64 上重新执行 PyTorch 参考端**没有做**。
该网络环境拿不到可用的 torch 构建——`registry-1.docker.io` 与 ACR 的 `library`
命名空间均不可达，aliyun 的 pytorch wheel 镜像没有 cp314 的包，conda 的 pytorch
通道装出的 torch 又因缺 `iJIT_NotifyEvent`（Intel ITT 符号）无法 import。
所以"torch 自己在另一种 BLAS / 指令集下会不会算出别的数"仍未验证。

**v0.2 新增的 7 个模型没有任何 Linux 记录。** 上表全部只覆盖 Qwen3-0.6B。

复现 Linux 侧只重算 NumPy 的那一步（不需要 torch）：

```bash
# 先将 SLOWQ_TEST_HOST / SLOWQ_TEST_CONTAINER 设为自己的 SSH 主机和容器名
COPYFILE_DISABLE=1 tar -cf - -C artifacts/validation/<模型目录名> ref.npz reference.json | \
  ssh "$SLOWQ_TEST_HOST" "docker exec -i $SLOWQ_TEST_CONTAINER tar -xf - -C /work/slowq/artifacts/validation/<模型目录名>"
# 容器内
python tools/validate_reference.py numpy --model /work/models/<name>-slowq
python tools/validate_reference.py compare --model /work/models/<name>-slowq
```

三个环境陷阱，留给复现的人：

* macOS 的 `tar` 默认写入 AppleDouble 元数据（`._ops.py`），它们以 `.py` 结尾，
  会被 `test_dependency_boundary.py` 的 AST 扫描当源码读并报 UnicodeDecodeError。
  传源码必须 `COPYFILE_DISABLE=1`。
* 该镜像的 conda（26.5.3）默认 libmamba solver 与镜像内老 libstdc++ 不兼容
  （`GLIBCXX_3.4.29 not found`），需要 `CONDA_SOLVER=classic`。
* 想验证"截断的分片报出自己的错误"，截断长度要选在**让 tied 权重自检也能通过**
  的位置（`lm_head.weight` 与 `model.embed_tokens.weight` 都要还在文件里），
  否则先撞上的是 §10.3 那条跨分片守卫，看不到本次要测的失败路径。
