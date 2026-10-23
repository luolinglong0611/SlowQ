# 模型支持

SlowQ v0.2 支持 **8 个模型**、**3 类任务**：文本生成、文本向量、图文生成。
本文记录每个模型的验证状态、来源版本、架构事实和复现命令。

返回 [README](../README.md)；实现见[架构说明](architecture.md)；
验证方法见[测试与数值验证](validation.md)；本轮改动见 [v0.2 接入报告](v0.2-多模型接入报告.md)。

## 支持矩阵

| 模型 | 任务 | 架构家族 | 转换 | 加载 | 参考数值 | 功能 | 离线 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 文本生成 | qwen3 | ✅ | ✅ | ✅ PASS 116 | ✅ 27 | ✅ |
| Qwen3-1.7B | 文本生成 | qwen3 | ✅ | ✅ | ✅ PASS 116 | ✅ 26 | ✅ |
| Qwen2.5-0.5B-Instruct | 文本生成 | qwen2 | ✅ | ✅ | ✅ PASS 116 | ✅ 26 | ✅ |
| Qwen2.5-Coder-0.5B-Instruct | 文本生成 | qwen2 | ✅ | ✅ | ✅ PASS 116 | ✅ 26 | ✅ |
| SmolLM2-135M-Instruct | 文本生成 | llama | ✅ | ✅ | ✅ PASS 116 | ✅ 26 | ✅ |
| SmolLM2-360M-Instruct | 文本生成 | llama | ✅ | ✅ | ✅ PASS 116 | ✅ 26 | ✅ |
| Qwen3-Embedding-0.6B | 文本向量 | qwen3 | ✅ | ✅ | ✅ PASS 82 | ✅ 25 | ✅ |
| SmolVLM-256M-Instruct | 图文生成 | idefics3 | ✅ | ✅ | ✅ PASS 82 | ✅ 26 | ✅ |

每一列的含义，以及它是**怎么跑出来的**：

| 列 | 判据 |
| --- | --- |
| 转换 | `tools/convert_model.py` 从官方 checkpoint 转出目录，manifest 记录 source / revision / 文件 sha256 / 张量清单 |
| 加载 | 断网状态下 `from_directory` 成功，config 与 manifest 交叉校验通过 |
| 参考数值 | `tools/validate_reference.py compare` 打印 `PASS: N tensors compared within the declared tolerances` |
| 功能 | `SLOWQ_MODEL_DIR=models/<name> pytest -m integration` 的 **passed** 数（每模型收集 62 项，其余按任务过滤跳过） |
| 离线 | 测试中阻断 socket 后，`slowq inspect --json` 与对应任务命令返回 0 |

**"功能"列的 62 项收集数对每个目录都相同**，但通过数不同，这是设计而非缺漏：
`test_integration_real_model.py` / `test_integration_embedding.py` /
`test_integration_vlm.py` 三个模块各自声明任务，manifest 里记录的任务不匹配时整模块跳过。
Qwen3-0.6B 多出的 1 项是 v0.1 遗留的固定输出回归，它显式检查 checkpoint 来源是不是
`Qwen/Qwen3-0.6B`，其他模型跳过并打印原因。

**上表全部数字来自 2026-10-20 ~ 10-21 在 macOS arm64 / Apple M5 Pro 上的实际执行**，
命令见[复现](#复现命令)。同一批代码在 Linux 上只跑过 v0.1 的 Qwen3-0.6B NumPy 侧，
v0.2 新增的 7 个模型没有 Linux 验证记录。

## 来源与版本

| 模型 | 官方 repo | commit revision | manifest | 张量 | 权重逻辑体积 | 目录体积 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | `local:models/Qwen3-0.6B-hf` | `c1899de289a04d12100db370d81485cdf75e47ca` | v1 | 310 | 2273.8 MiB | 2289.0 MiB |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | v2 | 310 | 6563.5 MiB | 6578.7 MiB |
| Qwen2.5-0.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | v2 | 290 | 1884.6 MiB | 1895.6 MiB |
| Qwen2.5-Coder-0.5B-Instruct | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | `ea3f2471cf1b1f0db85067f1ef93848e38e88c25` | v2 | 290 | 1884.6 MiB | 1895.6 MiB |
| SmolLM2-135M-Instruct | `HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` | v2 | 272 | 513.1 MiB | 516.4 MiB |
| SmolLM2-360M-Instruct | `HuggingFaceTB/SmolLM2-360M-Instruct` | `a10cc1512eabd3dde888204e902eca88bddb4951` | v2 | 290 | 1380.2 MiB | 1383.6 MiB |
| Qwen3-Embedding-0.6B | `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | v2 | 310 | 2272.7 MiB | 2288.0 MiB |
| SmolVLM-256M-Instruct | `HuggingFaceTB/SmolVLM-256M-Instruct` | `7e3e67edbbed1bf9888184d9df282b700a323964` | v2 | 471 | 978.4 MiB | 983.2 MiB |

除 Qwen3-0.6B 外，其余 7 个 revision 都是转换当时由转换器自己解析并写入 manifest 的，
没有套用 v0.1 为 Qwen3-0.6B 固定的那个 commit。Qwen3-0.6B 一行是 v0.1 已经转好的目录，
`repo_id` 带 `local:` 前缀表示它读的是本地 checkpoint 而不是下载快照；这个目录**未被重新转换**，
manifest 仍是 `format_version: 1`、没有 `task` 与 `model_family` 两个键，
用来验证 v0.2 读旧格式的行为（见下节）。

参考实现版本对 8 个模型一致：`transformers 4.51.3`、`torch 2.14.1`、
`attn_implementation="eager"`、`dtype=float32`、`device=cpu`，线程 4（SmolVLM 为 8）。

### manifest format_version

| 版本 | 写入者 | 是否带 `task` / `model_family` | v0.2 行为 |
| --- | --- | --- | --- |
| 1 | 只有 SlowQ v0.1 | 否 | 可读；`recorded_task()` 返回 `text-generation` |
| 2 | SlowQ v0.2 | 是，两者都必须存在 | 可读，也是唯一会写出的版本 |

版本 1 的默认不是"猜"出来的：v0.1 只转换过 Qwen3 causal LM，所以缺键时的唯一可能取值就是
文本生成。版本 2 缺任一键会被拒绝，其他版本号也会被拒绝，`Weights` 不会静默降级。
v0.1 目录因此**照常可用**，`slowq generate` / `chat` / `inspect` 的行为与 v0.1 一致。

## 架构家族事实

`family` 与 `task` 是两条独立的轴。family 由 `config.json` 的 `model_type` 决定，
描述"这些权重按什么规则计算"；task 由 manifest 记录，描述"这份目录拿来做什么"。
下面的表是 `src/slowq/config.py` 里 `TEXT_FAMILIES` 的内容，依据是 transformers 4.51.3
的官方 modelling 代码，不是模型名字：

| 家族 | Q/K/V bias | O 投影 bias | MLP bias | Q/K RMSNorm | head_dim | 尊重 config 的 sliding window |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3 | `config.attention_bias` | `config.attention_bias` | 恒关 | 有 | **必须显式给出** | 是 |
| qwen2 | 恒开 | 恒关 | 恒关 | 无 | 缺失时按 `hidden_size / heads` 推导 | 是 |
| llama | `config.attention_bias` | `config.attention_bias` | `config.mlp_bias` | 无 | 缺失时按 `hidden_size / heads` 推导 | 否 |
| idefics3 | 文本塔同 llama；视觉塔另有一套 | 同左 | 同左 | 文本塔无；视觉塔用 LayerNorm | 文本塔显式 64 | 否 |

qwen2 的三条 bias 规则是**硬编码在 `Qwen2Attention.__init__` 里的**，config 里没有对应字段，
所以不能写成 `config:attention_bias`；写错会让 Qwen2.5 少算 Q/K/V bias 或多算 O 投影 bias。
llama 的 `architectures` 里包含 `VLlama3ForCausalLM` —— 那是 SmolVLM 给内嵌文本 config 起的名字，
`Idefics3Model` 通过 `AutoModel.from_config` 按 `model_type` 分派，根本不看 `architectures`。

### head_dim 为什么不能统一成 `hidden_size / heads`

Qwen3-0.6B：`hidden_size=1024`、`num_attention_heads=16` → `1024/16 = 64`，
但 config 显式写着 `head_dim=128`。Qwen3-1.7B：`hidden_size=2048`、16 heads → `2048/16 = 128`，
`head_dim` 同样是 128。也就是说 0.6B 的 Q/K/V 投影输出维度是 `16×128 = 2048`，
**比 hidden_size 大一倍**，O 投影再把 2048 压回 1024。用 `hidden_size / heads` 推导会直接算错形状。

因此 qwen3 家族要求 `head_dim` 必须显式存在（`head_dim_must_be_explicit=True`），
缺失时报 `ConfigError` 而不是回退到推导；qwen2 与 llama 的官方 checkpoint 确实不写这个字段，
推导后才校验 `hidden_size % num_attention_heads == 0`。

两个 Qwen3 的 KV cache 都是 **224.0 KiB/token**（28 层 × 2 × 8 KV heads × 128 × 4 B），
尽管 hidden_size 差一倍 —— 这是 head_dim 独立于 hidden_size 的直接后果。

### sliding window 的判定

两个 Qwen2.5-0.5B checkpoint 的 config 里都有 `sliding_window: 32768`，
同时 `use_sliding_window: false`。SlowQ 按**实际启用状态**判断：`use_sliding_window`
为假时跑全注意力，既不因为字段存在就拒绝加载，也不会把真正启用的滑动窗口当成全注意力算。
qwen3 的两个 checkpoint 里 `sliding_window` 是 `null`。llama 家族的官方实现不读这个字段，
所以 `honours_config_sliding_window=False`。

## 各模型接入要点

### Qwen3-1.7B

复用 qwen3 家族，没有新增代码路径。它是 6.4 GiB 权重、2 个 safetensors 分片，
用来验证跨分片读取与 tied weights 在更大模型上仍然正确（`tie_word_embeddings: true`，
`lm_head.weight` 与 `model.embed_tokens.weight` 共享同一份数据）。

### Qwen2.5-0.5B-Instruct 与 Coder-0.5B-Instruct

两者 config 完全同构（896 hidden / 24 层 / 14 heads / 2 KV heads / 词表 151936），
差别只在权重和 chat template。接入时踩到的两点：Q/K/V 有 bias 而 O 投影没有（见上表），
以及 `sliding_window` 字段存在但未启用。`head_dim` 缺失，推导为 `896/14 = 64`。
两个模型分别独立验证，没有用其中一个的结果代替另一个。

### SmolLM2-135M-Instruct 与 360M-Instruct

llama 家族。没有 Q/K RMSNorm，没有 Qwen2 式的 QKV bias（`attention_bias: false`、
`mlp_bias: false`，两者都从 config 读），`rms_norm_eps` 是 `1e-5` 而不是 Qwen 的 `1e-6`，
`rope_theta` 是 `100000` 而不是 `1000000`，词表 49152。目录里多一个
`special_tokens_map.json`。head_dim 缺失，推导为 `576/9 = 64` 与 `960/15 = 64`。
先接 135M，再验证 360M；**两个模型通过数值测试不等于支持所有 llama 模型** ——
支持范围就是这两个 checkpoint 加上 `TEXT_FAMILIES` 里声明的规则。

### Qwen3-Embedding-0.6B

新任务 `text-embedding`，入口是 `EmbeddingEngine` 与 `slowq embed`。它的
`model_type` 是 `qwen3`、`architectures` 是 `["Qwen3ForCausalLM"]`，
**这两个字段与聊天模型取值完全一致**，所以任务只能从 manifest 读，不能从 config 猜。
整个 `config.json` 并不相同（向量模型 `vocab_size` 151669、`eos_token_id` 151643、
`max_position_embeddings` 32768；聊天模型 151936 / 151645 / 40960），
但这些差异没有一项能可靠地指示任务。
转换器靠 sentence-transformers 导出的三个标记文件（`modules.json`、
`config_sentence_transformers.json`、`1_Pooling/config.json`）识别它；
三个文件只出现一部分时报错而不是猜。

编码路径与官方 snippet 一致：

| 环节 | SlowQ 的做法 |
| --- | --- |
| query 前缀 | `f"Instruct: {instruction}\nQuery:{query}"`，`Query:` 后**无空格**；document 不加前缀 |
| 特殊 token | `add_special_tokens=True`，末尾追加 id 151643 |
| attention | 官方 Qwen3 的因果注意力，**不是**双向 |
| pooling | 最后一个有效 token 的 final hidden state，**不是**平均池化 |
| 归一化 | L2 归一化，输出单位向量 |
| 输出维度 | 官方全维度 1024（`hidden_size`），不是词表维度 151669 |
| lm_head | **不做**全词表投影，也不做自回归生成 |

`EmbeddingEngine.from_directory` 会拒绝文本生成目录，`Engine.from_directory` 会拒绝向量目录，
两边都指向正确的入口。KV cache 按当前文本长度分配，不按 `max_length` 预分配。

### SmolVLM-256M-Instruct

新任务 `image-text-generation`，入口是 `slowq.vlm.VisionEngine` 与 `slowq vlm`。
这是**真正的图像条件生成**，不是拿里面的文本塔跑纯文本：

| 环节 | 事实 |
| --- | --- |
| 顶层 config | `model_type: idefics3`、`architectures: ["Idefics3ForConditionalGeneration"]`、`image_token_id: 49190`、`scale_factor: 4` |
| 文本塔 | 30 层 llama（576 hidden / 9 heads / 3 KV heads / head_dim 64 / 词表 49280 / `tie_word_embeddings: false`） |
| 视觉塔 | 12 层 ViT，768 hidden / 12 heads / patch 16 / image_size 512 / `hidden_act: gelu_pytorch_tanh` / `layer_norm_eps: 1e-6` |
| 视觉塔归一化 | LayerNorm（weight + bias），**不是**文本塔的 RMSNorm |
| 视觉塔注意力 | 全连接**非因果**，无 RoPE，位置来自 `position_embedding.weight` |
| 预处理 | RGB → resize（最长边 2048，再向上取整到 512 的倍数）→ rescale `1/255` → normalize（mean/std 均 0.5）→ 切成 512×512 crop |
| 裁剪 | `do_image_splitting: true`，rows×cols 个 crop **加 1 个全局帧** |
| 每帧 token 数 | `image_seq_len = 64`（`(512/16)² / scale_factor² = 1024/16`，pixel shuffle 后） |
| 占位符排布 | 每 64 个 `<image>` 之间插入 `<row_r_col_c>` 标签，最后一段是全局帧 |
| connector | pixel shuffle（scale_factor 4）→ `modality_projection.proj`，输出维度 12288 → 576 |

一张图会产生官方要求的**多个 crop**，"只支持一张图"不等于可以跳过它们：
验证用的 1200×300 图被切成 1×4 共 5 帧、320 个 image token、prompt 360 token；
800×400 图切成 2×4 共 9 帧、576 个 image token、prompt 612 token。

图像读不到、占位符数量对不上、扩展后超过 `max_context`，都报错并说明原因，
不会静默截断或忽略图像。视觉分支不复用文本塔的 RMSNorm 与 SwiGLU。

## 任务入口

```bash
# 文本生成（qwen3 / qwen2 / llama）
slowq generate --model models/Qwen3-0.6B-slowq --prompt "用一句话解释什么是 KV cache"
slowq chat     --model models/SmolLM2-135M-Instruct-slowq

# 文本向量（Qwen3-Embedding）
slowq embed --model models/Qwen3-Embedding-0.6B-slowq \
    --query "太阳系里哪颗行星的卫星最多？" \
    --document "土星已确认 146 颗卫星，是太阳系中最多的行星。" \
    --document "水的沸点在海平面是 100 摄氏度。"

# 图文生成（SmolVLM，需要 slowq[vision]）
slowq vlm --model models/SmolVLM-256M-Instruct-slowq \
    --image /path/to/picture.png --question "图里有哪些形状？"

# 三种任务通用：查看 config、权重存储与 KV cache 尺寸
slowq inspect --model models/SmolVLM-256M-Instruct-slowq
```

Python API 对应三个入口：`Engine`、`EmbeddingEngine`、`slowq.vlm.VisionEngine`，
都用 `from_directory(...)` 加载。`slowq.vlm` 需要单独 import，因为它是唯一可能用到
Pillow 的部分；纯文本安装不会被强制装上视觉依赖。

## 参考数值验证结果

`tools/validate_reference.py compare` 的输出。门限：logits 与 hidden states
`rel_l2 ≤ 1e-3`、`max_abs ≤ 5e-2`，向量 `rel_l2 ≤ 1e-3`、`max_abs ≤ 1e-3`，
选定权重摘要必须逐字节相同。下表是**实测最差值**，不是门限：

| 模型 | 比较项 | 最差 rel_l2（张量） | 最差 max_abs（张量） | argmax 一致 |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 116 | 7.986e-06（`zh-single-turn.full_prefill.logits`） | 9.155e-04（`multi-turn.chunked_prefill.layer27.layer_output[-1]`） | 20/20 |
| Qwen3-1.7B | 116 | 1.595e-05（`multi-turn.full_prefill.layer27.layer_output[-1]`） | 7.202e-03（`multi-turn.decode0.layer27.layer_output[-1]`） | 20/20 |
| Qwen2.5-0.5B-Instruct | 116 | 2.719e-05（`emoji-and-mixed-script.decode2.logits`） | 1.114e-03（`emoji-and-mixed-script.chunked_prefill.final_norm[-1]`） | 20/20 |
| Qwen2.5-Coder-0.5B-Instruct | 116 | 1.067e-05（`zh-single-turn.decode0.layer0.layer_output[-1]`） | 2.480e-04（`emoji-and-mixed-script.chunked_prefill.final_norm[-1]`） | 20/20 |
| SmolLM2-135M-Instruct | 116 | 2.536e-06（`multi-turn.decode2.logits`） | 3.906e-03（`emoji-and-mixed-script.full_prefill.layer15.layer_output[0]`） | 20/20 |
| SmolLM2-360M-Instruct | 116 | 1.195e-05（`emoji-and-mixed-script.full_prefill.layer31.layer_output[0]`） | 1.562e-02（同上） | 20/20 |
| Qwen3-Embedding-0.6B | 82 | 4.935e-06（`retrieval-without-instruction.doc2.chunked_prefill.layer27.layer_output[-1]`） | 7.629e-04（同上） | 无 logits |
| SmolVLM-256M-Instruct | 82 | 8.587e-06（`grid-1x4.vision.encoder_output[4]`） | 4.395e-03（`grid-1x4.full_prefill.layer29.layer_output[0]`） | 10/10 |

按上表逐模型算比值（各自适用的门限：hidden/logits `max_abs 5e-2`、`rel_l2 1e-3`）：
**最差 rel_l2 是门限的 1/37 ~ 1/394，最差 max_abs 是门限的 1/3 ~ 1/202**。
全部 860 个张量里，rel_l2 最好的一个是门限的 1/131096，max_abs 最好的是 1/209732。
门限自 v0.1 起没有为任何失败样例放宽过。

最差值落在哪里，两个指标的分布不同：**最差 max_abs 八个模型全部落在深层 hidden states 上**；
最差 rel_l2 有 3 个模型落在 logits 上（Qwen2.5-0.5B、Qwen3-0.6B、SmolLM2-135M），
SmolVLM 落在**视觉塔输出**上，其余 4 个落在 hidden states 上。
SmolVLM 的 connector 输出最差 rel_l2 是 3.529e-06。

抽样层按家族和层数选择，不再固定在第 27 层：

| 模型 | 抽样层 | 视觉抽样层 | prefill chunk | decode 步数 |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B / 1.7B / Embedding | 0, 14, 27 | — | 8 | 3（Embedding 为 0） |
| Qwen2.5-0.5B / Coder | 0, 12, 23 | — | 8 | 3 |
| SmolLM2-135M | 0, 15, 29 | — | 8 | 3 |
| SmolLM2-360M | 0, 16, 31 | — | 8 | 3 |
| SmolVLM-256M | 0, 15, 29 | 0, 6, 11 | 100 | 3 |

"缺失的参数"与"合法的架构默认值"在比较前分开处理：`ARCHITECTURE_FIELDS` 里的每一项都由参考侧
用官方 modelling 代码自己的表达式从原始 `config.json` 复述一遍，两边一致才开始比数值，
所以架构层面的分歧不会被数值门限掩盖。

## 复现命令

```bash
# 1. 快速测试：不需要网络，不下载权重
python -m pytest -m "not integration" -q

# 2. 转换（每个模型一次；已有目录时不要重复转换，也不要覆盖用户已有模型）
python tools/convert_model.py --repo HuggingFaceTB/SmolLM2-135M-Instruct \
    --out models/SmolLM2-135M-Instruct-slowq

# 3. 真实模型集成测试：manifest 里的任务决定哪些模块运行
SLOWQ_MODEL_DIR=models/Qwen3-Embedding-0.6B-slowq python -m pytest -m integration -q

# 4. 参考数值验证：三个独立进程，按 numpy → ref → compare 顺序
M=models/SmolVLM-256M-Instruct-slowq
H=models/SmolVLM-256M-Instruct-slowq-hf/HuggingFaceTB--SmolVLM-256M-Instruct/7e3e67edbbed1bf9888184d9df282b700a323964
python tools/validate_reference.py numpy   --model "$M" --max-context 1024 --prefill-chunk-size 100
python tools/validate_reference.py ref     --model "$M" --hf-dir "$H" --threads 8
python tools/validate_reference.py compare --model "$M"
```

三条命令都要 `--model`：`--artifacts` 缺省时由它推导出
`artifacts/validation/<模型目录名>`，所以验证第二个模型不会覆盖第一个的抓取。
文本生成模型用默认参数即可（`--max-context 512 --prefill-chunk-size 8 --decode-steps 3`）；
SmolVLM 的 prompt 展开后是 612 token，必须放大 `--max-context`，`--prefill-chunk-size 100`
与上表记录的抓取配置一致。

第 4 步的 `ref` 子命令需要 `pip install -e ".[ref]"`（PyTorch + `transformers==4.51.3`）；
运行时、转换器和 `numpy` / `compare` 子命令都不需要。SmolVLM 还需要
`pip install -e ".[vision]"`（Pillow）才能读图。

## 已验证范围与未验证范围

已验证：上表 8 个 checkpoint，在 macOS arm64 / Apple M5 Pro 上完成转换、离线加载、
参考数值比较、功能集成测试与断网 CLI 检查。

未验证，因此**不能当作已支持**：

- 这 8 个之外的任何模型，包括同家族的其他尺寸（例如 Qwen2.5-1.5B、SmolLM2-1.7B、Llama-3.2 系列）。
- Base / 非 Instruct 变体。本轮只接了 Instruct 模型。
- Linux 上的 v0.2 新模型（v0.1 的 Qwen3-0.6B 有 Linux 的 NumPy 侧记录，见[验证文档](validation.md)）。
- 量化、GPU 后端、batch size > 1、并发服务、tool calling、训练与微调。
- SmolVLM 的多图输入：一次请求只接受一张图，第二张会报错。
- 向量检索的下游系统：只提供向量输出与余弦相似度示例，没有向量库或 RAG。
