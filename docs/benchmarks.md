# 性能与内存

本文只记录**实测数据**。批次 `v0.2`，2026-10-21 在 macOS arm64 / Apple M5 Pro 上执行，
每个模型跑在**独立进程**里，运行时机器上没有其他负载。结果只代表这组硬件、软件和运行条件。

返回 [README](../README.md)；模型清单见[模型支持](models.md)。

## 测量环境

| 项目 | 配置 |
| --- | --- |
| 机器 | Apple M5 Pro，18 逻辑核，24 GiB 内存 |
| 系统 | Darwin 25.6.0，arm64 |
| Python / NumPy | 3.13.14 / 2.5.4 |
| BLAS | Accelerate |
| 线程 | 未指定 `--threads`，BLAS 线程环境变量全部未设置 |
| 精度 / batch | FP32 / 1 |
| 采样 | greedy（`temperature=0`） |

生成模型的命令统一为：

```bash
python tools/benchmark.py --model models/<name>-slowq \
  --max-context 1024 --prefill-chunk-size 128 --decode-steps 32 --repeat 3 \
  --json artifacts/benchmark/v0.2-<name>-slowq.json
```

`tools/benchmark.py` 只支持文本生成（它用 `Engine.from_directory` 加载），
向量与图文模型的数字来自各自引擎自己的计时字段，测法见[对应小节](#文本向量qwen3-embedding-06b)。

**同一批数据里两次独立运行的波动约 5~8%**（例如 21 token 的向量编码，两次批次分别是
65.8 ms 与 70.7 ms）。下面的数字是一次测量的结果，不是多次测量的下界。

## 三个内存指标不能互相替代

| 指标 | 含义 | 来源 |
| --- | --- | --- |
| 权重逻辑体积 | 全部张量按 FP32 展开的字节数，tied weights 只算一次 | `Weights.deduplicated_bytes()` |
| KV cache 预分配 | 按 `max_context` 一次性分配的缓存容量 | `KVCache.allocated_bytes()` |
| 进程峰值 RSS | 截至某阶段的历史峰值驻留内存，不是该时刻的瞬时占用 | `slowq.metrics.peak_rss_bytes()` |

权重通过只读 mmap 加载，**建立映射不等于所有权重页都已驻留**。因此"加载后的峰值 RSS"
远小于权重体积是正常的；随着前向把张量读进来，RSS 才逐步上升。三者也不能相加。

## 文本生成

### 加载、decode 与内存

decode 探针测量短 prefill 之后的 32 次逐 token 前向加 greedy 选择，
不含模型加载与文本输出。

| 模型 | 权重逻辑体积 | KV cache @1024（每 token） | 加载 冷 / 热 | 冷启动首次前向 | decode 探针 |
| --- | --- | --- | --- | --- | --- |
| SmolLM2-135M-Instruct | 513.1 MiB | 45.0 MiB（45.0 KiB） | 0.032 / 0.029 s | 1.68 s | **148.42 tok/s**（6.74 ms） |
| SmolLM2-360M-Instruct | 1380.2 MiB | 80.0 MiB（80.0 KiB） | 0.029 / 0.029 s | 5.58 s | **73.65 tok/s**（13.58 ms） |
| Qwen2.5-0.5B-Instruct | 1884.6 MiB | 24.0 MiB（24.0 KiB） | 0.121 / 0.105 s | 5.73 s | **57.80 tok/s**（17.30 ms） |
| Qwen2.5-Coder-0.5B-Instruct | 1884.6 MiB | 24.0 MiB（24.0 KiB） | 0.103 / 0.110 s | 5.65 s | **57.72 tok/s**（17.33 ms） |
| Qwen3-0.6B | 2273.8 MiB | 224.0 MiB（224.0 KiB） | 0.134 / 0.129 s | 6.94 s | **43.41 tok/s**（23.04 ms） |
| Qwen3-1.7B | 6563.5 MiB | 224.0 MiB（224.0 KiB） | 0.124 / 0.156 s | 14.48 s | **17.49 tok/s**（57.19 ms） |

| 模型 | 峰值 RSS：加载后 | 探针后 | 生成后 |
| --- | --- | --- | --- |
| SmolLM2-135M-Instruct | 69.2 MiB | 0.67 GiB | 0.68 GiB |
| SmolLM2-360M-Instruct | 69.3 MiB | 1.56 GiB | 1.60 GiB |
| Qwen2.5-0.5B-Instruct | 122.8 MiB | 2.10 GiB | 2.14 GiB |
| Qwen2.5-Coder-0.5B-Instruct | 125.2 MiB | 2.04 GiB | 2.09 GiB |
| Qwen3-0.6B | 175.7 MiB | 2.65 GiB | 2.65 GiB |
| Qwen3-1.7B | 175.5 MiB | 6.78 GiB | 6.78 GiB |

两点值得注意：

- **两个 Qwen3 的 KV cache 完全相同（224.0 KiB/token）**，尽管 hidden_size 差一倍。
  Qwen3 的 `head_dim` 是独立字段（128），不是 `hidden_size / heads`；
  两个模型都是 28 层 × 8 个 KV head × 128，所以每 token 的缓存字节数一样。
- **"加载后"的 RSS 只有一两百 MiB，而权重是 GiB 级**，这是 mmap 的结果。
  Qwen3-1.7B 的冷启动首次前向 14.48 s 里绝大部分是把 6.4 GiB 权重页读进来的时间，
  同一进程内后续前向只需几十毫秒。冷启动与稳态必须分开看。

冷启动首次前向受操作系统页缓存状态影响：SmolLM2-135M 只有 513 MiB 权重、1.68 s，
而权重更小的 SmolLM2-360M 记录到 5.58 s。这一项不适合用来横向比较模型。

### Prefill 吞吐

`full` 一次处理整块输入；`chunked@128` 每次最多 128 token 并写入同一份 KV cache。
计时覆盖前向循环，不含采样、逐 token 生成或打印。每行是一次探针测量，
不采用下面工作负载表的三次中位数；前向次数由实际调用计数得到。

单位 tok/s：

| 模型 | 32 full | 32 chunked | 128 full | 128 chunked | 512 full | 512 chunked（4 次前向） |
| --- | --- | --- | --- | --- | --- | --- |
| SmolLM2-135M-Instruct | 916.2 | 932.0 | 1105.5 | 1117.4 | 922.1 | 969.2 |
| SmolLM2-360M-Instruct | 457.7 | 459.3 | 567.4 | 571.0 | 492.3 | 552.4 |
| Qwen2.5-0.5B-Instruct | 420.3 | 425.6 | 553.7 | 551.1 | 525.6 | 526.0 |
| Qwen2.5-Coder-0.5B-Instruct | 397.0 | 407.1 | 527.8 | 533.7 | 510.9 | 522.4 |
| Qwen3-0.6B | 380.9 | 388.8 | 534.5 | 530.8 | 500.6 | 506.2 |
| Qwen3-1.7B | 160.4 | 162.1 | 242.1 | 246.5 | 255.7 | 236.8 |

32 token 时 chunked 与 full 相同（一个 chunk 就装下了）。128 token 之后分块的开销与收益
都在几个百分点以内；512 token 时只有 Qwen3-1.7B 的分块比整块慢约 7%。
**8 token 的 `full` 一行没有列进表里**：它是进程内的第一次前向，
耗时被冷启动读权重支配（1.7 ~ 14.5 s），不能与后面的行比较，也不能用两者之差判断分块收益。

### 端到端工作负载

四个工作负载，greedy，每个跑 3 次取中位数。**输入 token 数各模型不同**，
因为 tokenizer 不一样（同一段中文在 SmolLM2 下是 52 token，在 Qwen3 下是 19 token）。

| 模型 | 工作负载 | 输入 / 生成 | 首 token | prefill | decode | 总耗时 | 停止原因 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SmolLM2-135M | 中文单轮 | 52 / 64 | 53.1 ms | 982.3 tok/s | 148.24 tok/s | 0.479 s | max_new_tokens |
| SmolLM2-135M | 英文与 system | 45 / 46 | 44.3 ms | 1018.6 tok/s | 151.23 tok/s | 0.342 s | eos |
| SmolLM2-135M | 多轮对话 | 65 / 11 | 61.7 ms | 1056.4 tok/s | 150.41 tok/s | 0.128 s | eos |
| SmolLM2-135M | Emoji 与混合文字 | 63 / 64 | 57.1 ms | 1107.4 tok/s | 147.10 tok/s | 0.486 s | max_new_tokens |
| SmolLM2-360M | 中文单轮 | 52 / 64 | 100.1 ms | 520.5 tok/s | 71.15 tok/s | 0.986 s | max_new_tokens |
| SmolLM2-360M | 英文与 system | 45 / 37 | 91.0 ms | 495.3 tok/s | 73.25 tok/s | 0.583 s | eos |
| SmolLM2-360M | 多轮对话 | 65 / 11 | 126.7 ms | 513.9 tok/s | 72.17 tok/s | 0.265 s | eos |
| SmolLM2-360M | Emoji 与混合文字 | 63 / 4 | 112.4 ms | 561.1 tok/s | 73.87 tok/s | 0.153 s | eos |
| Qwen2.5-0.5B | 中文单轮 | 36 / 39 | 95.6 ms | 378.6 tok/s | 56.29 tok/s | 0.775 s | eos |
| Qwen2.5-0.5B | 英文与 system | 42 / 46 | 101.8 ms | 413.6 tok/s | 57.61 tok/s | 0.883 s | eos |
| Qwen2.5-0.5B | 多轮对话 | 62 / 4 | 128.6 ms | 482.9 tok/s | 57.97 tok/s | 0.180 s | eos |
| Qwen2.5-0.5B | Emoji 与混合文字 | 47 / 64 | 109.2 ms | 431.2 tok/s | 57.54 tok/s | 1.208 s | max_new_tokens |
| Qwen2.5-Coder | 中文单轮 | 36 / 14 | 96.1 ms | 375.6 tok/s | 55.99 tok/s | 0.332 s | eos |
| Qwen2.5-Coder | 英文与 system | 42 / 81 | 103.5 ms | 408.3 tok/s | 55.04 tok/s | 1.556 s | eos |
| Qwen2.5-Coder | 多轮对话 | 62 / 4 | 135.4 ms | 459.6 tok/s | 53.50 tok/s | 0.192 s | eos |
| Qwen2.5-Coder | Emoji 与混合文字 | 47 / 2 | 112.1 ms | 420.4 tok/s | 53.81 tok/s | 0.130 s | eos |
| Qwen3-0.6B | 中文单轮 | 19 / 13 | 67.0 ms | 284.9 tok/s | 47.16 tok/s | 0.322 s | eos |
| Qwen3-0.6B | 英文与 system | 46 / 54 | 114.2 ms | 403.8 tok/s | 44.00 tok/s | 1.319 s | eos |
| Qwen3-0.6B | 多轮对话 | 66 / 12 | 150.3 ms | 440.0 tok/s | 44.51 tok/s | 0.397 s | eos |
| Qwen3-0.6B | Emoji 与混合文字 | 30 / 21 | 80.2 ms | 375.1 tok/s | 45.13 tok/s | 0.526 s | eos |
| Qwen3-1.7B | 中文单轮 | 19 / 19 | 174.9 ms | 109.2 tok/s | 17.55 tok/s | 1.202 s | eos |
| Qwen3-1.7B | 英文与 system | 46 / 17 | 268.6 ms | 171.4 tok/s | 17.43 tok/s | 1.186 s | eos |
| Qwen3-1.7B | 多轮对话 | 66 / 11 | 360.8 ms | 183.2 tok/s | 16.89 tok/s | 0.953 s | eos |
| Qwen3-1.7B | Emoji 与混合文字 | 30 / 64 | 198.1 ms | 152.0 tok/s | 16.86 tok/s | 3.934 s | max_new_tokens |

与 v0.1 记录（批次 `m5pro-recapture`，同一台机器）相比，Qwen3-0.6B 的 decode 探针从
44.16 tok/s 变为 43.41 tok/s，差 1.7%，在批次间波动范围内；
v0.2 的多家族改动没有让 Qwen3 路径变慢。

## 文本向量（Qwen3-Embedding-0.6B）

`tools/benchmark.py` 加载不了这个任务，数字取自 `EmbeddingResult.encode_seconds`。
测法（每条文本 3 轮，先做一次预热前向）：

```python
from slowq import EmbeddingEngine
engine = EmbeddingEngine.from_directory("models/Qwen3-Embedding-0.6B-slowq")
engine.encode_document(texts[0])                     # 预热：承担 mmap 页读入
for text in texts:
    result = engine.encode_document(text)            # result.encode_seconds / num_tokens
```

| 项目 | 记录值 |
| --- | --- |
| 权重逻辑体积 | 2272.7 MiB |
| 输出维度 | 1024（`hidden_size`，不是词表 151669） |
| 加载 | 0.116 s |
| 峰值 RSS：加载后 / 编码后 | 162.8 MiB / 1858.5 MiB |
| KV cache | 按当前文本长度分配，不按 `max_length`（默认 8192）预分配 |

编码耗时（3 轮中位数，逐条编码，batch size 1，全部按 document 编码即不加指令前缀）。
四条样例就是这一批测的全部文本，token 数相同的中英文两条分列，不做平均：

| 输入 tokens | 耗时 | 吞吐 | 样例（完整文本） |
| --- | --- | --- | --- |
| 12 | 59.2 ms | 202.8 tok/s | `Which planet in the solar system has the most moons?` |
| 18 | 66.5 ms | 270.7 tok/s | `The Great Wall of China is a series of fortifications made of stone and brick.` |
| 21 | 69.7 ms | 301.4 tok/s | `Saturn has 146 confirmed moons, more than any other planet in the solar system.` |
| 21 | 71.7 ms | 293.0 tok/s | `慢炖是用低温长时间加热，让结缔组织慢慢转化为明胶的烹饪方法。` |

短文本的 tok/s 更低，因为固定开销（28 层的逐层调度、RoPE 表、归一化）在总耗时里占比更大；
比较向量模型时应该看每条文本的毫秒数，而不是 tok/s。同样 21 token，
中文那条比英文那条慢 2.0 ms，这是同一批数据里 token 数相同、内容不同的两条实测，
不是噪声估计。

检索相似度：query 是上面那条 12 token 的问句（加官方指令前缀后编码为 31 token），
document 是其余三条（不加前缀），L2 归一化后点积：

| document | 余弦 |
| --- | --- |
| `Saturn has 146 confirmed moons, more than any other planet in the solar system.` | **+0.7566** |
| `The Great Wall of China is a series of fortifications made of stone and brick.` | +0.1242 |
| `慢炖是用低温长时间加热，让结缔组织慢慢转化为明胶的烹饪方法。` | +0.0724 |

最相关的一条排第一。这三个值只取决于 query 与该 document，与同时编码多少条无关。

峰值 RSS（1858.5 MiB）低于权重逻辑体积（2272.7 MiB）。编码路径**不做全词表投影**
（由 `tests/test_integration_embedding.py` 的消融测试守住：把 `output_projection`
换成抛异常的桩，编码仍然成功），因此不会为了产出向量而遍历整张词表；
RSS 与权重体积的差额没有做逐页归因。

## 图文生成（SmolVLM-256M-Instruct）

数字取自 `GenerationMetrics`，图片是参考验证用的两张生成图（`artifacts/validation/` 下，
不入库）。命令 `--max-context 1024 --prefill-chunk-size 128 --max-new-tokens 32`，greedy，
每个样例 3 轮取中位数：

```python
from slowq.vlm import VisionEngine
engine = VisionEngine.from_directory("models/SmolVLM-256M-Instruct-slowq",
                                     max_context=1024, prefill_chunk_size=128)
prompt = engine.build_prompt(image_path, question, system=system)   # 预处理 + prompt 扩展
result = engine.generate_prompt(prompt, max_new_tokens=32)          # result.metrics
```

两个样例的 `question` / `system` 沿用参考验证抓取里的两条（`samples.json`），
**两者并不相同**，所以 prompt token 数不能只用 image token 数解释：

| 样例 | question | system |
| --- | --- | --- |
| grid-1x4 | `Describe the shapes you can see in this image.` | `You are a helpful assistant.` |
| grid-2x4 | `What colours appear in this picture?` | 无 |

| 项目 | 记录值 |
| --- | --- |
| 权重逻辑体积（文本塔 + 视觉塔 + connector） | 978.4 MiB |
| KV cache @1024 | 45.0 MiB（45.0 KiB/token） |
| 加载 | 0.038 s |
| 峰值 RSS：加载后 / 生成后 | 85.1 MiB / 1471.4 MiB |
| 图像预处理（解码 + resize + 切 crop + patchify + prompt 扩展） | 冷 40.2 ms，热 24.6 ms |

| 样例 | 原图 → resize | crop 网格 | 帧数 | image token | prompt token | 预处理 |
| --- | --- | --- | --- | --- | --- | --- |
| grid-1x4 | 300×1200 → 512×2048 | 1×4 | 5 | 320 | 360 | 24.0 ms |
| grid-2x4 | 400×800 → 1024×2048 | 2×4 | 9 | 576 | 612 | 45.9 ms |

上表的 prompt token 数已按上述 question / system 逐字复算过（360 与 612）；
换一个问题就会变，例如同一张 grid-1x4 图配 `图里有哪些形状？` 且不带 system 是 356。

| 样例 | 视觉编码 | prefill | 首 token | decode | 总耗时 |
| --- | --- | --- | --- | --- | --- |
| grid-1x4 | 2409 ms | 350 ms（1029.9 tok/s） | 2759 ms | 151.24 tok/s | 2.965 s |
| grid-2x4 | 4293 ms | 645 ms（948.8 tok/s） | 4943 ms | 140.23 tok/s | 5.167 s |

**视觉编码占首 token 延迟的 81%（grid-1x4）与 87%（grid-2x4）**，是这个任务的主要成本；
按帧摊算是 482 ms/帧 与 477 ms/帧，两个样例一致。文本 prefill 的吞吐
（约 1000 tok/s）与 SmolLM2-135M 同量级 —— 图文模型的文本塔就是 30 层 llama。

decode 从 grid-1x4 的 151.24 tok/s 降到 grid-2x4 的 140.23 tok/s，
因为上下文从 360 token 涨到 612 token。**一张图产生的 crop 数直接决定 prompt 长度**，
所以图片尺寸会影响整个请求的耗时，不只是预处理那一步。

冷启动同样明显：grid-1x4 的第一轮是 vision 3878 ms / prefill 2019 ms / 总 6100 ms，
第二、三轮稳定在 vision 约 2400 ms / prefill 约 350 ms。峰值 RSS（1471.4 MiB）
超过权重体积（978.4 MiB），差额是 pixel 张量与视觉塔的中间激活
（grid-2x4 的 pixel_values 是 `9×3×1024×2048` 之外的 9 帧 512×512，约 28 MiB，
加上 12 层 ViT 的注意力矩阵）。

## 在自己的机器上复现

先按 [README](../README.md) 安装依赖并转换模型。生成模型：

```bash
python tools/benchmark.py --model models/Qwen3-0.6B-slowq \
  --max-context 1024 --prefill-chunk-size 128 --decode-steps 32 --repeat 3 \
  --json artifacts/benchmark/local-default.json

python tools/benchmark.py --model models/Qwen3-0.6B-slowq --threads 4 --repeat 3 \
  --json artifacts/benchmark/local-threads4.json
```

向量与图文模型。下面两条命令用的是上表**逐字相同的输入**，所以能复现表里的
token 数与相似度（耗时取决于机器）：

```bash
python -m slowq embed --model models/Qwen3-Embedding-0.6B-slowq --json \
  --query "Which planet in the solar system has the most moons?" \
  --document "Saturn has 146 confirmed moons, more than any other planet in the solar system." \
  --document "The Great Wall of China is a series of fortifications made of stone and brick." \
  --document "慢炖是用低温长时间加热，让结缔组织慢慢转化为明胶的烹饪方法。"

python -m slowq vlm --model models/SmolVLM-256M-Instruct-slowq --json \
  --image artifacts/validation/SmolVLM-256M-Instruct-slowq/vision-grid-1x4.png \
  --question "Describe the shapes you can see in this image." \
  --system "You are a helpful assistant." \
  --max-context 1024 --prefill-chunk-size 128 --max-new-tokens 32
```

第二条命令里的图片是参考验证抓取时生成的，不入库；没有 `artifacts/` 的机器要先跑一次
`tools/validate_reference.py numpy --model models/SmolVLM-256M-Instruct-slowq`
把两张 PNG 生成出来，或者换成自己的图（那样 prompt token 数会与上表不同）。
grid-2x4 一行换 `vision-grid-2x4.png`、`--question "What colours appear in this picture?"`
且不带 `--system`。

两条命令的 JSON 里分别有 `encode_seconds` 与完整的 `metrics`（含 `vision_encode_seconds`）。

第二条 benchmark 命令用于观察同一台机器的线程设置影响，不代表推荐线程数。
脚本报告请求的线程设置和检测到的 BLAS 后端；**有效线程数标为不可观测**。
比较时应固定模型、上下文、输入、线程设置和机器负载，并区分冷启动与后续运行。
一个进程里连续测多个模型会互相污染：本轮数据是每模型一个进程重测得到的，
第一次批量跑（六个模型串在同一个 shell 循环里、同时还有集成测试在跑）记录的
Qwen3-1.7B 512-token prefill 是 4262 ms / 8004 ms，单独重测是 2002 ms / 2162 ms。
共享机器的负载会带来额外波动，不同机器和 BLAS 的数字不能直接用于性能排名。

参数与计时实现见 [benchmark.py](../tools/benchmark.py)，张量和缓存设计见
[架构说明](architecture.md)。这些记录没有包含与其他推理框架的同条件对照测试。
