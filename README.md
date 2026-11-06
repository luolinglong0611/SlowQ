# 慢炖 · SlowQ

> 用 CPU 慢慢炖出 token。

SlowQ 是面向学习和实验的大模型推理框架，用 **Python + NumPy** 实现前向计算与生成流程，
让每一步都能读懂。当前版本 **0.3.0**，支持 **8 个模型、3 类任务**：文本生成、文本向量、图文生成。

支持 KV cache、分块 prefill、temperature / top-k / top-p 采样、流式输出和多轮聊天。
运行时依赖只有 `numpy`、`tokenizers`、`Jinja2`，无需 PyTorch 或远程模型 API。

## 模型概览

下表性能数字来自 **v0.2 接入批次（2026-10-21）**，用于说明模型规模，
不是 v0.3.0 的性能承诺。后续加速的测量条件、收益与未确认项见
[推理加速第 1 轮实测](docs/推理加速-第1轮-2026-10-23.md)。

| 模型 | 任务 | 架构家族 | FP32 权重 | 实测（Apple M5 Pro） |
| --- | --- | --- | --- | --- |
| SmolLM2-135M-Instruct | 文本生成 | llama | 513 MiB | 148.4 tok/s decode |
| SmolVLM-256M-Instruct | 图文生成 | idefics3 | 978 MiB | 2.76 s 首 token（含视觉编码） |
| SmolLM2-360M-Instruct | 文本生成 | llama | 1380 MiB | 73.7 tok/s decode |
| Qwen2.5-0.5B-Instruct | 文本生成 | qwen2 | 1885 MiB | 57.8 tok/s decode |
| Qwen2.5-Coder-0.5B-Instruct | 文本生成 | qwen2 | 1885 MiB | 57.7 tok/s decode |
| Qwen3-0.6B | 文本生成 | qwen3 | 2274 MiB | 43.4 tok/s decode |
| Qwen3-Embedding-0.6B | 文本向量 | qwen3 | 2273 MiB | 59 ms / 12 token 文本 |
| Qwen3-1.7B | 文本生成 | qwen3 | 6564 MiB | 17.5 tok/s decode |

八个模型都在 macOS arm64 上完成了转换、离线加载、官方参考数值比较、真实权重功能测试
与断网 CLI 检查，参考比较全部 `PASS`。逐模型状态、来源 commit 与实测最差误差见
[模型支持](docs/models.md)；性能与内存的完整数据见[性能与内存](docs/benchmarks.md)。

**支持范围就是这 8 个 checkpoint**，不包括同家族的其他尺寸与 base 变体。

## 快速开始

需要 **Python 3.11+**，在 Linux 或 macOS 上运行。转换后的 FP32 权重按上表体积占用磁盘，
推理时还需要 KV cache 与中间张量的内存（Qwen3-1.7B 生成时峰值 RSS 约 6.8 GiB）。

### 1. 安装

```bash
git clone https://github.com/luolinglong0611/SlowQ.git
cd SlowQ
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

图文生成还需要 `python -m pip install -e ".[vision]"`（Pillow）；
纯文本安装不需要它，`import slowq.vlm` 本身也不会导入它。

### 2. 下载并转换权重

```bash
python tools/convert_model.py \
    --repo Qwen/Qwen3-0.6B \
    --out models/Qwen3-0.6B-slowq
```

转换不需要 PyTorch，任务类型由目录里的文件证据自动识别（也可用 `--task` 指定）。
已有官方 checkpoint 时，用 `--source /path/to/checkpoint --skip-download` 替代 `--repo`。
转换先在暂存目录完成并自检再安装，默认拒绝覆盖非空目录。
转换完成后推理完全离线，权重不随本仓库分发。

### 3. 运行

```bash
export SLOWQ_MODEL_DIR=models/Qwen3-0.6B-slowq   # 或者每条命令都带 --model

# 文本生成
python -m slowq generate --prompt "用一句话解释什么是 KV cache"
python -m slowq chat                      # 交互式多轮，支持 /clear /history /help /exit

# 文本向量（需要向量模型的目录）
python -m slowq embed --model models/Qwen3-Embedding-0.6B-slowq \
    --query "太阳系里哪颗行星的卫星最多？" \
    --document "土星已确认 146 颗卫星，是太阳系中最多的行星。"

# 图文生成（需要 slowq[vision]）
python -m slowq vlm --model models/SmolVLM-256M-Instruct-slowq \
    --image picture.png --question "图里有哪些形状？"

# 三种任务通用：查看 config、权重存储与 KV cache 尺寸
python -m slowq inspect
```

用错入口会明确报错并指出正确的命令（例如把向量目录交给 `generate`）。
`generate` / `embed` / `vlm` / `inspect` 都支持 `--json`（交互式的 `chat` 没有这个开关）；
更多参数见 `python -m slowq <子命令> --help`。

## Python API

三个任务对应三个入口，都用 `from_directory(...)` 加载：

```python
from slowq import Engine, EmbeddingEngine
from slowq.vlm import VisionEngine          # 需要 slowq[vision]

engine = Engine.from_directory("models/Qwen3-0.6B-slowq", max_context=1024)
print(engine.generate(prompt="你好，介绍一下你自己", max_new_tokens=128).text)

embedder = EmbeddingEngine.from_directory("models/Qwen3-Embedding-0.6B-slowq")
batch, ranking = embedder.search(
    "太阳系里哪颗行星的卫星最多？",
    ["土星已确认 146 颗卫星，是太阳系中最多的行星。", "水的沸点在海平面是 100 摄氏度。"],
)
print(batch.results[0].vector.shape, ranking)   # (1024,) [(0, 0.813…), (1, 0.1816…)]

vision = VisionEngine.from_directory("models/SmolVLM-256M-Instruct-slowq", max_context=1024)
prompt = vision.build_prompt("picture.png", "图里有哪些形状？")
print(vision.generate_prompt(prompt, max_new_tokens=32).text)
```

两处复用接口，都是**显式**的：不调用就还是原来那条每次都从头算的路径。

```python
# 多轮对话：每轮只 prefill 与上一轮不同的那段 token（纯文本）
session = engine.session()
history = [{"role": "user", "content": "什么是 KV cache？"}]
first = session.generate(history, max_new_tokens=64)
history += [{"role": "assistant", "content": first.text},
            {"role": "user", "content": "那它占多少内存？"}]
second = session.generate(history, max_new_tokens=64)
print(second.metrics.reused_tokens, second.metrics.prefill_tokens)   # 复用的 / 这次真 prefill 的
session.reset()                                  # 丢掉前缀，保留 cache 分配

# 同一批文档问多次：先编码一次，之后每个 query 只编码自己
index = embedder.index_documents(corpus)
batch, ranking = index.search("第一个问题")
batch, ranking = index.search("第二个问题")

# 同一张图问多次：视觉塔只跑一次（VisionEngine 默认就带这个缓存）
print(vision.image_features.stats())             # hits / misses / evictions / bytes
vision = VisionEngine.from_directory(model_dir, image_cache_bytes=0)   # 关掉
```

**换掉 `vision.model` 时缓存会自己作废，不需要手动清。** 每次查缓存之前都比两样：
模型**对象**本身，以及 `_vision_identity()` 的五元组（权重 manifest 指纹 +
视觉配置 + 预处理配置 + `image_token_id` + `scale_factor`）。任一样变了就整表作废，
下一次请求重算一次、之后重新命中。**同权重、不同配置也算换模型**——
指纹描述的是 manifest，所以在同一份权重上改 `layer_norm_eps` 或 `image_mean`
重建出来的模型指纹完全相同，只比指纹会把上一个模型的特征交给新模型。
绑一个不能编码图片的对象（文本模型）会抛 `TypeError` 并点名支持的是
`SmolVLMModel`，**不会**静默返回旧特征。这次检查实测 **0.0326–0.0331 ms**，
只读 manifest、不读张量。

`slowq chat` 默认走会话（`--no-reuse` 关掉），每轮打印复用了多少 token。

流式输出与采样示例见 [examples/quickstart.py](examples/quickstart.py)。

## 范围与限制

- v0.3.0 为实验版：仅 CPU、FP32、batch size 1。
- 面向代码阅读和推理实验；不提供量化、GPU 后端、并发服务、tool calling、训练与微调。
- 图文生成一次请求只接受一张图，第二张会报错；一张图仍会按官方规则切成多个 crop。
- 向量任务只提供向量输出、余弦相似度与一个**进程内、不落盘**的预编码索引
  （`index_documents`），没有持久化向量库、没有 ANN、没有 RAG 管道。
- `Engine.session()` 只支持纯文本；图文请求会报错，因为缓存的前缀不记录自己来自哪张图。
  会话自己分配一份满尺寸 KV cache，`max_context=2048` 的 Qwen3-0.6B 上约 448 MiB。
- prompt 与预留生成长度之和超过 `max_context`、或文本超过 `max_length` 时会报错，不会自动截断。
- Windows 和大端平台暂不支持。性能取决于 CPU、BLAS、线程数及上下文长度。
- 8 个模型的真实权重与参考数值验证记录来自 macOS arm64；Linux x86_64
  仍只有 v0.1 的 Qwen3-0.6B 记录。macOS/Linux 的自动化快速测试使用微型模型，
  不代表这 8 个 checkpoint 已在两种平台完成真实权重验证。

## 测试与文档

快速测试不需要下载模型，也不需要 PyTorch：

```bash
python -m pip install -e ".[dev]"
python -m pytest -m "not integration" -q
```

完整覆盖图像读取还需安装 `python -m pip install -e ".[dev,vision]"`；
未安装 Pillow 或缺少本地历史性能证据时，相应用例会明确跳过。
[CI 工作流](.github/workflows/ci.yml) 在 macOS/Linux 上执行快速测试，
并检查 wheel 的纯文本安装、视觉安装、命令行入口、版本一致性与依赖边界。
当前验证范围、执行命令与按批次区分的历史结果见[测试与数值验证](docs/validation.md)。

真实权重的集成测试用 `SLOWQ_MODEL_DIR=<转换后的目录> python -m pytest -m integration -q`。

- [模型支持](docs/models.md)：8 个模型的验证状态、来源版本、架构事实与复现命令。
- [架构说明](docs/architecture.md)：模块、前向流程、张量形状、KV cache 与三个任务的实现。
- [测试与数值验证](docs/validation.md)：测试分层、参考验证协议、容差与实测结果。
- [性能与内存](docs/benchmarks.md)：实测数据、测量条件与复现命令。
- [变更记录](docs/changelog.md)：v0.3.0、v0.2.0 与 v0.1.0。
- [v0.2 多模型接入报告](docs/v0.2-多模型接入报告.md)：那一轮的设计决策、踩坑记录与未完成项。
- [推理加速第 1 轮实测](docs/推理加速-第1轮-2026-10-23.md)：本轮五项改动省掉了什么、
  怎么验证正确性、A/B 实测数字与适用条件。

## 许可

代码采用 [MIT License](LICENSE)。模型权重需另行下载，遵循上游模型的许可。
