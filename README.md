# 慢炖 · SlowQ

> 用 CPU 慢慢炖出 token。

SlowQ 是面向学习和实验的大模型推理框架，用 **Python + NumPy** 实现
Qwen3-0.6B 的前向计算与生成流程，让每一步都能读懂。

支持 KV cache、分块 prefill、temperature / top-k / top-p 采样、流式输出和多轮聊天。
运行时依赖 `numpy`、`tokenizers`、`Jinja2`，无需 PyTorch 或远程模型 API。

## 快速开始

需要 **Python 3.11+**，运行平台为 Linux x86_64 或 macOS。转换后的 FP32 权重约
**2.2 GiB**，推理还需要 KV cache 和中间张量的内存。

### 1. 安装

```bash
git clone https://github.com/luolinglong0611/SlowQ.git
cd SlowQ
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### 2. 下载并转换权重

```bash
python tools/convert_model.py \
    --repo Qwen/Qwen3-0.6B \
    --out models/Qwen3-0.6B-slowq
```

转换不需要 PyTorch。已有官方 checkpoint 时，可用
`--source /path/to/checkpoint --skip-download` 替代 `--repo`；保留 `--out`。
转换完成后，推理完全离线，权重不会随本仓库分发。

### 3. 运行

```bash
export SLOWQ_MODEL_DIR=models/Qwen3-0.6B-slowq

# 交互式聊天
python -m slowq chat

# 单次生成
python -m slowq generate --prompt "用一句话解释什么是 KV cache"

# 查看模型与内存配置
python -m slowq inspect
```

聊天支持 `/clear`、`/history`、`/help` 和 `/exit`；Ctrl-C 取消当前生成。
更多参数可通过 `python -m slowq generate --help` 查看。

## Python API

```python
from slowq import Engine

engine = Engine.from_directory("models/Qwen3-0.6B-slowq", max_context=1024)
result = engine.generate(prompt="你好，介绍一下你自己", max_new_tokens=128)
print(result.text)
```

流式输出与采样示例见 [examples/quickstart.py](examples/quickstart.py)。

## 范围与限制

- v0.1 为实验版，当前验证的模型是 **Qwen3-0.6B**，仅 CPU、FP32、batch size 1。
- 面向代码阅读和推理实验；暂不提供量化、并发服务、多模态或 tool calling。
- prompt 与预留生成长度之和超过 `max_context` 时会报错，不会自动截断。
- Windows 和大端平台暂不支持。性能取决于 CPU、BLAS、线程数及上下文长度。

## 测试与文档

快速测试不需要下载模型：

```bash
python -m pip install -e ".[dev]"
python -m pytest -m "not integration" -q
```

- [架构说明](docs/architecture.md)：模块、前向流程、张量形状和 KV cache。
- [测试与数值验证](docs/validation.md)：真实模型测试、参考数据准备与验证结果。
- [性能与内存](docs/benchmarks.md)：历史实测、测量条件与复现命令。
- [变更记录](docs/changelog.md)：v0.1 发布前的主要修复。

## 许可

代码采用 [MIT License](LICENSE)。模型权重需另行下载，遵循上游模型的许可。
