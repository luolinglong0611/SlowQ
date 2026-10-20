# 测试与参考验证

本文说明如何验证 SlowQ v0.1。所有命令从项目根目录执行；安装与模型下载需要网络，
准备好依赖和本地模型后，测试与参考比较可以离线运行。

返回 [README](../README.md)；实现细节见 [架构说明](architecture.md)。

## 快速测试

需要 Python 3.11 或更新版本，建议使用独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -m "not integration"
```

快速测试使用临时构造的微型模型和 tokenizer，不下载真实权重，覆盖算子、KV cache、
采样、流式输出、转换器和验证工具。`.[dev]` 增加 pytest，不安装 PyTorch。

## 准备真实模型

`models/` 和 `artifacts/` 均被 Git 忽略，**模型权重与参考 artifact 不随仓库分发**。
完整验证需要官方 checkpoint、转换后的 FP32 权重和一次参考抓取。
转换后权重约 2.2 GiB；还需保留官方 checkpoint，并为推理预留数 GiB 可用内存。

已按 README 快速开始完成转换时，只设置以下路径，然后跳到下一节；不要重复转换。
如果使用过自定义下载路径或 revision，请相应调整 `SLOWQ_HF_DIR`：

```bash
SLOWQ_REV=c1899de289a04d12100db370d81485cdf75e47ca
SLOWQ_MODEL_DIR=models/Qwen3-0.6B-slowq
SLOWQ_HF_DIR="models/Qwen3-0.6B-slowq-hf/Qwen--Qwen3-0.6B/$SLOWQ_REV"
```

尚未准备模型时，改用下面的首次准备命令。它使用独立验证目录，固定历史验证的 revision：

```bash
SLOWQ_REV=c1899de289a04d12100db370d81485cdf75e47ca
SLOWQ_MODEL_DIR=models/Qwen3-0.6B-validation
python tools/convert_model.py \
    --repo Qwen/Qwen3-0.6B --revision "$SLOWQ_REV" \
    --download-dir models/hf-cache --out "$SLOWQ_MODEL_DIR"
SLOWQ_HF_DIR="models/hf-cache/Qwen--Qwen3-0.6B/$SLOWQ_REV"
```

在同一个 shell 中继续执行后续命令。两条路径必须对应同一份权重。
转换器使用标准库和 NumPy，不需要 PyTorch，默认拒绝覆盖非空输出目录。

## 首次建立参考

仅 `ref` 子命令需要额外的参考实现依赖：

```bash
python -m pip install -e ".[ref]"
python tools/validate_reference.py numpy \
    --model "$SLOWQ_MODEL_DIR" \
    --max-context 512 --prefill-chunk-size 8 --decode-steps 3 \
    --artifacts artifacts/validation
python tools/validate_reference.py ref \
    --hf-dir "$SLOWQ_HF_DIR" --threads 4 \
    --artifacts artifacts/validation
python tools/validate_reference.py compare \
    --artifacts artifacts/validation
```

按 `numpy → ref → compare` 顺序执行。`ref` 会读取 NumPy 侧记录的输入与解码路径，
使用 CPU、FP32、`eval()`、eager attention 和 batch size 1 计算参考结果。
三个独立进程通过文件交换数据，避免两套模型同时常驻内存。

`.[ref]` 安装 PyTorch、`transformers==4.51.3` 等参考依赖。运行时、转换器以及
`numpy` / `compare` 子命令不需要 PyTorch 或 Transformers。安装器仍会安装运行时包的
传递依赖；完整声明见 [pyproject.toml](../pyproject.toml)。

抓取会更新所选目录中的四个文件：

| 文件 | 内容 |
| --- | --- |
| `samples.json` | 当前 SlowQ 的 tokenizer 输出、权重摘要、抓取配置与来源 |
| `slowq.npz` | 当前 SlowQ 的 logits 和 hidden states |
| `reference.json` | 参考实现的 tokenizer 输出、权重摘要、配置与环境 |
| `ref.npz` | 参考实现的 logits 和 hidden states |

保留已有基准时先备份该目录，或在三个命令中统一另选 `--artifacts` 路径。
NPZ 文件各约 11 MiB。以后检查运行时代码改动，可保留参考侧文件，只重跑 `numpy` 和
`compare`；模型、抓取配置或协议发生变化时，按工具提示重新抓取参考。

## 真实模型集成测试

```bash
SLOWQ_MODEL_DIR="$SLOWQ_MODEL_DIR" python -m pytest -m integration
```

集成测试覆盖真实模型上的确定性、流式输出、上下文边界、分块 prefill、cached decode，
以及当前 NumPy 实现与固定参考的比较。测试会在临时目录重算当前 tokenizer 输出与权重摘要。

未设置 `SLOWQ_MODEL_DIR` 时，真实模型测试会跳过。参考回归固定读取项目内的
`artifacts/validation/ref.npz` 与 `reference.json`；缺少其中任一文件时，相关测试会跳过。
因此请检查 pytest 的 skipped 列表，不能把部分通过当作完整集成验证。
独立运行 `compare` 时，缺少必要文件会失败并返回非零退出码。

## 比较内容与判定

默认抓取配置为 chunk size 8、3 步 cached decode，与集成参考回归的配置一致：

| 检查 | 覆盖 |
| --- | --- |
| tokenizer | 4 个样例的渲染文本与 token IDs |
| 权重 | 6 个选定张量的完整 FP32 字节摘要；不是全部 310 个张量的逐项对齐 |
| 数值 | 116 项：full/chunked prefill 与 decode 的 logits、第 0/13/27 层及最终归一化输出 |
| greedy 输出 | 20 个 logits 比较位置的 argmax |
| 来源与结构 | checkpoint 摘要、架构、抓取配置与摘要协议 |

logits 和 hidden states 的门限为 `rel_l2 ≤ 1e-3`、`max_abs ≤ 5e-2`；
选定权重的摘要必须相同。内部的分块/缓存等价性集成测试使用更紧的
`rel_l2 < 1e-5`、`max_abs < 1e-3` 门限。

比较器检查预期数值项、tokenizer 样例与权重摘要是否齐全，并拒绝形状、dtype、有限性
或协议不匹配的数据。具体实现见 [验证工具](../tools/validate_reference.py)，
故障注入用例见 [验证工具测试](../tests/test_validate_reference.py)。

## 历史验证记录

下表是已记录的历史结果，**不是本次文档整理重新运行的测试**：

| 批次 / 平台 | 快速测试 | 真实模型集成 | 合计 |
| --- | --- | --- | --- |
| 2026-10-19 发布收尾，macOS arm64 / Apple M5 Pro | 480 | 23 | 503 |
| 上一轮修复，Linux x86_64 / Xeon Platinum 8575C | 462 | 23 | 485 |

macOS 参考抓取使用 PyTorch 2.14.1、Transformers 4.51.3、FP32 eager attention。
记录中的 116 项数值比较、4 个 tokenizer 样例、6 个权重摘要与 20 个 logits argmax 均通过；
最差 logits `rel_l2=7.986e-06`、`max_abs=1.063e-04`。

Linux 重算的是 **SlowQ NumPy 侧**，比较对象为 macOS 抓取的固定参考，116 项通过，
最差 `rel_l2=1.011e-05`、`max_abs=1.159e-04`。PyTorch 参考端没有在 Linux 重跑；
Linux 的历史测试数也不表示后续新增测试在那里执行过。

历史数字用于说明已验证的范围；当前改动是否通过，应以本地执行上述命令的结果为准。
