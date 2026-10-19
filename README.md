# 慢炖 · SlowQ

> 用 CPU 慢慢炖出 token。

Qwen3-0.6B 的**完整**推理实现：纯 Python + NumPy，约 2800 行可读代码。
不依赖 PyTorch、Transformers、ONNX Runtime、llama.cpp，也不调用任何远端模型 API。

从 safetensors 权重转换、chat template 渲染、byte-level BPE 增量解码、RoPE、
GQA attention、KV cache、分块 prefill、采样，到流式输出和多轮对话，全部自己写。

**优先级：计算正确 > 完整可用 > 代码可读 > 性能优化。**

---

## 这不是什么

- **不是**一个有竞争力的推理引擎。同一台机器上 llama.cpp 会快一个数量级。
  这里每一个矩阵乘法都写成能被读懂的形式，没有算子融合、没有量化、没有 SIMD
  intrinsic。
- **不是**一个通用框架。只支持 Qwen3，只支持 FP32，只支持 CPU，batch size 恒为 1。
- **不是**为了"能跑出中文"。正确性由与 Transformers 参考实现的逐张量数值对齐
  来保证，容差在测量之前就声明好（见下文）。

它为了一个目的存在：**把一次 LLM 推理的每一步都摊开在 2800 行能读完的代码里。**

---

## 快速开始

### 1. 安装

需要 Python ≥ 3.11。运行时依赖只有三个：`numpy`、`tokenizers`、`Jinja2`。

从 GitHub 获取源码后安装（本项目尚未发布到 PyPI）：

```bash
git clone https://github.com/luolinglong0611/SlowQ.git
cd SlowQ
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # 运行时
pip install -e ".[dev]"     # 加上 pytest，跑测试用
```

### 2. 下载并转换权重

转换**不需要 torch**，也不需要 `safetensors` / `huggingface_hub`：它自己解析
safetensors 头部、用 `mmap` 取字节，下载走标准库 `urllib`。上面装好的运行时依赖
就够了。它把官方 BF16 safetensors 变成每个张量一个 `.npy` 文件 + 一份
`manifest.json`：

```bash
python tools/convert_model.py \
    --repo Qwen/Qwen3-0.6B \
    --out models/Qwen3-0.6B-slowq
```

已经下载过官方 checkpoint 的话，用 `--source` 指向那个目录并加 `--skip-download`，
全程不联网：

```bash
python tools/convert_model.py \
    --source models/Qwen3-0.6B-hf --skip-download \
    --out models/Qwen3-0.6B-slowq
```

产出约 2.2 GiB（0.6B 参数 × FP32，tied 的 `lm_head` 只存一份）。
`models/` 已在 `.gitignore` 里，权重不会进版本库。

> **转换之后的推理完全离线。** `Engine.from_directory` 不会访问网络，
> 集成测试里有一个把 `socket.create_connection` 换成抛异常桩的用例守着这一点。

### 3. 聊天

```bash
export SLOWQ_MODEL_DIR=models/Qwen3-0.6B-slowq
python -m slowq chat
```

```
你> 用一句话解释什么是 KV cache
SlowQ> KV cache 是一种在自回归解码中缓存键值对的技术……
```

`/clear` 清空历史，`/history` 查看历史，`/help` 帮助，`/exit` 或 Ctrl-D 退出。
生成过程中按 Ctrl-C 只取消当前这一轮，回到提示符。

### 4. 单次生成

回答走 stdout，统计和诊断走 stderr，所以可以直接重定向：

```bash
python -m slowq generate --prompt "你好，介绍一下你自己" > answer.txt
python -m slowq generate --prompt "写一首五言绝句" --json
python -m slowq generate --prompt "..." --temperature 0.7 --top-k 20 --top-p 0.8 --seed 42
```

退出码：`0` 成功，`1` `SlowQError`（模型损坏、上下文溢出、采样参数非法），
`2` 用法错误，`130` 被中断。

### 5. 查看模型信息

```bash
python -m slowq inspect
python -m slowq inspect --json
```

输出模型配置、权重 dtype、去重后的权重字节数、KV cache 容量、停止 token 集合、
BLAS 与线程设置，以及加载后的峰值 RSS。

---

## Python API

```python
from slowq import Engine, SamplingParams

engine = Engine.from_directory("models/Qwen3-0.6B-slowq", max_context=1024)

result = engine.generate([{"role": "user", "content": "你好"}], max_new_tokens=128)
print(result.text)
print(result.metrics.decode_tokens_per_second)

for step in engine.stream(prompt="继续讲下去"):
    print(step.text_delta, end="", flush=True)
```

`examples/quickstart.py` 是一份能直接跑的完整示例：

```bash
python examples/quickstart.py --model models/Qwen3-0.6B-slowq
```

---

## 实测结果

以下全部是**实际测量值**，不是估计，也没有性能承诺。

### 环境

| | |
| --- | --- |
| 机器 | Apple M5 Pro，18 逻辑核，Darwin 25.6.0 arm64 |
| Python | 3.13.14 |
| NumPy | 2.5.4（BLAS: Accelerate） |
| 线程设置 | 未显式设置（`--threads` 未给，5 个 BLAS 线程环境变量均为 `None`） |
| 模型 | Qwen3-0.6B，revision `c1899de289a0`，FP32，`max_context=1024`，`prefill_chunk_size=128` |
| 原始数据 | `artifacts/benchmark/m5pro-recapture.json`（每个工作负载 3 次） |

复现：

```bash
python tools/benchmark.py --model models/Qwen3-0.6B-slowq
```

报告里的 `[threads]` 段把**请求的线程设置**和**探测到的 BLAS 后端**分开列出，
并且明确写出"有效线程数不可观测"。请求值不会被当成测量值呈现。

### 性能（greedy，3 次取中位数）

| 工作负载 | prompt tokens | 生成 tokens | 首 token 延迟 | prefill | decode |
| --- | --- | --- | --- | --- | --- |
| zh-short | 19 | 13 (eos) | 66.6 ms | 286.5 tok/s | **46.54 tok/s** |
| en-with-system | 46 | 54 (eos) | 110.8 ms | 416.1 tok/s | **44.73 tok/s** |
| multi-turn | 66 | 12 (eos) | 146.2 ms | 452.5 tok/s | **44.94 tok/s** |
| emoji-mixed-script | 30 | 21 (eos) | 79.7 ms | 377.7 tok/s | **45.11 tok/s** |

微观探针（与端到端分开测，便于归因）。**每个长度测两种模式**，因为它们是两条
不同的代码路径：`full` 是一次覆盖整块的前向，`chunked` 是引擎真正跑的分块循环。
"前向次数"是**实测调用次数**（用包装器数的），不是 `长度 // chunk_size` 推出来的：

| prefill 长度 | 模式 | 耗时 | 吞吐 | 实测前向次数 |
| --- | --- | --- | --- | --- |
| 8 tokens | full | 6832.7 ms | 1.2 tok/s ← **冷启动，从磁盘读入 2.2 GiB** | 1 |
| 8 tokens | chunked@128 | 55.3 ms | 144.7 tok/s | 1 |
| 32 tokens | full | 81.7 ms | 391.8 tok/s | 1 |
| 32 tokens | chunked@128 | 80.7 ms | 396.4 tok/s | 1 |
| 128 tokens | full | 236.4 ms | 541.4 tok/s | 1 |
| 128 tokens | chunked@128 | 229.1 ms | 558.6 tok/s | 1 |
| 512 tokens | full | 992.0 ms | 516.1 tok/s | 1 |
| 512 tokens | chunked@128 | 1024.6 ms | 499.7 tok/s | **4** |

decode 单独测：**44.16 tok/s（22.64 ms/token）**。

计时区间里只有前向：采样、decode、打印都不在 prefill 计时内。

> **更正（v0.1 缺陷 #6）。** 本文早先版本把 `512 tokens / 1005.9 ms / 509.0 tok/s`
> 标注为"4 个 chunk"。那次测量走的是**单次完整前向**，分块数是按
> `512 // 128` 推算后写上去的，代码并没有执行那 4 次前向。1005.9 ms 作为
> **单次完整 prefill** 的历史数据仍然成立（本次重测同一模式是 992.0 ms），
> 但它不是分块结果。真正跑 4 次前向的分块路径，本次实测是 **1024.6 ms /
> 499.7 tok/s**。

> **第一行的 6832.7 ms 不是 prefill 变慢了，是权重第一次被读进内存。**
> 权重是 mmap 的，`Engine.from_directory` 只建立映射（123 ms），真正的缺页
> 中断由第一个前向承担。冷启动的代价完全取决于页缓存里还有没有这 2.2 GiB：
> 紧接着再跑一次是 **536.3 ms**，另一次 CLI 冷启动（19 token prompt）是
> **7228.8 ms**，写这份文档时又测到一次 **7168.6 ms**（同一条 prompt，输出
> sha256 仍是 `23c4791e…`）。所以冷启动 prefill 的实测区间是 0.2 s ～ 7 s，
> 报单一数字没有意义。
> 这就是为什么 `load_seconds` 和 `prefill_seconds` 必须分开报告——把 page-in
> 算进加载时间会低估首 token 延迟，算进 prefill 吞吐会低估稳态性能。

### 内存

| | |
| --- | --- |
| 权重逻辑大小 | 2.2 GiB |
| `.npy` 文件总大小 | 2.2 GiB |
| KV cache（`max_context=1024`） | 224.0 MiB（224.0 KiB/token） |
| **峰值 RSS：刚映射完权重** | **184.6 MiB** |
| **峰值 RSS：跑完探针** | **2.5 GiB** |
| **峰值 RSS：跑完生成** | **2.6 GiB** |

映射后与生成后的 RSS 差 14 倍，正是 mmap 的语义：没被读过的页不常驻。
**把 mmap 数组的 `nbytes` 相加冒充 RSS 是错的**，所以三个数字分开报告。


### 与 Transformers 参考实现的对齐

参考端固定为 `transformers==4.51.3` + `torch==2.14.1`、`torch_dtype=float32`、
`eval()`、`attn_implementation="eager"`、CPU、batch size 1，与 SlowQ 用
**完全相同的 token id**。

**容差在测量之前声明**（`tools/validate_reference.py` 的 `TOLERANCES`）：

| 量 | rel_l2 | max_abs |
| --- | --- | --- |
| logits | 1e-3 | 5e-2 |
| hidden states | 1e-3 | 5e-2 |
| weights | 0.0（必须逐 bit 相同） | 0.0 |

**实测结果：116 个张量全部通过。**

116 不是"两边文件里恰好有多少个共同 key"，而是**由抓取配置推导出的应有数量**
（4 个样例 × {full prefill, chunked prefill, 3 步 decode} × {logits, 第 0/13/27 层
hidden, final_norm} × 各自的位置）。少一个就报错，多一个也报错，一个都没有更是
直接失败——`compare` 不会因为"没东西可比"而给 PASS。

| 检查项 | 结果 |
| --- | --- |
| tokenizer：渲染文本 + token id vs `AutoTokenizer` | 4/4 样例逐字符、逐 id 相同 |
| weights：抽样 6 个张量 vs 官方 BF16 的 FP32 上采 | 6/6 **逐 bit 相同**（sha256 相同，覆盖整个张量） |
| provenance：参考端读的 `model.safetensors` vs manifest 声称的 | sha256 `f47f71177f32bcd1…` 两侧相同 |
| full prefill logits + 第 0/13/27 层 hidden states | 通过 |
| chunked prefill（chunk=8） | 通过 |
| 3 步 cached decode | 通过 |
| greedy argmax（全部 20 个比较位置） | 20/20 一致 |

最差的一项：`zh-single-turn.full_prefill.logits`，rel_l2 **7.986e-06**，
max_abs **1.063e-04**——比声明的门限紧约 100 倍。hidden states 更紧。

`compare` 只比对**这一轮自己算出来的数据**，不采信任何写死的结论。权重检查存的是
sha256 摘要而不是整份张量，摘要规范（`slowq-weight-digest-v1`）随结果一起记录：
算法、preimage 的构造顺序、dtype（`<f4`）、字节序（little-endian）、以及
**覆盖整份张量、不做步长抽样、不截断**。所以逐 bit 门限依然覆盖每一个元素，
交换文件却从 300 MiB 降到 11 MiB。

复现：

```bash
python tools/validate_reference.py numpy   --model models/Qwen3-0.6B-slowq
python tools/validate_reference.py ref     --hf-dir  models/Qwen3-0.6B-hf
python tools/validate_reference.py compare
```

三条里只有 `ref` 需要 PyTorch/Transformers 参考端，也就是 `pip install -e ".[ref]"`
（`torch` / `transformers==4.51.3` / `safetensors` / `huggingface_hub`）；`numpy` 和
`compare` 只用运行时依赖。注意 `artifacts/` 在 `.gitignore` 里，**参考数据不随仓库
分发**：本机的 `artifacts/validation/` 已经抓好，所以这里复跑 `numpy` + `compare`
就够；换一台机器要自己跑一次 `ref`（需要 `.[ref]` 和官方 HF checkpoint），否则
集成测试会 skip 而不是假装通过。

两套实现在**独立进程**中运行，通过 `artifacts/validation/` 交换数据，
这样 2.4 GiB 的 torch FP32 模型和 NumPy 模型不会同时常驻。NumPy 侧写
`samples.json` + `slowq.npz`，参考侧写 `reference.json` + `ref.npz`，各约 11 MiB。

`compare` 还会先检查**这份参考是不是在描述这一轮**：artifact 格式版本、样例名、
抓取层、抓取位置、chunk 大小、decode 步数、架构指纹、摘要规范。任一项对不上就
要求重新抓取，而不是拿旧结论继续比。比较之前先查形状（展平之前）、dtype、
以及两侧是否有限（NaN/Inf 直接判失败）；`nan > 容差` 恒为 False，旧版比较器正是
因此把 NaN 判成了通过。

### 内部一致性

分块和缓存只改变 FP32 归约顺序，所以门限比上面的跨实现门限更紧
（rel_l2 1e-5，max_abs 1e-3）。在真实 Qwen3-0.6B 上实测（21 token prompt，
chunk ∈ {1, 3, 7, 16}，实测前向次数 21 / 7 / 3 / 2）：

| 比较 | rel_l2 | max_abs | argmax |
| --- | --- | --- | --- |
| 分块 prefill vs 一次性 prefill（logits） | 最差 1.636e-06（chunk=7） | 5.627e-05 | 一致 |
| 分块 prefill vs 一次性 prefill（hidden） | 最差 2.282e-06（chunk=1） | 6.580e-05 | — |
| 3 步 cached decode vs 一次长 prefill | 1.274e-06 | 4.482e-05 | 一致 |


### 干净运行环境验证

在一个**没有安装 torch 和 transformers** 的全新 venv 里，用转换后的真实权重
完成一次生成：

```bash
python3 -m venv /tmp/slowq-clean
/tmp/slowq-clean/bin/pip install .
/tmp/slowq-clean/bin/python -c "import torch"    # ModuleNotFoundError ✓

/tmp/slowq-clean/bin/slowq generate \
    --model models/Qwen3-0.6B-slowq \
    --prompt "用一句话解释什么是 KV cache。" --max-new-tokens 64 --max-context 512
```

实际输出，退出码 `0`：

```
KV cache 是一种用于缓存键值对（Key-Value Pair）的内存结构，用于减少重复访问和提高数据访问效率。

loaded in 0.10 s (weights 2.2 GiB logical, KV 112.0 MiB allocated for 512 tokens)
stop reason              eos
prompt tokens            19 (1 chunk(s))
generated tokens         31
prefill                  397.5 ms
decode speed             46.64 tok/s
peak RSS                 2.4 GiB
```

该 venv 只装了 `numpy 2.5.4`、`tokenizers 0.23.3`、`Jinja2 3.1.6`、`slowq 0.1.0`
（外加 tokenizers 自己的传递依赖）。快速测试套件在同一个 venv 里也全绿：
`372 passed, 17 skipped in 1.34s`。

> 上面这一段是 v0.1 **修复前**在 macOS 上的记录，保留原样（含当时的测试数）。
> 修复之后重跑同一条 prompt，输出文本的 sha256 是
> `23c4791e233b1262a0ca53f95a75d99db0b34bbd7910ced5c7db25e61ff18319`，与当时逐字相同；
> 那次是冷启动，prefill 7228.8 ms、decode 42.82 tok/s。

> 那次 `prefill 397.5 ms` 对 19 个 token 而言慢得离谱，因为它是**冷启动**：
> 权重页第一次被读进内存。同一条 prompt 又跑了一次（`--max-context 1024`），
> 输出逐字相同、退出码同样是 `0`，但 prefill 变成 288.8 ms、decode 42.94 tok/s
> ——冷启动的 prefill 每次都不同，decode 在 43–47 tok/s 之间波动，与
> `tools/benchmark.py` 在开发 venv 里测到的 44–47 tok/s 一致。

### 跨平台验证（Linux x86_64）

> **本节是上一轮（v0.1 修复轮，2026-10-18）的测量批次，原样保留。** 本轮发布收尾
> **没有连接这台机器**，所以这里的测试数（462 / 23 / 485）对应的是上一轮的套件
> 版本，不是现在的 480 / 23 / 503；性能数字也仍是那一次测的，本轮**没有重跑基准**。
> 本轮在 macOS 上实测的部分见"测试"一节与 `docs/architecture.md` §11–§12。

宿主机是 Alibaba Cloud Linux 3、内核 5.10.134、2× Intel Xeon Platinum 8575C
（192 逻辑核，AVX-512 + AMX）、2 TiB 内存；**全程在 docker 验证容器
里操作，宿主机零改动**（复核后确认宿主机 home 目录下没有任何 slowq 相关文件）。

容器里有两个环境，必须分开说：

- `/opt/miniconda3`（Python 3.14.8，numpy 2.5.4 + scipy-openblas，tokenizers 0.21.4）
  ——**不是干净环境**：早先为了尝试构建 torch 参考端，往里装过 `transformers` /
  `safetensors` / `huggingface_hub`（torch 最终没装上）。本文早先版本把这一行写成
  "容器内本就无系统级 torch"，那是**不准确的**，特此更正。
- `/work/clean-venv`（`pip install .` 现场建的）——**干净环境**：只有
  numpy 2.5.4、tokenizers 0.23.3、Jinja2 3.1.6、MarkupSafe 3.0.4、slowq 0.1.0，
  以及 tokenizers 自己声明的 `huggingface_hub 2.2.0`。torch / transformers /
  safetensors 三个都 `ModuleNotFoundError`。

| 项目 | 结果 |
| --- | --- |
| 快速套件 | **462 passed, 23 skipped**（miniconda 环境 3.63 s；干净 venv 3.60 s） |
| 集成套件 | **23 passed**，19.40 s（干净 venv；`SLOWQ_MODEL_DIR` 指向真实模型）；两个环境把全部测试一并选中时都是 **485 passed**（干净 venv 23.06 s / miniconda 22.35 s） |
| NumPy 侧 vs **已固定的** macOS 参考 | **PASS: 116 tensors**；最差 rel_l2 **1.011e-05**（`multi-turn.decode0.logits`）、max_abs 1.159e-04；20/20 argmax 一致。用同一份源码重跑第二次仍是 `PASS: 116`，两次输出只差一处末位数字（`multi-turn.decode2.logits` 的 rel_l2 1.809e-06 vs 1.808e-06）——OpenBLAS 在 192 线程下的归约顺序不是逐次确定的，量级远在门限之内，但**别把 Linux 上的 rel_l2 当成逐 bit 可复现的常数** |
| 权重摘要 | 6/6 与官方 BF16 的 FP32 上采 sha256 相同；provenance `model.safetensors` = `f47f71177f32bcd1…`，与参考端读到的同一份 |
| 权重转换（新的暂存式转换器） | 310 个张量，6.3 s；310/310 个 `.npy` 与 macOS arm64 的转换结果 **sha256 逐个相同** |
| 覆盖已有模型的守卫 | 非空 `--out` 且未给 `--overwrite` 时拒绝，退出码 1，目标目录一个字节都没动 |
| 截断的 checkpoint | 1.5 GB 的 `model.safetensors` 截到 1.2 GB：在第 202/310 个张量处报错，指出张量名、字节区间、分片名和文件实际长度，退出码 1；已有模型 manifest sha256 不变、310 个张量齐全、仍能生成；无暂存目录残留 |
| 端到端生成（干净 venv） | 输出与 macOS **sha256 逐字节相同**（`23c4791e…`，两个平台各自重跑过一次，仍然相同）；19 prompt / 31 generated，eos，退出码 0；decode 43.4–49.8 tok/s（共享机，波动大） |
| 依赖边界 | 干净 venv 里 `tests/test_dependency_boundary.py` 4 passed |
| `--threads=4`（等号写法） | requested 4、exported before numpy **True**、5 个环境变量全部为 4、探测到的后端单独列为 scipy-openblas 0.3.34.106.0、有效线程数报 "not observable" |

上表的测试数与耗时来自**同一次运行**（那一轮把当时的最新源码重新送进容器后跑的，
干净 venv 里的 `slowq` 包也重新同步过并核对了 13 个 `.py` 的 sha256）。
`artifacts/linux-recheck/` 里是原始输出，其中 `linux-compare.txt` 是第一次跑、
`linux-compare-rerun.txt` 是重跑那一次。

**这一行的含义要说清楚，不能含糊：** Linux 上做的是"**只重算 NumPy 侧**，与
macOS 上抓取并固定下来的 PyTorch 参考张量对比"。它证明的是
**x86_64 + scipy-openblas 的 NumPy 实现与那份参考一致**；
它**不等于**"在 Linux 上重新执行了 PyTorch 参考实现"——后者没有做，原因见下。
容器里 `import torch` 依然是 `ModuleNotFoundError`。

这也是**上一轮**修复补上的最大证据缺口：**"换一种 BLAS 之后 rel_l2 落在哪"现在有实测值了**，
是 1.011e-05，比 macOS 的 7.986e-06 大约 1.3 倍，仍然比声明的 1e-3 门限紧约 100 倍。
之前这个数只能靠推测。

性能（`tools/benchmark.py`，`--repeat 2`，原始数据见 `artifacts/linux-recheck/`）：

| 设置 | decode | 512-token full prefill | 512-token chunked@128（实测 4 次前向） |
| --- | --- | --- | --- |
| 不设线程（BLAS 默认） | **49.69 tok/s**（20.12 ms/token） | 3322.7 ms | 2130.4 ms |
| `--threads 4` | 15.50 tok/s（64.53 ms/token） | 2774.1 ms | 2487.5 ms |

两行的差别就是缺陷 #9 的意义：线程数被真正设进去了，所以测出来的东西变了
（49.69 → 15.50 tok/s）。旧代码里 `--threads=4` 被 argparse 接受但从未导出，
报告却照样打印线程信息，看起来像是生效了。

> **跨平台的性能对比不成立，已删除。** 本文早先版本写过"Linux decode 50.51 tok/s
> 高于 macOS，方向与内存带宽 roofline 推算一致"。两台机器的线程设置、BLAS、
> 独占程度都不同（这台是共享机，测量期间 `/proc/loadavg` 从 2.07 升到 16.13），
> 那个对比支撑不了任何结论。上面只列同一台机器、同一份代码、不同线程设置的实测值。
> macOS 与 Linux 之间**能**成立的对比只有正确性相关的：转换结果逐字节相同、
> 生成文本逐字节相同、与同一份参考的 rel_l2 都在 1e-5 量级。

**没有完成的部分，如实记录：** 在 x86_64 上重新执行 PyTorch 参考端**没有做**。
该网络环境拿不到可用的 torch 构建——dockerhub 与 ACR 的 library 命名空间都拉不到
python 镜像，aliyun 的 pytorch wheel 镜像没有 cp314 的包，conda 的 pytorch 通道装出来
的 torch 又因缺 `iJIT_NotifyEvent`（Intel ITT）符号而无法 import。所以
"torch 自己在另一种 BLAS/指令集下会不会算出别的数"仍未验证；已验证的是
SlowQ 这一侧换了 BLAS 之后仍然与那份固定的参考一致。

> 复现提示：把源码传进容器时要加 `COPYFILE_DISABLE=1 tar ...`。macOS 的 tar 默认会
> 带上 AppleDouble 元数据文件（`._ops.py` 这类），它们以 `.py` 结尾，会被
> `test_dependency_boundary.py` 的 AST 扫描当成源码读而报 UnicodeDecodeError。
>
> Linux 侧只重算 NumPy 侧的完整命令：
>
> ```bash
> # 先将 SLOWQ_TEST_HOST / SLOWQ_TEST_CONTAINER 设为自己的 SSH 主机和容器名
> # 把 macOS 上抓取并固定的两个文件送进容器
> COPYFILE_DISABLE=1 tar -cf - -C artifacts/validation ref.npz reference.json | \
>   ssh "$SLOWQ_TEST_HOST" "docker exec -i $SLOWQ_TEST_CONTAINER tar -xf - -C /work/slowq/artifacts/validation"
> # 容器内：只跑 numpy 侧，再与固定参考比较（不需要 torch）
> python tools/validate_reference.py numpy --model /work/models/Qwen3-0.6B-slowq \
>     --artifacts /work/validation-linux --max-context 512 \
>     --prefill-chunk-size 8 --decode-steps 3
> python tools/validate_reference.py compare --artifacts /work/validation-linux
> ```


---

## 测试

```bash
python -m pytest                                    # 默认：快速测试，约 3.9 秒
SLOWQ_MODEL_DIR=models/Qwen3-0.6B-slowq \
    python -m pytest -m integration                 # 真实模型，14–26 秒
```

**默认 CI 只跑快速测试**，真实模型测试必须显式启用。

快速测试不下载任何权重：`tests/conftest.py` 现场构造一个约 40 KiB 的微型 Qwen3
目录（`manifest.json` + 每层一个 `.npy`）和一个**真正训练过**的 byte-level BPE
tokenizer。它刻意保留了 Qwen3 的别扭之处而不是简化掉：

- `head_dim * num_attention_heads != hidden_size`（8 × 4 = 32 ≠ 16），
  任何从 `hidden_size` 推导 `head_dim` 的代码在这里都会失败；
- `num_attention_heads / num_key_value_heads == 2`，与 Qwen3-0.6B 相同的 GQA 分组；
- `tie_word_embeddings = true`，`lm_head.weight` 只作为别名存在；
- norm 权重是 `N(1, 0.02)` 而非全 1，忽略 norm 权重的 bug 藏不住。

| 文件 | 测试数 | 覆盖 |
| --- | --- | --- |
| `test_ops.py` | 33 | RMSNorm、RoPE 半分割配对、稳定 softmax、GQA vs 独立写出的 `repeat_kv` 参考、causal mask 绝对位置、SiLU 不溢出 |
| `test_cache.py` | 28 | 预分配、`commit` 每前向一次、绝对位置写入、容量抛错而非截断、失败写入不污染 cache、请求隔离 |
| `test_config.py` | 47 | 派生维度、`head_dim` 必须读取而非推导、拒绝不支持的配置（sliding window / rope_scaling / attention_bias） |
| `test_weights.py` | 30 | 只读 mmap、别名共享存储、shape/dtype 校验、**拒绝 pickle `.npy`** |
| `test_model.py` | 44 | 分块 prefill 等价、cached decode 等价、位置偏移、残差结构（`layer_output == embed + attention + mlp`）、**`DebugProbe` 在非末位置捕获的是该位置而不是末位** |
| `test_sampling.py` | 43 | greedy、temperature/top-k/top-p 顺序与边界、固定 seed 复现、不碰全局 RNG、6000 次抽样分布检查 |
| `test_tokenizer.py` | 56 | chat template、`enable_thinking` 分支、中文/emoji 增量解码、**证明朴素逐 token 解码是错的** |
| `test_engine.py` | 55 | 停止条件、流式、上下文容量、多轮、中途中断不污染下一个请求、**流式 `text_delta` 拼接 == 阻塞式 `result.text`（EOS 与 max_new_tokens 两条路径）** |
| `test_cli.py` | 40 | stdout/stderr 分流、退出码 0/1/2/130、`chat` 的 `/clear` `/history` `/help` `/exit`、Ctrl-C |
| `test_dependency_boundary.py` | 4 | 子进程 import 全部运行时模块 + AST 扫描，确保 torch/transformers 不在依赖图里 |
| `test_convert_model.py` | 27 | **转换失败不能破坏已有模型**：非空 `--out` 被拒、`--overwrite` 先建后换、失败后旧模型逐字节不变且仍可生成、跨分片 tied 权重各读自己的分片、下载缓存按 commit 分目录、**截断分片报出自己的错误而不是 numpy traceback**；本轮新增：**已存在的空 `--out` 不需要 `--overwrite`**、只有一个隐藏文件也算非空、**转换中途目录变非空则拒绝安装且里面文件不动**、**中断的首次下载重跑同一条命令能续上**、半截文件不算缓存命中、永远匹配不上的快照永远不会被标记完成、完成的快照复用且不重写标记、无标记目录仍被拒绝但错误里给出三条出路 |
| `test_validate_reference.py` | 46 | **比较器不能给假 PASS**：应有 key 集合由抓取配置推导、两侧都缺也算失败、0 个比较算失败、NaN/Inf 算失败、展平前先查形状、容差未被放宽、**照抄历史结论不能掩盖已经漂移的 tokenizer/权重**；本轮新增：**元数据的应有集合也从协议推导**（样例名来自抓取配置、权重名来自 `WEIGHT_CHECK_SAMPLE`），同一条 tokenizer 样例或权重摘要在**两侧同时缺失**也失败并指名是哪个，以及一个真的 `subprocess` 起 CLI、断言缺数据时**退出码非零**的测试 |
| `test_benchmark.py` | 27 | **基准不能报告没执行的分块**：full 与 chunked 分开测、分块数是实测的前向调用次数、计时区间里没有采样/decode/打印；`--threads 4` 与 `--threads=4` 都在 import numpy 之前导出、非法值交给 argparse、请求值不冒充测量值 |
| **合计** | **480** | 全部通过，本轮实测 4.28 秒（macOS arm64，Python 3.13.14） |

集成测试（23 个，`pytest -m integration` 显式启用）覆盖真实 Qwen3-0.6B：greedy
确定性、固定 seed 采样复现、增量解码无 U+FFFD、流式与阻塞输出一致、多轮历史
真的被重新渲染、上下文溢出显式拒绝、分块/缓存数值等价，以及**用当前代码重跑
NumPy 侧再与已固定的 torch 参考张量对比**——其中 6 个专门守着"不能沿用历史
结论"：每次运行都必须自己重新渲染 tokenizer、自己重新算权重摘要，改动当前
tokenizer 输出或权重数据都会让比较失败，缺 `reference.json` 会要求重新抓取而不是
跳过。整个仓库共 **503** 个测试（480 快速 + 23 集成）。

480 / 23 / 503 是**本轮（2026-10-19 发布收尾）在 macOS arm64 上实测**的；上一轮
（v0.1 修复轮）结束时是 462 / 23 / 485，差值就是本轮新增的 18 个回归测试。本轮
**没有连接远程 Linux**，下文那份容器结果属于上一轮的测量批次。


---

## 项目结构

```
src/slowq/
  ops.py         无状态数值算子（RMSNorm、RoPE、softmax、SiLU、GQA attention）
  cache.py       预分配 KV cache
  config.py      config.json / generation_config.json 解析与校验
  weights.py     manifest 解析、只读 mmap 加载、别名与完整性检查
  model.py       Qwen3 前向
  tokenizer.py   chat template、BPE、增量解码器
  sampling.py    temperature / top-k / top-p，独立 RNG
  engine.py      唯一的生成循环
  metrics.py     计时、内存、吞吐
  cli.py         generate / chat / inspect
tools/
  convert_model.py       safetensors BF16 -> .npy FP32 + manifest（自己解析 safetensors
                         头 + mmap，下载走 urllib：不需要 torch，也不需要 `.[ref]`）
  validate_reference.py  与 Transformers 参考实现逐张量对齐（`ref` 子命令是全项目
                         唯一 import torch 的地方，`numpy`/`compare` 不用）
  benchmark.py           性能与内存测量
examples/quickstart.py   最小可运行示例
docs/architecture.md     架构、张量 shape、cache 语义、计时区间定义
tests/                   快速测试 + 真实模型集成测试
```

详细的模块职责、张量 shape 推导、KV cache 不变量、GQA 折叠技巧和计时区间定义
见 **[docs/architecture.md](docs/architecture.md)**。

---

## 已知限制

诚实列出，不掩饰：

1. **慢。** decode 约 45 tok/s，prefill 约 300–540 tok/s。同一台机器上
   llama.cpp 或 torch 会快得多。这是"每一步都写成可读形式"的直接代价。
2. **内存大。** FP32 权重 2.2 GiB，生成时峰值 RSS 2.6 GiB。没有量化。
3. **只支持 Qwen3。** `config.py` 会显式拒绝其他 `model_type`、带
   `attention_bias` / `use_sliding_window` / `rope_scaling` 的配置，而不是
   静默算错。
4. **batch size 恒为 1。** 没有 continuous batching，没有并发。
5. **上下文溢出直接报错，不截断。** prompt + max_new_tokens 超过 `max_context`
   会抛 `ContextOverflowError`。这是有意为之：静默截断会让结果看起来正常但
   实际丢了内容。
6. **不支持 tool calling、多模态、结构化输出。** 遇到 `tool_calls`、
   `function_call`、list/dict 类型的 content、`tool`/`developer` role 会抛
   `UnsupportedInputError` 并说明原因，而不是丢掉字段继续跑。
7. **不要求跨平台逐 bit 相同。** 不同 BLAS（Accelerate / OpenBLAS / MKL）的
   求和顺序不同，实测 rel_l2 在 1e-6 ～ 1e-5 量级（macOS arm64 最差 7.986e-06，
   Linux x86_64 最差 1.011e-05，都对同一份参考）。采样生成的文本也不保证与
   Transformers 完全一致——只保证 greedy argmax 一致。
8. **KV cache 不分页、不复用。** 每个请求从头开始，没有 prefix caching。
9. **`pip install` 会装进 `huggingface_hub`。** 这是 `tokenizers` 自己声明的
   硬依赖（0.21 与 0.23 都是），SlowQ 无法控制。但**装进来不等于被用到**：
   在干净 venv 里 import 全部 12 个运行时模块后，264 个已加载模块中的第三方
   顶层包只有 `numpy` / `tokenizers` / `jinja2` / `markupsafe`；
   `torch` / `transformers` / `safetensors` / `huggingface_hub` 一个都没有，
   `huggingface_hub` 的 HTTP 栈也随之没有被加载。
   `tests/test_dependency_boundary.py` 用子进程 import 加源码 AST 扫描守着这几个
   名字，离线加载另由 `test_loading_never_reaches_the_network`（把 socket 换成
   会抛异常的桩）看守。这一项在 macOS 和 Linux 的干净 venv 里都实测通过。
10. **PyTorch 参考端只在 macOS arm64 上执行过。** Linux x86_64 上重算的是
    **NumPy 侧**，比较对象是 macOS 抓取并固定的参考张量——结论是"Linux 的
    NumPy 实现与那份参考一致（rel_l2 1.011e-05）"，**不是**"PyTorch 参考在
    Linux 上重跑过"。后者因为该环境拿不到可用的 torch 构建而没有做，所以
    "torch 换一种 BLAS 会不会算出别的数"仍未验证。快速套件、权重转换、
    端到端生成、集成套件、干净环境这五类在两个平台都实测过。
11. **性能数字只代表测过的两台机器，且冷启动区间很宽。** 8-token prefill 在
    页缓存冷的时候实测 6832.7 ms、热的时候 536.3 ms；同一条 19-token prompt 的
    CLI 冷启动 prefill 也测到过 7228.8 ms。Linux 那台是共享机，测量期间
    loadavg 从 2.07 升到 16.13。**不要把这些数字当承诺，也不要跨机器比。**
12. **Windows 上 `import slowq` 会失败。** `src/slowq/metrics.py` 顶层
    `import resource`，而该模块是 Unix-only。已定位、未修复（无法在 macOS 上
    验证修复效果），前后两轮修复都未扩大范围去动它。
13. **大端机器（s390x 等）加载会显式报错。** 转换工具的 `uint16` 位操作按本机
    字节序读取小端数据。会抛错而不是算错，但没有做适配。
14. **`artifacts/validation/` 里的交换文件各约 11 MiB**（`slowq.npz` / `ref.npz`，
    外加 `samples.json` / `reference.json`）。权重检查改成 sha256 摘要之后，
    这里从 300 MiB 降到 11 MiB。集成测试每轮会把它们拷进 `tmp_path` 再重算
    NumPy 侧，不会覆盖已签收的记录。

---

## v0.1 修复记录（第一轮，2026-10-18）

下面 10 项是 v0.1 交付后复查发现的问题，全部已修复并有回归测试守着。
每一项的测试都**在修复前的代码上验证过会失败**——只写测试不验证它咬得住，
等于没写。

| # | 问题 | 修复 | 回归测试 |
| --- | --- | --- | --- |
| 1 | 转换失败或中断会把新旧权重混在一份仍然有效的旧 manifest 后面 | 默认拒绝非空 `--out`；`--overwrite` 时先在旁边建好并验证，最后一步才替换 | `test_convert_model.py` 4 项 |
| 2 | 比较器在张量缺失或出现 NaN 时仍报 PASS（`nan > 容差` 恒为 False） | 应有 key 集合由抓取配置推导；两侧都缺也失败；0 个比较失败；比较前查形状/dtype/有限性；退出码 1 并指出样例与张量名 | `test_validate_reference.py`（当时 38 项，本轮之后 46 项） |
| 3 | 流式结束丢掉 `decoder.flush()` 的尾字节 | 生成循环结束时补一次 flush | `test_engine.py` 流式一致性 |
| 4 | 下载缓存只按 repo 命名，不绑定 revision | 缓存目录改成 `<cache>/<repo>/<commit>`；manifest 只记录真正确认过的来源 | `test_convert_model.py` 下载缓存组 |
| 5 | 集成回归沿用历史 tokenizer/权重检查结论 | 两侧都改成记录**内容**（渲染文本、token id、权重 sha256 摘要 + 摘要规范），`compare` 只采信本轮数据；照抄历史布尔值不再有用 | `test_validate_reference.py` + 集成 6 项 |
| 6 | 基准把单次完整前向报成"4 个 chunk" | full 与 chunked 分开测；chunked 真的跑引擎的分块循环；分块数是包装器数出来的实际前向次数；计时区间里只有前向 | `test_benchmark.py` 27 项 |
| 7 | `DebugProbe` 在非末位置捕获的是末位 logits | 按请求的位置切片 | `test_model.py` |
| 8 | 跨分片 tied 权重拿两个分片的字节区间互相比 | 各自从自己的分片读，比 sha256 | `test_convert_model.py` tied 组 3 项 |
| 9 | `--threads=4` 被接受但从未生效 | 两种写法都在 import numpy 之前导出；报告把"请求的设置"与"探测到的后端"分开，有效线程数明确标注不可观测 | `test_benchmark.py` 子进程组 |
| 10 | 分片被截断时报 `BufferError` 掩盖真因 | `memoryview` 切片会静默 clamp，现在先校验实际长度并抛 `SlowQError`（指出张量名/字节区间/分片名/文件实际长度）；`mapped()` 的清理不再覆盖原错误 | `test_convert_model.py` 3 项 |

第 10 项是**在 Linux 上复核第 1 项时发现的**，不在原始清单里：把真实的 1.5 GB
checkpoint 截到 1.2 GB，旧代码在第 202/310 个张量处抛出一个 numpy reshape 错误，
紧接着 `mmap.close()` 又抛 `BufferError` 把它盖掉，用户看到的是"cannot close
exported pointers exist"。修复后同一场景输出：

```
error: tensor 'model.layers.25.mlp.gate_proj.weight' claims bytes
1194972904..1201264360 of model.safetensors, but that file is only 1200000000
bytes long: the checkpoint is truncated or was not fully downloaded
```

退出码 1，已有模型 manifest sha256 不变、310 个张量齐全、仍能正常生成，
没有暂存目录残留。

**没有改动的：** 数值容差一个字都没放宽（rel_l2 1e-3 / max_abs 5e-2，权重 0.0）。
修复前后与参考对齐的最差值都是 rel_l2 7.986e-06、max_abs 1.063e-04。


---

## v0.1 发布收尾修复记录（第二轮，2026-10-19）

上面十项之后又发现三个问题，都与"工具在失败或不完整状态下的行为"有关，不影响
数值路径。本轮同时补齐了发布文件（LICENSE、依赖说明、仓库地址占位符）。
**本轮没有重跑性能基准，没有重新抓取 PyTorch 参考端，也没有连接远程 Linux。**

| # | 问题 | 触发条件（都已实测复现） | 修复 | 回归测试 |
| --- | --- | --- | --- | --- |
| 1 | **首次下载中断后无法重试**：缓存目录变成死路 | `download_snapshot()` 先建目录、`.slowq-snapshot.json` 只在**全部文件下完之后**才写。中途失败留下一个**没有标记**的目录；重跑同一条命令时 `verify_snapshot()` 先拒绝它（无法证明来源），下载器根本没机会执行 | 标记改成**下载第一个文件之前**就写，带 `complete: false`；每个文件下完后按 revision 的 listing（先 size 再 LFS sha256）复核，全部通过才改写成 `complete: true`。无标记目录**仍然拒绝**（不自动删除、不清缓存），但错误信息给出三条出路：删掉那一个目录后重跑、换 `--download-dir`、用 `--source`。旧标记缺 `complete` 字段按完成处理，已有缓存不会被重下 | `test_convert_model.py` 3b 组 7 项，**其中 5 项在修复前实测失败** |
| 2 | **空输出目录的前后检查互相打脸** | `check_output_is_installable()` 允许"已存在且为空"，`install()` 却拒绝任何已存在的目录。用户预先 `mkdir` 好 `--out`，前检查放行，几十秒转换之后才被告知"转换过程中出现了东西，不覆盖" | 后检查不再自己判断，直接让 `staging.replace(output)` 这个原子 rename 去判：空目录则成功，非空则 `OSError`（`ENOTEMPTY`），报错时列出实际条目并说明"里面一个字节都没动"。判据只有一份，就不会漂移；**没有新增 CLI 选项** | `test_convert_model.py` 1c 组 3 项，**其中 2 项在修复前实测失败** |
| 3 | **`compare` 的元数据完整性检查可被"两侧同时缺一条"绕过** | 数值张量由 `expected_numeric_keys()` 兜住，但 `check_tokenizer()` / `check_weights()` 遍历的是**两个文件各自记了什么**的并集：同一条记录在两侧同时缺失时根本不进入循环 | 应有的样例名从抓取配置推导（`capture["sample_names"]`），应有的权重名从 `WEIGHT_CHECK_SAMPLE` 推导；单侧缺失、两侧同时缺失都失败并指名是哪个样例/哪个权重。**容差与验证内容都没减少**，`expected_numeric_keys()` 原样保留 | `test_validate_reference.py` 新增 8 项，**其中 6 项在修复前实测失败** |

"在修复前实测失败"的取证方式，两组不同，都说清楚：

- **第 1、2 项**：先写测试、在**尚未修改的 `tools/convert_model.py`** 上跑
  （`-k "empty_output or hidden_file or becomes_non_empty or interrupted or
  half_written or never_match or without_rewriting or cannot_prove or cannot_be_read
  or another_repo_or_endpoint"` → **7 failed, 4 passed, 16 deselected**），
  7 项失败各自的错误原文都指向被修的那一处：两项是
  `FileNotFoundError: …/.slowq-snapshot.json`（标记当时还没写）、一项
  `DID NOT RAISE SlowQError`（永远匹配不上的文件却被当成下载成功）、一项
  `KeyError: 'complete'`、一项 `assert 1 == 0`（`main()` 因为安装那一步拒绝空目录而
  返回 1）、一项 `assert "not empty" in err` 失败（当时打印的是旧的
  "appeared while converting"）、一项拒绝信息里没有 `--download-dir` 这条出路。
  另外 3 项在修复前就通过（隐藏文件也算非空、标记读不出来被拒、标记指向别的
  repo/endpoint 被拒），它们守的是**不能被这次改动破坏**的既有行为；第 4 个 passed
  是第一轮既有的"非空 `--out` 被拒"，被 `-k empty_output` 顺带选中。
- **第 3 项**：改完代码之后重新取证——把 `tools/validate_reference.py` 复制到
  `/tmp/prefix-check/`，只把新增的 `expected_sample_names()` / `expected_weight_names()`
  改成返回空（这正是修复前"只看两个文件并集"的行为），再跑同一个测试文件：
  **6 failed, 40 passed**。其中 CLI 那项的输出是假 PASS 的原文：
  `assert 0 == 1`，子进程 stdout 结尾为
  `PASS: 44 tensors compared within the declared tolerances.`——而两侧都已删掉同一条
  tokenizer 样例和同一个权重摘要。真实仓库里的文件没有被这次取证改动。

发布文件（不是缺陷，但发布前必须补齐）：

| 项 | 之前 | 现在 |
| --- | --- | --- |
| LICENSE | 仓库里**没有** LICENSE 文件，只有 `pyproject.toml` 写着 `license = { text = "MIT" }` 和 README 一句"MIT。" | 新增根目录 `LICENSE`，标准 MIT 全文，署名 `Copyright (c) 2026 SlowQ contributors`（与 `pyproject.toml` 的 `authors` 一致）；`pyproject.toml` 补 `License :: OSI Approved :: MIT License` classifier 和 `[tool.setuptools] license-files = ["LICENSE"]` |
| 转换的依赖说明 | README 写"转换是**唯一**需要 torch 的步骤"，快速开始里为了转换装 `.[ref]`；项目结构里写 `convert_model.py …（唯一接触 torch 的地方）` | 转换器实际只用标准库 + numpy（自己解析 safetensors 头、`mmap` 取字节、`urllib` 下载），**不需要 torch，也不需要 `.[ref]`**。README 三处都已改正，并写明 `.[ref]` 只服务于 `validate_reference.py ref`（全项目唯一 `import torch`
的地方，在 `run_reference_side()` 里；`src/`、`examples/` 与其余工具一个都没有）。
`docs/architecture.md` §10 原本就是对的，未改 |
| 仓库地址 | `pyproject.toml` 和 README 使用占位仓库地址 | 第二轮修复时尚未创建远程仓库，因此先删除占位地址。首次公开提交时已补上真实仓库 `https://github.com/luolinglong0611/SlowQ`、项目元数据和克隆命令。 |

本轮实际执行的验证（macOS arm64，`.venv` 里 Python 3.13.14、numpy 2.5.4、
tokenizers 0.21.4、Jinja2 3.1.6）：

| 命令 | 结果 | 退出码 |
| --- | --- | --- |
| `.venv/bin/python -m pytest -m "not integration" -q` | **480 passed, 23 deselected**，4.28 s | 0 |
| `SLOWQ_MODEL_DIR=$PWD/models/Qwen3-0.6B-slowq .venv/bin/python -m pytest -m integration -q` | **23 passed, 480 deselected**，17.20 s | 0 |
| `.venv/bin/python tools/validate_reference.py compare` | **PASS: 116 tensors compared within the declared tolerances.**；4/4 tokenizer 样例、6/6 权重摘要、20/20 argmax；最差 `zh-single-turn.full_prefill.logits` rel_l2 7.986e-06、max_abs 1.063e-04（与上一轮同一份参考的实测值逐位相同） | 0 |
| `shasum -a 256 artifacts/validation/*`（`compare` 前后各一次） | 四个文件 sha256 **逐个相同**：本轮没有改写已签收的参考数据 | 0 |
| `.venv/bin/python -m pip wheel --no-build-isolation --no-deps -w /tmp/slowq-dist .` + `setuptools.build_meta.build_sdist('/tmp/slowq-dist')` | `slowq-0.1.0-py3-none-any.whl`（53788 B）含 `slowq-0.1.0.dist-info/licenses/LICENSE`（1075 B）；`slowq-0.1.0.tar.gz`（135925 B）顶层含 `LICENSE`。两者 METADATA / PKG-INFO 都有 `License: MIT`、MIT classifier、`License-File: LICENSE`，且**没有任何 `Project-URL`** | 0 |
| `/tmp/prefix-check` 里的行为回退取证（见上） | **6 failed, 40 passed** | 1 |

`build` 模块没装（也不为此联网安装），所以 wheel/sdist 是用 pip + `build_meta`
直接构建的；构建产物全部写在 `/tmp/slowq-dist`，故障注入全部写在 `/tmp`，
临时模型目录由 pytest 的 `tmp_path` 提供。**现有真实模型
`models/Qwen3-0.6B-slowq`、`models/Qwen3-0.6B-hf` 与 `artifacts/validation/`
一个字节都没动。**

**本轮未验证 / 未做的：**

- **没有在 Linux 上重跑任何测试。** §"跨平台验证"里的 462 / 23 / 485 与所有性能
  数字仍属上一轮批次；本轮的 480 / 23 / 503 只在 macOS arm64 上测过。
- **没有重跑性能基准**，也没有重新抓取参考端。抓取不是被环境卡住的——本机 `.venv`
  里装着 torch 2.14.1 + transformers 4.51.3，技术上能抓；是**选择不抓**：重抓会原地
  覆盖已签收的 `artifacts/validation/`，而本轮要求不动这份数据，且抓取格式
  （`artifact_format` 仍是 2）没变、本来也不需要重抓。因此"本轮改动对性能无影响"
  这句话**没有本轮的实测支撑**，只有"没有触碰数值路径"这个代码层面的依据。
- **没有做 GitHub Release、没有推代码、没有发 PyPI**（按要求）。逐条的发布条件核对
  见 `SlowQ-v0.1-修复报告.md` §8：技术前提已满足，卡住的是"没有真实远程仓库"这一条。
- 一个**顺手发现、本轮未修**的问题：`http_download()` 遇到网络层错误抛的是
  `urllib.error.URLError`，而 `main()` 只捕 `SlowQError`，所以断网/DNS 失败会以
  traceback 形式暴露给用户，而不是一行 `error: …`。缓存状态仍然是可续的（标记
  已写、文件按 listing 复核），所以它只是观感问题，不在本轮三项之内，未改动。


---

## 许可

MIT，全文见仓库根目录的 [`LICENSE`](LICENSE)。版权声明的署名与 `pyproject.toml`
里的 `authors` 一致，都是 `SlowQ contributors`；构建出的 wheel
（`slowq-0.1.0.dist-info/licenses/LICENSE`）与 sdist（顶层 `LICENSE`）都带着它。

Qwen3-0.6B 权重归阿里巴巴通义实验室所有，本仓库不包含任何权重文件。
