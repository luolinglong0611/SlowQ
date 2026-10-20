# 性能与内存

以下数字来自 v0.1 第一轮修复后的历史重测批次 `m5pro-recapture`。
2026-10-20 本轮仅整理文档，没有重新测量。结果只代表这组硬件、软件和运行条件。

## 测量环境

| 项目 | 配置 |
| --- | --- |
| 机器 | Apple M5 Pro，18 逻辑核 |
| 系统 | Darwin 25.6.0，arm64 |
| Python / NumPy | 3.13.14 / 2.5.4 |
| BLAS | Accelerate |
| 线程 | 未指定 `--threads`，相关 BLAS 线程环境变量未设置 |
| 模型 | Qwen3-0.6B，FP32，revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| 上下文 / prefill 分块 | `max_context=1024` / `prefill_chunk_size=128` |

## 生成速度

独立 decode 探针记录为 **44.16 tok/s（22.64 ms/token）**；它测量短 prefill 后的
32 次逐 token 前向及 greedy token 选择，不包含模型加载和文本输出。
下面是端到端工作负载的记录，greedy，每个工作负载运行 3 次并取中位数：

| 工作负载 | 输入 / 生成 tokens | 首 token 延迟 | Prefill | Decode |
| --- | --- | --- | --- | --- |
| 中文单轮 | 19 / 13 | 66.6 ms | 286.5 tok/s | 46.54 tok/s |
| 英文与 system 消息 | 46 / 54 | 110.8 ms | 416.1 tok/s | 44.73 tok/s |
| 多轮对话 | 66 / 12 | 146.2 ms | 452.5 tok/s | 44.94 tok/s |
| Emoji 与混合文字 | 30 / 21 | 79.7 ms | 377.7 tok/s | 45.11 tok/s |

## Prefill 探针

`full` 一次处理整个输入；`chunked@128` 每次最多处理 128 tokens，并写入同一份
KV cache。计时覆盖前向循环，不包含采样、逐 token 生成或打印。每行是一次探针测量，
不采用上表的三次中位数；前向次数由实际调用计数得到。

| 输入 tokens | 模式 | 耗时 | 吞吐 | 前向次数 |
| --- | --- | --- | --- | --- |
| 32 | full | 81.7 ms | 391.8 tok/s | 1 |
| 32 | chunked@128 | 80.7 ms | 396.4 tok/s | 1 |
| 128 | full | 236.4 ms | 541.4 tok/s | 1 |
| 128 | chunked@128 | 229.1 ms | 558.6 tok/s | 1 |
| 512 | full | 992.0 ms | 516.1 tok/s | 1 |
| 512 | chunked@128 | 1024.6 ms | 499.7 tok/s | 4 |

首次前向还可能承担权重的磁盘读取：该次记录中，最先执行的 8-token `full`
耗时 6832.7 ms。随后执行的路径受益于已读入的权重页，不能用两者之差判断分块收益。
加载映射、首次前向和后续运行应分别记录，页缓存状态也会影响复测结果。

## 内存

| 项目 | 记录值 |
| --- | --- |
| FP32 权重逻辑大小 / `.npy` 文件总大小 | 各约 2.2 GiB |
| KV cache，容量 1024 tokens | 224.0 MiB（224.0 KiB/token） |
| 映射权重后的进程峰值 RSS | 184.6 MiB |
| 完成探针后的进程峰值 RSS | 2.5 GiB |
| 完成生成后的进程峰值 RSS | 2.6 GiB |

权重通过只读 mmap 加载；建立映射不等于所有权重页已经驻留内存。
权重逻辑大小、预分配 KV cache 容量与进程 RSS 是不同指标，不能相互替代或直接相加。
表中 RSS 是截至相应阶段的进程历史峰值，不是该时刻的瞬时内存占用。

## 在自己的机器上复现

先按 [README](../README.md) 安装运行依赖并转换模型。以下命令都在项目根目录执行：

```bash
python tools/benchmark.py \
  --model models/Qwen3-0.6B-slowq \
  --max-context 1024 --prefill-chunk-size 128 \
  --decode-steps 32 --repeat 3 \
  --json artifacts/benchmark/local-default.json

python tools/benchmark.py \
  --model models/Qwen3-0.6B-slowq --threads 4 --repeat 3 \
  --json artifacts/benchmark/local-threads4.json
```

第二条用于观察同一台机器的线程设置影响，不代表推荐线程数。
脚本报告请求的线程设置和检测到的 BLAS 后端；有效线程数标为不可观测。
比较时应固定模型、上下文、输入、线程设置和机器负载，并区分冷启动与后续运行。
共享机器的负载会带来额外波动，不同机器和 BLAS 的数字不能直接用于性能排名。

参数与计时实现见 [benchmark.py](../tools/benchmark.py)，张量和缓存设计见
[架构说明](architecture.md)。这些记录没有包含与其他推理框架的同条件对照测试。
