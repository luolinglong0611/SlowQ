# 测试与参考验证

本文说明 SlowQ v0.3.0 怎么验证：跑什么命令、比什么、门限是多少、实测结果如何。
所有命令从项目根目录执行。安装与模型下载需要网络；准备好依赖和本地模型后，
测试与参考比较可以完全离线运行。

返回 [README](../README.md)；模型清单与支持范围见[模型支持](models.md)；
实现细节见[架构说明](architecture.md)。

<a id="current-validation"></a>

## 当前验证入口（v0.3.0）

| 层 | 命令或入口 | 执行范围 | 需要真实权重 | 证明什么 |
| --- | --- | --- | --- | --- |
| 无权重自动测试 | `python -m pytest -m "not integration" -q` | macOS / Linux CI；本地也可运行 | 否 | 微型模型、工具与依赖边界的回归 |
| 干净 wheel 安装 | `python tools/ci/check_wheel.py` | macOS / Linux；纯文本与带 `vision` 两种安装 | 否 | 构建产物可安装、CLI 可用、可选依赖边界正确 |
| 真实模型集成 | `SLOWQ_MODEL_DIR=<dir> python -m pytest -m integration -q` | 显式运行，按 manifest 的任务过滤 | 是 | 指定 checkpoint 的真实权重功能与离线行为 |
| 官方参考数值 | `validate_reference.py` 的 `numpy` → `ref` → `compare` | 显式准备模型与参考环境后运行 | 是 | NumPy 实现与固定官方参考的数值一致性 |

快速测试用临时构造的微型模型和 tokenizer，覆盖算子、KV cache、采样、流式输出、
config 家族规则、转换器、验证工具、性能工具与依赖边界。`.[dev]` 只增加 pytest，不安装 PyTorch。

自动运行的定义见 [CI 工作流](../.github/workflows/ci.yml)：Ubuntu / macOS ×
Python 3.11 / 3.15，运行源码快速测试和下面的 wheel 检查。
工作流配置与实际执行结果分开记录；配置了某个平台不等于已获得该平台的通过记录。

CI 从不含 `models/` 和本地 `artifacts/` 的源码开始。依赖本地历史性能证据的测试，
在证据缺失时带原因跳过；没有 Git 元数据的源码副本会跳过相关 Git 用例。
Git checkout 需要完整历史，以便验证记录中的基线 revision；CI 使用 `fetch-depth: 0`，
浅 clone 中缺少所需 commit 不属于可自动跳过的情况。一般的工具回归和微型模型测试仍运行。
因此，带本地证据的开发目录与干净源码副本的 passed / skipped 数可以不同，
应同时记录命令、源码版本、可用证据和跳过原因，不能只比较 passed 数。

**无权重 CI 通过不代表八个真实模型已经在 CI 上验证，也不代表 Linux 的真实模型支持矩阵
已经补齐。** 真实权重与官方参考验证继续按下文协议独立执行；既有 macOS 结果和
Linux v0.1 记录保留原始日期、平台与批次，不改写成 v0.3.0 新测结果。

### v0.3.0 收尾验证（2026-11-07）

以下为发布准备阶段的本地执行结果；后续 GitHub 托管 runner 的执行记录见
[CI 运行列表](https://github.com/luolinglong0611/SlowQ/actions/workflows/ci.yml)。

| 平台与 Python | 源码与证据条件 | 无权重快速测试 | 干净 wheel 检查 |
| --- | --- | --- | --- |
| macOS arm64 / 3.13.14 | 工作区保留历史 `artifacts/` | 1121 passed，62 deselected；38.03 s | 纯文本 / `vision` 两个新 venv 均 PASS |
| Linux arm64 容器 / 3.11.18 | 普通用户；完整 Git clone；无 `models/`、`artifacts/` | 1076 passed，46 skipped，62 deselected；8.20 s | 纯文本 / `vision` 两个新 venv 均 PASS |
| Linux arm64 容器 / 3.15.2 | 普通用户；完整 Git clone；无 `models/`、`artifacts/` | 1076 passed，46 skipped，62 deselected；8.84 s | 纯文本 / `vision` 两个新 venv 均 PASS |

六个 wheel 安装环境的 `pip check` 均通过；工作流也通过官方 Docker 镜像的 `actionlint`
检查（退出码 0）。Linux 在线安装曾遇到 PyPI 超时，最终用本轮下载的 wheel 缓存在
断网容器和全新 venv 中完成同一 `check_wheel.py` 检查，校验内容没有改变。
这些 Linux 结果来自 arm64 容器，不是 Linux x86_64 或 GitHub 托管 runner 的实跑记录。

原始日志保存在本机 `artifacts/ci-v0.3.0-20261107/`，该目录被 Git 忽略、不随仓库分发：

- macOS：`macos-python313-tests.log`、`macos-python313-wheel.log`。
- Linux 测试及在线安装尝试：`linux-python311-tests-online-wheel-attempt.log`、
  `linux-python315-tests-online-wheel-attempt.log`。
- Linux 最终离线 wheel 检查：`linux-python311-wheel-offline.log`、
  `linux-python315-wheel-offline.log`。

## 快速测试

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -m "not integration" -q
```

图文相关快速测试使用 `python -m pip install -e ".[dev,vision]"` 安装 Pillow 后运行。
纯文本环境必须仍能导入 `slowq.vlm`；只有实际读图时才需要 Pillow。
干净 wheel 安装检查可在本地用同一个入口复现：

```bash
python tools/ci/check_wheel.py
```

[检查脚本](../tools/ci/check_wheel.py) 从当前源码建立临时干净构建目录，避免已有 `build/`
混入 wheel；再创建纯文本与带 `vision` 的两个新环境，在 checkout 之外检查安装后的
模块路径、包与 CLI 版本、依赖边界、各子命令帮助、Pillow 可选行为和 `pip check`。
它不读取真实模型或历史性能产物；构建和安装依赖需要网络或预先准备的包缓存。
这个检查验证交付的 wheel，与源码 editable 安装的测试互补。

<a id="historical-quick-tests"></a>

### 历史快速测试记录（2026-10-21—10-28）

以下完整保留当时的计数、耗时、路径及审查过程。这里的“当前树”“最终树”“本轮”
均指所标日期的源码快照，**1042 等计数不是 v0.3.0 当前测试数量**。
后续新增测试及干净源码中因证据缺失而跳过的测试，应以当前验证入口和本次运行日志为准。

**2026-10-28 二次审查修复后在同一台机器上的实际输出（当时的源码树）：**

```
1042 passed, 62 deselected in 22.66s
```

落盘在 `artifacts/perf-run-20261028-185623/quick-fixed.log`
（**两步落盘**：先跑到 `/tmp/quick-r5.log` 再 `cp` 进来，避免复现块测试读到
正在被截断的日志）。
**同一棵最终树上这条命令跑过多次，计数每次都是 1042 passed / 62 deselected，
退出码都是 0**；其中**落盘成权威日志的五次**耗时 22.24 s、22.99 s、23.66 s、23.48 s、
**22.66 s**——最后一次（22.66 s）就是上面引用的那一份，
前四次的日志按跑的顺序改名留在旁边
（`quick-fixed.log.prev-1042`、`.prev-1042-b`、`.prev-1042-c`、`.prev-1042-d`）。
此外还有若干次「文档改完之后的确认运行」，命名与用意见下文。
上一轮那份 `995 passed, 62 deselected in 13.08s` 原样留在
`artifacts/perf-round1-rerun/quick-fixed.log`（它前两代留在同目录的
`quick-fixed.log.prev-994` 与 `quick-fixed.log.prev-991`）：
**四代计数（991 / 994 / 995 / 1042）各有一份原始日志在盘上，一份没删也没改名冒充。**
被取代的那几份**故意都不叫 `quick*.log`**：
生成器的 `quick_suite_expectation()` 按名字排序取第一个匹配，留在 `quick-fixed-prev.log`
这种名字下会排在新日志前面，把已被取代的计数喂给 §10；`.prev-NNN`／`.confirm`
这类后缀根本不匹配 `quick*.log`（`sorted(glob("quick*.log"))` 在这个目录里只返回
`quick-fixed.log` 一项，2026-10-28 现场打印过）。命令就是
[推理加速第 1 轮实测](推理加速-第1轮-2026-10-23.md) §10 第 1 步那一条。

**这个「跑一次、把耗时写进文档、文档又变了」的循环是怎么收尾的，说清楚：**
上面那个 22.66 s 落盘成了权威的 `quick-fixed.log`。把这一句话写进本节本身又是一次
文档改动，于是每改完一次就再跑一次确认，确认日志按顺序留在同目录的
`quick-fixed.log.confirm`、`.confirm2`……（后缀不匹配 `quick*.log`，
不会顶掉权威那份）。**盘上最后那一份 `.confirm*` 就是字面最终树上的一次运行**——
本节不写「一共跑了几次」，因为那个数字在本节每次被编辑之后都会变，
写下来立刻就过期；要数就 `ls artifacts/perf-run-20261028-185623/quick-fixed.log*`。
**能这么收尾的前提是：本节与 §14 这两份文档没有任何测试读。**
这一条是可核的，不是断言：
`grep -rn --include='*.py' "validation.md\|architecture.md" tests/`
只命中 `tests/test_perf_tooling.py` 里的一条 docstring（第 3734 行，讲 `run_reference.sh`
不写 `$dir/compare.log`，与测试计数无关）；
该文件里每一处 `report.DOC` 赋值都指向 `tmp_path`（生成到临时目录再断言），
唯一读真实文件的那一项读的是 `quick-fixed.log` 的**计数**（`pytest_summary`），不是耗时。
所以耗时是浮动的观测量、计数才是门禁；**每一次运行的计数都是
`1042 passed, 62 deselected`、rc 都是 0**，这一条不随文档编辑而变。
**1042 = 上一轮的 995 + 二次审查轮新增的 47 项**，收集总数 1057 → **1104**
（1042 快速 + 62 集成；`pytest -q --collect-only` 与 `pytest -q -m integration --collect-only`
分别给出 `1104 tests collected` 与 `62/1104 tests collected (1042 deselected)`）。
逐文件对得上（`pytest -q -m "not integration" --collect-only`，2026-10-28 重算，
17 个文件逐项相加正好 1042）：

| 文件 | 第 1 轮 | 上一轮 | 当前 | 本轮差 | 本轮这两个文件里新增的是什么 |
| --- | --- | --- | --- | --- | --- |
| `test_perf_tooling.py` | — | 135 | 176 | **+41** | 问题 1（`out_guard` 与三个驱动的输出目录／退出码）、问题 3（剔除后重算均衡、四个判定档位、汇总与逐行一致）、问题 4（两个驱动的累计失败退出码），外加 §10 期望值跟着新运行根走 |
| `test_vlm.py` | 100 | 105 | 111 | **+6** | 问题 2：同权重不同 vision／预处理配置的替换、替换后重算再命中、未替换持续命中、文本模型明确拒绝 |
| `test_model.py` | 73 | 91 | 91 | 0 | —（上一轮：审查 1c） |
| `test_embedding.py` | 46 | 52 | 52 | 0 | —（上一轮：审查 1a） |
| `test_session.py` | 52 | 57 | 57 | 0 | —（上一轮：审查 2） |
| 其余 12 个文件 | 555 | 555 | 555 | 0 | 未改动 |
| 合计 | **826** | **995** | **1042** | **+47** | |

**41 与 6 这两个数是算出来的**（176−135、111−105），
**但「41 项里问题 1 占几项、问题 3 占几项」这一层拆分本轮给不出来**：
上一轮那份 `tests/test_perf_tooling.py` 在磁盘上没有任何副本
（`find . artifacts /tmp -name "test_perf_tooling.py*" -not -path "./tests/*"` 无结果，
`artifacts/perf-round1/baseline-tree/` 只有 `BASELINE-REVISION` 与 `src/`），
所以没有可比对的前一代文件。**能验证的是覆盖，不是增量归属**：
反证脚本对每个问题用的选择器各选中多少项、其中多少项在回退修复后变红，
逐项落在 `artifacts/perf-round2-falsification-20261028-233602/*.log` 里
（见 §反证：四个问题各自的最小回归）。

**本轮 47 项全部落在 `test_perf_tooling.py` 与 `test_vlm.py` 两个文件里**，
其余 15 个文件逐项复核一个没变——这与「只修四个问题、不扩范围」是一致的：
`src/slowq/` 里被改的只有 `vlm.py`（问题 2）与 `ops.py` 的一条 docstring，
问题 1／3／4 改的全是 `tools/perf/`，所以新增测试也只可能落在这两个文件里。

「其余 12 个文件」里包含第 1 轮那张表单独列过、本轮没有再动的
`test_ops.py`（48）、`test_cli.py`（46）、`test_cache.py`（34）、`test_weights.py`（35）、
`test_benchmark.py`（29），以及原本就归在「其余 7 个文件」里的 363 项
（`test_validate_reference.py` 111、`test_config.py` 62、`test_tokenizer.py` 56、
`test_engine.py` 55、`test_sampling.py` 43、`test_convert_model.py` 30、
`test_dependency_boundary.py` 6）——这 363 项本轮逐项复核，一个没变。

`test_perf_tooling.py` 的 135 项测的是**工具本身**，不是被测模型：
「按路径 import 不得解析 `sys.argv`」「docstring 里引用的每个区间必须覆盖所有保留的运行」
「最新一次运行的代码哈希必须仍与源码一致」，以及本轮新加的一批
「报告里的每个数字必须能从它引用的那份原始数据算出来」——
包括 §0 引用的端到端倍率与集成测试耗时、§9.4 的三行待重测、§10 的期望值、
§3 跨遍自检列的判定、§1.5 每棵被测树 digest 是否仍能复现的四分类、
`make_report` 里那份 `source_identity.digest` 副本与原件逐块相符、以及本文自身的生成器哈希。
其中三项测的是 `run_reference.sh`：汇总表的每一列都必须出自**本次运行自己产生的**
日志，tagged 汇总表能用 Python 从它引用的文件逐列复算，被取代的汇总表必须原样
留在磁盘上并且旁边有更正说明。
**第 135 项把同一条规矩用到文档的散文上**：
`test_the_pre_round1_identity_row_counts_the_models_it_actually_covers`
从 `artifacts/validation/*/` 重算 [推理加速第 1 轮实测](推理加速-第1轮-2026-10-23.md)
§2.2 那行「与 pre-round1 备份逐字节相同」到底覆盖了几个模型，
要求该行按 `；` 分句后，点名 `slowq.npz` 的那句带上 npz 的计数、
点名 `compare.log` 的那句**同时**带上原始计数与滤掉计时行后的计数、
并写出把两个数分开的那个模型名。生成器现在也是从磁盘算这一格的
（`reference_identity()` / `timing_only_difference()` / `reference_identity_cell()`），
所以文档与它的检查者不可能各说一套。详见 §参考验证：五代 npz 逐字节相同 里
「与 pre-round1 备份的比较」那一段。
每一处都做过反证：把派生值改回硬编码，对应的测试必须变红，
记录在 `artifacts/perf-round1-rerun/falsification/`（**24 份**）。
最后那份 `preround1-blanket-byte-identity.log` 是这批里唯一**不需要注入**的：
缺陷就是生成器当时印的那句字面量，新测试直接对它变红。

**2026-10-23 推理加速第 1 轮在 macOS arm64 / Apple M5 Pro 上的实际输出：**

```
826 passed, 62 deselected in 5.58s
```

那 62 项是真实模型集成测试，被 `-m "not integration"` **deselect**（根本不收集）。
不加标记过滤直接跑 `python -m pytest -q`，它们会被收集后**skip**（888 = 826 + 62）。
skip 原因是没设 `SLOWQ_MODEL_DIR`。两种写法通过的快速测试项数相同，
差别只在 pytest 报告这 62 项的方式；引用输出时要与命令对应。

**826 = 上一轮的 685 + 本轮新增的 141 项**，逐文件对得上（`--collect-only -q`
在两棵树上的收集数之差）：

| 文件 | 685 那轮 | 本轮 | 差 | 属于哪一项改动 |
| --- | --- | --- | --- | --- |
| `test_session.py` | — | 52 | **+52** | 优化 3：新文件，会话接口 |
| `test_model.py` | 48 | 73 | **+25** | 优化 1（分块只投影末 chunk）与优化 2（单 token 不建 mask） |
| `test_vlm.py` | 81 | 100 | **+19** | 优化 4：`ImageFeatureCache` |
| `test_ops.py` | 33 | 48 | **+15** | 优化 2：softmax / sigmoid / 全零 mask |
| `test_embedding.py` | 35 | 46 | **+11** | 优化 4：`DocumentIndex` |
| `test_cli.py` | 40 | 46 | **+6** | 优化 3：`slowq chat` 的会话路径 |
| `test_cache.py` | 28 | 34 | **+6** | 优化 3：`KVCache.truncate` |
| `test_weights.py` | 30 | 35 | **+5** | 优化 4：`Weights.fingerprint` |
| `test_benchmark.py` | 27 | 29 | **+2** | 优化 1：基准里的分块探针 |
| 其余 7 个文件 | 363 | 363 | 0 | 未改动 |

收集数之差大于新增 `def test_` 之数，差额来自参数化用例（`test_model.py` 与
`test_ops.py` 里按 chunk 大小、scale、mask 长度展开的那些）。

**2026-10-21 验证工具修复轮在同一台机器上的实际输出（上一轮，保留作对照）：**

```
685 passed, 62 deselected in 6.42s
```

**685 = 接入批次的 646 + 发布前修复轮的 19 项 + 验证工具修复轮的 20 项。**
后 20 项全在 `test_validate_reference.py`（91 → 111）：16 项守
`compare` 的向量输入字段校验，4 项直接执行生产的 `_reference_embedding_capture`。
发布前修复轮那 19 项是 `test_embedding.py` +2、`test_model.py` +4、
`test_vlm.py` +6、`test_validate_reference.py` +7，当时合计 665。
耗时随宿主机内存压力浮动（同一台机器空载约 4.1 s，历次实测 4.09 ~ 6.42 s）；
**通过项数不受影响。**
v0.1 的检查没有被删掉，而是按架构规则重写：原来断言"Qwen3 一定有 Q/K RMSNorm"的地方，
现在断言"qwen3 家族有、qwen2 与 llama 家族没有"，负向用例保留。

## 准备真实模型

`models/` 和 `artifacts/` 都被 Git 忽略，**权重、转换结果与验证产物不随仓库分发**。
完整验证需要官方 checkpoint、转换后的 FP32 目录和一次参考抓取。

```bash
python tools/convert_model.py --repo <官方 repo id> --out models/<name>-slowq
```

转换不需要 PyTorch。已有官方 checkpoint 时用 `--source /path --skip-download` 替代 `--repo`。
转换器会解析并记录该 repo **当时**的 commit，不会把 v0.1 为 Qwen3-0.6B 固定的 revision
套到别的 repo 上。默认拒绝覆盖非空输出目录；转换先在暂存目录完成并自检，再安装到目标路径，
所以中途失败不会破坏已有目录。**不要覆盖或删除你已有的模型目录。**

8 个模型的 repo、commit、体积见[模型支持 · 来源与版本](models.md#来源与版本)。

## 参考验证：三进程协议

```bash
python -m pip install -e ".[ref]"       # 只有 ref 子命令需要
M=models/<name>-slowq                   # 转换后的目录
H=<官方 checkpoint 快照目录>

python tools/validate_reference.py numpy   --model "$M"   # 只依赖 NumPy
python tools/validate_reference.py ref     --model "$M" --hf-dir "$H"   # 需要 torch/transformers
python tools/validate_reference.py compare --model "$M"   # 只依赖 NumPy
```

三个子命令是**三个独立进程**，通过文件交换数据，避免两套模型同时常驻内存。
`ref` 读取 NumPy 侧记录的输入与解码路径，用 CPU、FP32、`eval()`、eager attention、
batch size 1 计算参考结果。`--artifacts` 缺省时由 `--model` 推导为
`artifacts/validation/<模型目录名>`，所以验证第二个模型不会覆盖第一个的抓取。

`.[ref]` 安装 PyTorch、`transformers==4.51.3` 等参考依赖。**运行时、转换器以及
`numpy` / `compare` 子命令都不需要 PyTorch 或 Transformers**；这条边界由
`tests/test_dependency_boundary.py` 守住。

每个抓取目录里的文件：

| 文件 | 内容 |
| --- | --- |
| `samples.json` | NumPy 侧的 tokenizer 输出、权重摘要、抓取配置与来源 |
| `slowq.npz` | NumPy 侧的 logits / hidden states / 向量 / 视觉中间结果 |
| `reference.json` | 参考侧的 tokenizer 输出、权重摘要、配置与环境 |
| `ref.npz` | 参考侧的同名数值 |
| `compare.log` | 最近一次 `compare` 的完整输出 |
| `vision-*.png` | 仅图文任务：NumPy 侧生成、参考侧按 sha256 校验的输入图 |

`compare.log` 里不含结论以外的状态，`ref.npz` 与 `reference.json` 也不含判定结果，
所以没有可以从旧运行里"抄过来"的 PASS。

**上表这一行曾经是假的。** `compare` 子命令对抓取目录是只读的（全工具只有 4 处写入，
都在抓取一侧），所以 `compare.log` 只可能是**某条命令把 stdout 重定向进去的**。
`tools/perf/run_reference.sh` 早先重定向到 `$OUT/<model>-compare.log`，然后
`tensors=`／`worst_*_rel_l2=`／`log=` 四列全部从 `$dir/compare.log` 解析——即从
**上一次运行留下的那一代**解析，而 `npz=` 那一列比的却是本次运行刚写的
`slowq.npz`。两列看起来是同一句话，实际指向两次运行。更糟的是 `log=IDENTICAL`
由 `cmp "$dir/compare.log" "$dir/compare-$TAG.log"` 决定，而后者正是脚本几行之前
从前者 `cp` 出来的：**一个拿文件和它自己的副本比的列不可能失败**。它确实没失败，
而且确实是错的——SmolLM2-135M 本次运行的日志与那份陈旧日志差一行
（`[compare] took 0.1 s` → `0.0 s`），汇总表仍印 `log=IDENTICAL`。

现在驱动自己 `cp` 一份到 `$dir/compare.log`，上表这一行才对每一次运行都成立；
四个列全部从**本次运行自己产生的**日志解析；日志列只滤掉那行墙钟（它测的是机器，
不是数值），并且改名为 `log_no_timing=`，让"这是过滤后的比较"写在读者真正会看的
列名上；`summary.txt` 头部记下备份后缀（`# backup tag: …`）与比较规则，
`IDENTICAL` 因此可归属、可复算。三条回归测试在
`tests/test_perf_tooling.py`（`…reference_summary_is_derived_from_the_log_this_run_produced`、
`…a_tagged_reference_summary_reproduces_from_its_own_logs`、
`…a_superseded_reference_summary_is_kept_and_says_why`），红跑留档在
`artifacts/perf-round1-rerun/falsification/reference-log-self-comparison.log`。
被取代的两份 `summary.txt` **原样保留**，旁边各放一份
`LOG-COLUMN-CORRECTION.txt` 说明哪一列当时没在测任何东西。

## 三个任务的抓取协议

抓取点按**家族、任务和层数**选择，不再固定在 q_norm 或第 27 层：

| 模型 | 抽样层 | 权重摘要 | tokenizer 样例 | prefill chunk / decode 步 |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 0, 14, 27 | 9 | 4 | 8 / 3 |
| Qwen3-1.7B | 0, 14, 27 | 9 | 4 | 8 / 3 |
| Qwen3-Embedding-0.6B | 0, 14, 27 | 8（无 lm_head） | 2 样例 × 4 文本 | 8 / 0 |
| Qwen2.5-0.5B-Instruct | 0, 12, 23 | 8（含 q_proj.bias） | 4 | 8 / 3 |
| Qwen2.5-Coder-0.5B-Instruct | 0, 12, 23 | 8（含 q_proj.bias） | 4 | 8 / 3 |
| SmolLM2-135M-Instruct | 0, 15, 29 | 7（无 bias、无 q/k norm） | 4 | 8 / 3 |
| SmolLM2-360M-Instruct | 0, 16, 31 | 7（同上） | 4 | 8 / 3 |
| SmolVLM-256M-Instruct | 文本 0, 15, 29；视觉 0, 6, 11 | 22 | 2 | 100 / 3 |

抽样权重张量的选择跟着家族走：qwen3 取 `q_norm` / `k_norm`，qwen2 取 `q_proj.bias`，
llama 两者都不取。**"参数缺失"与"合法的架构默认值"分开处理** —— 前者是错误，
后者记录为该家族的规则。

向量任务的抓取配置里还有一个键 `embedding_input_format`，当前是 **2**。
它存在的理由是：参考侧必须**自己**从原始 query / document、instruction 和角色重建
要编码的字符串，而不是拿 SlowQ 记录的 `text` 去分词 —— 后者会让一个写错的 query
前缀被两侧同时继承，比较照样 `PASS`。格式 1 的 artifact 是"参考侧回显 SlowQ 输入"
那一版留下的，`compare` 会直接拒绝它并说明要重抓哪一侧，不会静默复用：

```
- capture mismatch: embedding_input_format is None in the artifacts but 2 in this
  version of the tool — re-capture the reference
```

这个版本号**只作用于向量任务**，所以没有并进全局的 `artifact_format`（仍是 3）：
格式 3 的文本生成与图文证据继续有效、不需要重抓，只有向量目录需要重跑一次
`ref` 子命令。

每个任务比较的内容：

| 任务 | `compare` 的检查段 |
| --- | --- |
| 文本生成 | `[capture]` `[provenance]` `[tokenizer]` `[weights]` `[numerics]` `[internal]` `[argmax]` |
| 文本向量 | 同上，`[numerics]` 换成向量与相似度，另有 `[embedding]` 段检查输入格式、pooling 位置与向量维度 |
| 图文生成 | 另有 `[internal] 图像特征在融合序列中的位置` 与 `[vision]` 段：预处理、crop 网格、image token 排布 |

数值覆盖 full prefill、chunked prefill、逐 token cached decode 与最终 logits；
图文任务另外逐段验证官方 processor 的 pixel 张量与 mask、视觉塔输出、connector 输出、
图文融合结果与最终 logits。**"生成了一句看起来合理的话"不算数值验证。**

## 容差与实测最差值

跨实现门限（自 v0.1 起未变，也没有为任何失败样例放宽过）：

| 量 | rel_l2 | max_abs |
| --- | --- | --- |
| logits | 1e-3 | 5e-2 |
| hidden states | 1e-3 | 5e-2 |
| 向量 | 1e-3 | 1e-3 |
| 选定权重摘要 | 0（sha256 必须逐字节相同） | 0 |

内部一致性门限更紧，因为分块与缓存只改变**同一实现内部**的 FP32 归约顺序：
`rel_l2 < 1e-5`、`max_abs < 1e-3`。这两个值也是 v0.1 的原值，由
`tests/real_models.py` 的 `assert_same_numerics` 执行，用在真实模型集成测试里；
`compare` 的 `[internal]` 段沿用上表的 logits 门限。
6 个生成 checkpoint 上的实测最差值：logits `rel_l2` 3.916e-06、hidden `rel_l2` 2.940e-06、
`max_abs` 1.755e-04（21~39 token prompt，chunk size 1/3/7/16，另加 cached decode 对长 prefill）。

8 个模型全部 `PASS`，实测最差值见[模型支持 · 参考数值验证结果](models.md#参考数值验证结果)。
最差 rel_l2 落在门限的 1/37 ~ 1/394，最差 max_abs 落在门限的 1/3 ~ 1/202。
最差 max_abs 八个模型都在深层 hidden states 上；最差 rel_l2 有 3 个模型在 logits 上，
SmolVLM 在视觉塔输出上；
所有 greedy argmax 位置（文本生成 6 个模型各 20 个、SmolVLM 10 个）两侧完全一致。

**这一段说的是跨实现（`[numerics]` 段）的最差值。** `compare.log` 里还有一个
`[internal]` 段也印 `rel_l2`，量的是 SlowQ 的分块 prefill 对它自己的整块 prefill，
门限沿用 logits 的 `1e-3`、**不是**上面那个更紧的 `1e-5`。两者本轮相差最多 4 个数量级
（Qwen2.5-0.5B：跨实现 2.719e-05、内部 0.000e+00），引用时必须说清是哪一个，
详见[审查修复后重跑 · 两个「最差 rel_l2」不是同一个量](#两个最差-rel_l2-不是同一个量)。
上面这些数字在 2026-10-23 与 2026-10-26 两次重跑中**逐字节复现**（`slowq.npz` 相同）。

`compare` 不给假 PASS：它检查预期数值项、tokenizer 样例与权重摘要是否齐全，
拒绝形状、dtype、有限性或协议不匹配的数据，缺文件时返回非零退出码。
故障注入用例见 [tests/test_validate_reference.py](../tests/test_validate_reference.py)。

### 发布前修复后重跑参考验证（2026-10-21）

改了参考流程就必须重建**确实受影响**的证据，不能只引用旧 `compare.log` 里的 PASS。
三步分开做，命令与结果如下。

**(i) 只用现有 artifact 重新 `compare`** —— 不重抓任何一侧，检验当前工具怎么判旧证据。
这一步是在重抓**之前**做的，当时 7 个目录 `exit=0`（6 个生成模型各 116 项、SmolVLM 82 项），
`Qwen3-Embedding-0.6B-slowq` `exit=1`，两条问题正是新增的版本拒绝：

```
- capture mismatch: embedding_input_format is None in the artifacts but 2 in this
  version of the tool — re-capture the reference
- the numeric comparison was skipped: the two artifacts do not describe the same
  capture, so there is no agreed key set to grade
```

旧向量证据**没有被静默复用**，也没有被降级成 PASS。

**这条 `exit=1` 现在无法用仓库里的命令重现**：第 (ii)(iii) 步已经把那份格式 1 的
artifact 重抓成格式 2 了，`artifacts/` 又不入库，所以旧证据已经不存在。
下面这条命令现在对 8 个目录全部返回 `exit=0`：

```bash
for d in artifacts/validation/*/; do
  n=$(basename "$d"); [ -f "$d/ref.npz" ] || continue
  python tools/validate_reference.py compare --model "models/$n" >/dev/null 2>&1
  echo "$n exit=$?"
done
```

拒绝格式 1 证据的行为由回归测试固定，可离线重现：
`pytest tests/test_validate_reference.py -k before_version_two`。

**(ii) 用当前代码重抓 NumPy 侧，(iii) 重抓官方参考侧**（`ref` 需要 `.[ref]`，离线跑）：

```bash
M=models/Qwen3-Embedding-0.6B-slowq
H=models/Qwen3-Embedding-0.6B-slowq-hf/Qwen--Qwen3-Embedding-0.6B/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3

python tools/validate_reference.py numpy   --model "$M" --prefill-chunk-size 8   # 8.7 s，82 tensors
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python tools/validate_reference.py ref     --model "$M" --hf-dir "$H" --prefill-chunk-size 8   # 5.3 s，82 tensors
python tools/validate_reference.py compare --model "$M"                          # exit=0
```

结果 `PASS: 82 tensors compared within the declared tolerances.`，
`[capture]` 段出现 `embedding_input_format slowq=2 ref=2 [ok]`，
`[embedding]` 段 8 条文本全部 `input=ok pooled=ok`，0 条 MISMATCH。
容差没有改动，预期张量数没有减少。本轮向量最差值：`rel_l2` 4.935e-06（门限 1e-3）、
hidden `max_abs` 7.629e-04（门限 5e-2）、向量 `max_abs` 5.364e-07（门限 1e-3）。

生成任务没有受这次改动影响，也实测确认了：重抓 SmolLM2-135M 的 NumPy 侧后，
`slowq.npz` 与改动前那份**逐字节相同**（只有 `samples.json` 的 `captured_at` 时间戳不同），
对现有 `ref.npz` 比较仍是 `PASS: 116 tensors`；生成任务的抓取配置里**不含**
`embedding_input_format` 这个键，所以版本升级没有波及它的证据。

**故障注入（真实 artifact 级，不只是单元夹具）**：把 `slowq.embedding.format_query`
换成在 `Query:` 后多一个空格的版本，**只重抓 NumPy 侧**、`reference.json` 保持不动，
`compare` 返回 `exit=1`，23 条问题，头两条直接点名出错的那条文本：

```
- tokenizer sample 'retrieval-with-instruction/query': SlowQ renders different input
  text than the reference did (first difference at character 91)
- tokenizer sample 'retrieval-with-instruction/query': SlowQ produced different token
  IDs than the reference did (31 vs 31 tokens)
```

`[embedding]` 段显示 `retrieval-with-instruction/query … input=MISMATCH pooled=ok` ——
两个判定分开标注，因为 token 数、pooling 位置和维度都没变，只有**编码的字符串**变了。
恢复未打补丁的代码重抓后，`slowq.npz` 与注入前逐字节相同，
`compare` 回到 `PASS: 82 tensors`。

**这一次注入只证明单侧检出**：reference 用的还是修复前抓好的、由官方格式独立生成的
正确证据，坏的只有 NumPy 一侧。它**不能**用来证明"两侧被同一个坏前缀一起污染时会失败"，
而后者才是这条修复要堵的洞。双侧那一半在验证工具修复轮补上，见下面的
[验证工具修复轮](#验证工具修复轮2026-10-21)。

这一步是**手工一次性验证**，用的是一个 monkeypatch `format_query` 后调用
`validate_reference.main(["numpy", …])` 的临时脚本，脚本没有入库，
所以它不像上面的命令那样可以直接重跑。可离线重现的等价断言在仓库里：

```bash
pytest tests/test_validate_reference.py -k "rebuilds_the_query_prefix or broken_query_prefix or document_that_received"
```

三条分别证明：参考侧自己重建前缀、不受被测 `format_query` 影响；
query 前缀多一个空格时 `compare` 返回 1 并点名该文本；
document 被套上 query 前缀时同样失败。

### 验证工具修复轮（2026-10-21）

第二轮审查在验证工具里又找出两个问题。**只改了验证工具、它的测试和文档**：
没有动前向计算、采样行为、数值容差或支持范围，没有重抓任何参考证据，
也没有重跑八模型集成测试。

**问题一：`compare` 对缺失的输入字段给假 PASS。** 向量 artifact 的比较此前用
`ours_entry.get(field) == theirs_entry.get(field)` 判 `raw` / `is_query` / `text`，
于是两侧都缺同一个字段时两边都是 `None`，`None == None` 成立——
**"两边都没说自己嵌入了什么"被判成"两边说的一样"**，打印 `input=ok` 并以退出码 0 通过。
现在逐侧逐字段判"键在不在 / 是不是 null / 类型对不对"，不合格的记 `input=INVALID`、
写进 problems、退出码非 0，错误信息带上哪一侧、哪个样例、哪条文本、哪个字段。
`is_query` 用 `isinstance(value, bool)` 判而不是和 `True` 比相等：Python 里 `bool`
是 `int` 的子类，JSON 的 `1` 会被当成"是 query"。原有的数值、tokenizer、权重摘要、
pooling 与格式版本检查一条没减。

实测（把真实向量 artifact 复制到 `/tmp`，两侧删掉 `raw` / `is_query` / `text` 共 48 个字段）：

| 工具 | 结果 |
| --- | --- |
| 修复前 | 退出码 **0**、`PASS: 82 tensors`、8 行 `input=ok`、0 行 `input=INVALID` |
| 修复后 | 退出码 **1**、`FAIL (48 problem(s))`、0 行 `input=ok`、8 行 `input=INVALID` |

48 = 8 条文本 × 3 个字段 × 2 侧，逐条对得上。**合法证据没有被误伤**：
上面那节里真实的向量 artifact 修复后仍 `PASS: 82 tensors`、8 行 `input=ok`。

修复后还直接跑在**审查方自己的复现证据**上（不是另造一份）：
`artifacts/repair-review-20261021/missing-input-fields-repro/` 就是他们
`review-summary.json` 里 `evidence` 字段指的目录，两侧同样删掉了三个字段。
把它复制到 `/tmp` 之后：

```bash
.venv/bin/python tools/validate_reference.py compare --model missing-input-fields-repro \
    --artifacts /tmp/slowq_rev_repro/missing-input-fields-repro
# exit=1，FAIL (48 problem(s))，0 行 input=ok，8 行 input=INVALID
```

审查方原件的 `compare.log` 跑前跑后 sha256 都是 `3dd007d03bccfdea4368…`，
**一个字节没改**。

> **更正（2026-10-23）**：这里原先写的是"复制出去跑就是为了不让 `compare` 把新日志
> 写进他们的证据目录"。那句是错的：**`compare` 对 artifact 目录只读**。
> 工具里唯一的 4 处写文件全在抓取侧——`_write_numpy_artifacts`
> （`tools/validate_reference.py:1105`）写 `samples.json` / `slowq.npz`，
> `run_reference_side`（`:1681`）写 `reference.json` / `ref.npz`；
> `compare`（分派在 `:3611`）没有 tee、没有 logging handler、没有 stdout 重定向。
> 目录里那些 `compare.log` 是**产生它的命令用 shell 重定向写的**，不是工具写的。
> 实测：`artifacts/repair-review-20261021/compare_readonly_probe.sh`
> 把真实向量 artifact 复制到 `/tmp`，逐文件 sha256 → 跑 `compare` → 再 sha256 → diff，
> **6 个文件哈希全部相同、没有多出 `compare.log`、退出码 0、`PASS: 82 tensors`**。
> 交叉验证仍然复制出去跑，但理由改成"在别人的证据目录上做实验不是取得
> '不会改'这个结论的好地方"，不是"不然会覆盖他们的证据"。
>
> **这条更正后来抓住了一个真缺陷**：正因为 `compare` 不写 `compare.log`，
> `run_reference.sh` 解析的那个 `$dir/compare.log` 就永远是上一次运行留下的，
> 而它据此印出的 `log=IDENTICAL` 是拿该文件与它自己的副本比出来的。
> 详见 §参考验证：三进程协议 里 `compare.log` 那一行的更正，以及
> `artifacts/perf-round1-rerun/reference-{fixed,final}/LOG-COLUMN-CORRECTION.txt`。

**问题二：生产的参考抓取函数此前没有测试真的执行过。**
`_reference_embedding_capture` 是"参考侧自己重建输入"这条修复的核心，
但把它的函数体第一句换成 `raise AssertionError`，当时那 91 项测试全绿。
现在 4 条用例直接调用它：mock 的只是它外围的 torch 接口、
tokenizer（记录被喂了哪些字符串）与一个返回 `(batch, seq, hidden)` 的假模型，
**没有**在测试里另写一份"正确的抓取"；8 条官方字符串逐字写在测试里，
不调用被测的 `reference_embedding_texts`。撤回这个函数（改回读 payload 的
`entry["text"]`）→ **4 failed, 107 passed**；函数体第一句 `raise AssertionError`
→ 同样 **4 failed, 107 passed**。

输入协议的故障注入也随之补上：monkeypatch 把真实 `format_query` 改成多一个空格，
用它构造 SlowQ 侧的输入记录，再执行生产参考抓取函数（外围 torch、tokenizer、模型
使用测试替身），把返回的输入字段写入合成 artifact 后运行 `compare`。
**没有执行生产 NumPy 抓取流程**；token IDs、token 数与数值数组保留合成 fixture 的值，
以隔离输入格式检查。这验证了参考输入的独立性，不是两个真实模型抓取进程的端到端重跑。

| 工具 | `compare` |
| --- | --- |
| 修复版 | 退出码 **1**、`FAIL (2 problem(s))`、7 行 `input=ok` + 1 行 `input=MISMATCH`、指名"冒号后多了一个空格"；被分词的 8 条字符串 **0 条**不是官方文本 |
| 抓取退回读 `entry["text"]` | 退出码 **0**、`PASS: 66 tensors`、8 行全是 `input=ok`、**没有**指名前缀；8 条里 **1 条**不是官方文本，日志打出它继承自 SlowQ 的 `…'swer the query\nQuery: Which planet '` |

仓库内常驻的等价覆盖是
`tests/test_validate_reference.py::test_a_broken_prefix_is_caught_when_the_reference_is_re_captured_by_production`。

**全部负向验证由一条命令重跑**（脚本在 `/tmp` 里生成撤回变体与被删字段的 artifact 副本，
仓库的 `src/` `tools/` `tests/` `artifacts/validation/` 一个字节没改）：

```bash
zsh artifacts/repair-review-20261021/negative-validation/run_all.sh
```

脚本与 9 份日志留在本机 `artifacts/repair-review-20261021/negative-validation/`
（其中 `compare-reviewer-repro-fixed.log` 是上面那条交叉验证，不由 `run_all.sh` 产生）。
`artifacts/` 被 `.gitignore` 忽略，所以它不随仓库分发——换一台机器要先跑
`tools/validate_reference.py numpy` + `ref` 生成向量 artifact，才能重现问题一那两行。

**对 8 份既有 artifact 重新比较**（不是重抓：本轮没跑 `numpy`，也没跑 `ref`）：

```bash
for d in artifacts/validation/*/; do m=$(basename "$d"); \
    python tools/validate_reference.py compare --model "models/$m"; done
```

8 / 8 退出码 0，6 份 `PASS: 116 tensors` + 2 份 `PASS: 82 tensors`，
合计 **860 个张量**，与上一轮同一个数。向量那份额外确认 8 行 `input=ok`、
0 行 `input=MISMATCH` / `input=INVALID`、`embedding_input_format slowq=2 ref=2 [ok]`。
**容差一项没动。**

### 推理加速第 1 轮后重跑参考验证（2026-10-23）

这一轮改的是 prefill 的投影次数、attention 的 mask 与 softmax 的写法，
全都是**会动到浮点归约路径**的改动，所以不能只重新 `compare` 既有 artifact——
必须用改过的代码**重新抓取** SlowQ 侧，再和 2026-10-21 那份固定的官方参考比。
`ref` 侧没有重跑：参考实现（torch / transformers）本轮一行没改，
重跑只会得到同一份 `ref.npz`，而它是比较的锚点，动它就没有意义了。

抓取前先备份，这样"变没变"是可检验的而不是口头保证：

```bash
for d in artifacts/validation/*/; do
  cp -p "$d/slowq.npz" "$d/slowq-pre-round1.npz"
  cp -p "$d/compare.log" "$d/compare-pre-round1.log"
done

for m in SmolLM2-135M-Instruct-slowq SmolLM2-360M-Instruct-slowq \
         Qwen2.5-0.5B-Instruct-slowq Qwen2.5-Coder-0.5B-Instruct-slowq \
         Qwen3-0.6B-slowq Qwen3-1.7B-slowq Qwen3-Embedding-0.6B-slowq; do
  python tools/validate_reference.py numpy   --model "models/$m"
  python tools/validate_reference.py compare --model "models/$m"
done
```

**SmolVLM 必须显式给出它当初被记录下来的协议**，工具默认值跑不动它：

```bash
python tools/validate_reference.py numpy --model models/SmolVLM-256M-Instruct-slowq \
  --max-context 1024 --prefill-chunk-size 100 --decode-steps 3
```

`numpy` 子命令用命令行参数，`compare` 用 `reference.json` 里记录的值，两者不是同一处来源。
按默认 `--max-context 512` 跑会直接失败而不是给出宽松的 PASS：

```
slowq.errors.ContextOverflowError: this image expands to 576 of the prompt's 612 tokens,
which already exceeds max_context=512.
```

这不是本轮引入的，是工具默认值与 SmolVLM 抓取协议本来就不一致；
`--prefill-chunk-size 100 --decode-steps 3` 取自该模型 `reference.json` 的 `capture` 块。
（向量模型的 `decode_steps` 记录为 0，`numpy` 也确实产出 0，所以给它传默认的 3 是惰性的；
`compare` 的 `decode_steps slowq=0 ref=0 [ok]` 一行可以查证。）

结果：**8 / 8 退出码 0**，6 份 `PASS: 116 tensors` + 2 份 `PASS: 82 tensors`，合计 **860 个张量**，
与上一轮同一个数，**容差一项没动**。

比"仍然 PASS"更强的证据是逐字节比较新旧 `compare.log`。`compare` 会打印每个张量的
`max_abs` / `mean_abs` / `rel_L2`，所以两份日志相同就意味着**每一个张量的实测误差都没有变**，
而不只是都落在门限内：

| 模型 | 张量 | 退出码 | 与 pre-round1 的差异行 | 其中非计时行 |
| --- | --- | --- | --- | --- |
| SmolLM2-135M-Instruct | 116 | 0 | 2 | **0** |
| SmolLM2-360M-Instruct | 116 | 0 | 2 | **0** |
| Qwen2.5-0.5B-Instruct | 116 | 0 | 0 | **0** |
| Qwen2.5-Coder-0.5B-Instruct | 116 | 0 | 0 | **0** |
| Qwen3-0.6B | 116 | 0 | 0 | **0** |
| Qwen3-1.7B | 116 | 0 | 0 | **0** |
| Qwen3-Embedding-0.6B | 82 | 0 | 0 | **0** |
| SmolVLM-256M-Instruct | 82 | 0 | 0 | **0** |

那 2 行差异只是末行 `[compare] took 0.0 s` → `0.1 s`，是**比较器自己的耗时**，
不是被比较的数值。八个模型的最差实测值：

| 模型 | 最差 rel_L2 | 最差 max_abs | 最差张量 |
| --- | --- | --- | --- |
| SmolLM2-135M-Instruct | 2.536e-06 | 2.670e-05 | `multi-turn.decode2.logits` |
| SmolLM2-360M-Instruct | 1.195e-05 | 1.562e-02 | `emoji-and-mixed-script.full_prefill.layer31.layer_output[0]` |
| Qwen2.5-0.5B-Instruct | 2.719e-05 | 3.948e-04 | `emoji-and-mixed-script.decode2.logits` |
| Qwen2.5-Coder-0.5B-Instruct | 1.067e-05 | 1.645e-05 | `zh-single-turn.decode0.layer0.layer_output[-1]` |
| Qwen3-0.6B | 7.986e-06 | 1.063e-04 | `zh-single-turn.full_prefill.logits` |
| Qwen3-1.7B | 1.595e-05 | 2.808e-03 | `multi-turn.full_prefill.layer27.layer_output[-1]` |
| Qwen3-Embedding-0.6B | 4.935e-06 | 7.629e-04 | `retrieval-without-instruction.doc2.chunked_prefill.layer27.layer_output[-1]` |
| SmolVLM-256M-Instruct | 8.587e-06 | 2.403e-04 | `grid-1x4.vision.encoder_output[4]` |

最差 rel_L2 2.719e-05 对门限 1e-3 是 **1/37**，最差 max_abs 1.562e-02 对 hidden 门限 5e-2
是 **1/3**，与上面「容差与实测最差值」一节记录的区间（1/37 ~ 1/394、1/3 ~ 1/202）端点一致——
因为数字根本没变。全部 greedy argmax 位置两侧一致，`MISMATCH` 计数为 0。

**为什么数值能一模一样而不是"在容差内漂移"：** 优化 1 省掉的是**从未被读取**的投影
（末位 logits 由末 chunk 的 `forward` 产出，`test_prefill_matches_projecting_every_chunk_exactly`
断言两者逐位相同）；优化 2 的 mask 跳过发生在全零 mask 的情形，减去全零不改变任何值，
`sigmoid` 的改写用的是同一组表达式在同一个输入上求值（含 `-0.0`、`±inf`、`nan`）。
所以这不是"误差小到看不见"，是**同一串浮点运算**。
真正被回退的那个实验（把 softmax 的三个临时数组合并成一个缓冲区）之所以回退，
是因为它实测没有收益，而不是因为它改变了数值。

日志留在本机 `artifacts/validation/<model>/{numpy-round1.log,compare.log,compare-pre-round1.log}`；
`artifacts/` 被 `.gitignore` 忽略，不随仓库分发。

### 审查修复后重跑（2026-10-26）

审查的判词是「这些属于已有证据，不代表修复后的最终代码已经验证。
**新的结论必须关联新的执行记录**」，所以三层验证都用修复后的源码重跑了一遍。
新日志一律写进**新目录**：`artifacts/perf-round1/` 与
`artifacts/validation/*/compare-pre-round1.log` 一个字未改——旧证据保留，
不覆盖之后再假装它不存在。

| 层 | 采集时刻 | 日志位置 | 结果 |
| --- | --- | --- | --- |
| 快速测试 | 2026-10-27 06:01（最终树、§2.2 那句"与 pre-round1 备份逐字节相同"改成从磁盘派生之后重跑；**两步落盘**——先跑到 `/tmp` 再 `cp` 进来，避免复现块测试读到正在被截断的日志，见该测试 docstring） | `artifacts/perf-round1-rerun/quick-fixed.log`（前两代日志留在 `quick-fixed.log.prev-994` 与 `quick-fixed.log.prev-991`） | `995 passed, 62 deselected` |
| 真实模型集成 | 15:24:18（docstring 论证批）；**最终树重跑 10-27 02:43** | `artifacts/perf-round1-rerun/integration-fixed/`；**`integration-final/logs/`** | **208 passed、0 failed**（288 skipped）；最终树重跑同为 **208 passed、0 failed、`OVERALL_RC=0`** |
| 参考验证 | 15:32:00–15:33:25（docstring 论证批）；**最终树重跑 10-27**；**驱动修好后按新格式再跑 10-27 04:57** | `artifacts/perf-round1-rerun/reference-fixed/`；`reference-final/`；**`reference-final2/`** | **8 / 8 退出码 0、860 个张量**、`OVERALL_RC=0`；三次重跑同为 **8/8 rc=0、860 张量**，`reference-final2/` 的 8 份 `slowq.npz` 与 8 份比较日志相对 `pre-final2` 备份**逐字节相同**（`npz=IDENTICAL log_no_timing=IDENTICAL`），且与 `reference-final/` 的每一列共有字段逐个相符 |

产生那一批的命令是（**记录，不是今天可以直接跑的命令**，原因见下）：

```bash
.venv/bin/python -m pytest -q -m "not integration"
bash tools/perf/run_integration.sh artifacts/perf-round1-rerun/integration-fixed
bash tools/perf/run_reference.sh   artifacts/perf-round1-rerun/reference-fixed pre-review
```

**后两条现在会被拒绝，这是 2026-10-28 问题 1 的修复本身。**
`tools/perf/out_guard.py` 把「路径已存在」一律当冲突，而
`artifacts/perf-round1-rerun/` 在 `PUBLISHED_ROOTS` 里——它是已发布文档正在引用的
那一份证据，**没有 `--force`**：

```console
$ .venv/bin/python tools/perf/out_guard.py reserve artifacts/perf-round1-rerun/integration-fixed
out_guard: 拒绝写入，输出目录会与已有证据冲突：
  - artifacts/perf-round1-rerun/integration-fixed：落在已发布证据目录 `artifacts/perf-round1-rerun/` 内
  已有实验结果、历史状态与日志一律不覆盖，也没有 --force：重新测量请换一个新目录（不带参数调用驱动脚本会自动创建一个并打印位置）。
$ echo $?
1
```

今天要重跑，**不带参数**即可——驱动会创建一个唯一的新目录并把路径打到 stderr：

```bash
.venv/bin/python -m pytest -q -m "not integration" -p no:randomly
bash tools/perf/run_integration.sh          # → artifacts/perf-run-<UTC 时间戳>/
bash tools/perf/run_reference.sh "" round3  # 空串=用默认新目录，第二个参数是备份后缀
```

**退出码就是结论**：三个驱动都是每个模型都跑完、然后用累计的失败作为退出码
（原先印一行 `OVERALL_RC=1` 再 `exit 0`，把结论留在状态行、把成功留给退出码）。
解释器缺失也计为失败——`rc=127` 在 pytest 启动之前就退出，原先会被当成
「没有模型失败」。

**执行环境**由两个驱动现场采集，不是事后回忆
（`integration-fixed/environment.txt`、`reference-fixed/source-identity.txt`）：
Darwin 25.6.0 arm64、macOS 26.6.2（build 25G83）、Python 3.13.14、numpy 2.5.4，
BLAS 与 LAPACK 都探测为 Apple `accelerate`；`git` HEAD `8968e39`，工作区 dirty
（24 个已跟踪文件被修改、5 个未跟踪路径）。**机器不是空闲的**：
load average `2.05 2.16 2.35`，`top` 里 CloudShell 占 101.8% CPU、
Codex Renderer 92.2%、一个 Virtualization 虚拟机 11.5%；
同时 `vm.swapusage: total = 0.00M used = 0.00M`。
正确性结论与耗时无关，但**这批记录里的任何耗时都不能与已发布那批直接比大小**。

#### 这两批执行记录早于最终源码，差的是 docstring

集成 15:24、参考 15:32，而 `src/slowq/model.py` 在 **15:35:39**、
`src/slowq/vlm.py` 在 **16:00:48–16:09:22 之间 6 次**、`tests/test_perf_tooling.py`
在 **16:13:15** 又被改过。逐条核对后，那 7 次 `src/` 编辑**全部落在 docstring 里**：
`model.py` 那次是把「低于本机实测的 3% 跨进程噪声底线」改写成
「与跨进程离散度同量级，判读为尚不能确认收益或回退」（即 §3 的撤回），
`vlm.py` 那 6 次是把身份检查与像素哈希的单价换成实测区间（优化 4）。
`vlm.py` 的**代码哈希**——把语法树里的 docstring 清空后再算——在 16:06 的探针产物里
记为 `77167e65…`，与当前源码一致，所以那 6 次编辑没有动到代码。
用 `source-identity.txt` 里 15 个运行时文件的 sha256 逐个重算比对，
15:32 之后**只有 `model.py` 与 `vlm.py` 两个文件的原始哈希变了**，其余 13 个未变。

反过来，**当时** `find src tools tests docs -newermt '2026-10-26 16:20:15'` 为空，
说明 16:17:18 的快速测试与 16:20:15 启动的性能重采跑在**同一棵最终树**上
（重采的源码身份摘要 `32fcc329fa440d9bf39ee34bc560664f74a87414a8211e362e4b768a39e92bc7`）。

**这条 `find` 现在不再为空，别再拿它当证据。** 10-27 02:34–02:40 那一轮就地注入的
「改坏应变红」把 `embedding.py`／`model.py`／`vlm.py`／`engine.py` 的 **mtime** 顶掉了
（内容用 `cp -p` 还原、`cmp` 逐个确认逐字节相同——`cp -p` 还原的是**备份的** mtime，
不是原来那个），其后 `tools/perf/make_report.py`、`tests/test_perf_tooling.py`、
`tools/perf/README.md` 与 `docs/` 又因 §2.2 那处逐字节断言的更正**真的改过**。
**站得住的是哈希不是 mtime**：那个摘要同时覆盖 `src/` 与 `tools/perf/` 两半，
整树摘要因此已经复现不出来（本文写这句话时是 `70b9670bdb29…`），
但 **`src/` 那 15 个逐文件哈希与重采时记下的那份 15/15 相符**——计时数据对应的正是
这一半，采集它的 harness 一个字没动。自己核一遍，不要相信这句话：

```bash
cd "/Users/lilele/Desktop/慢炖 · SlowQ"
.venv/bin/python - <<'PY'
import json, pathlib, subprocess
cur = json.loads(subprocess.run(
    [".venv/bin/python", "tools/perf/source_identity.py", "--src", "."],
    capture_output=True, text=True, check=True).stdout)["source"]
rec = json.loads(pathlib.Path(
    "artifacts/perf-round1-rerun/ab/Qwen3-0.6B-slowq-opt.json"
).read_text())["source_identity"]["source"]
diff = [k for k in sorted(set(cur) | set(rec)) if cur.get(k) != rec.get(k)]
print(f"src/ 逐文件相符: {cur == rec}  ({len(rec)} 个文件)")
print("不同的文件:", diff or "无")
PY
```

期望 `src/ 逐文件相符: True  (15 个文件)` 与 `不同的文件: 无`。
权威文档 §1.5 对三棵被测树逐棵给出同样结论（`src/` 15/15 相符，变的只是工具那一半）。

所以 15:24 / 15:32 那两批记录**描述的就是最终代码的运行时行为**。
性能采集结束后**已按最终树把集成与参考验证各重跑一次**——不是为了改变结论，
是为了让时间戳自己说话，不必依赖上面这段论证：

- **集成**（`artifacts/perf-round1-rerun/integration-final/logs/`，10-27 02:43）：
  8 个模型 **208 passed、0 failed、`OVERALL_RC=0`**，逐模型 25–27 passed，
  与 15:24 那批**逐项相同**。
- **参考验证**（`artifacts/perf-round1-rerun/reference-final/`）：
  **8/8 `numpy_rc=0 compare_rc=0`、860 个张量、`OVERALL_RC=0`**；8 份 `slowq.npz`
  与 `compare.log` 相对重采前备份（后缀 `pre-final`，即 15:32 那批）**逐字节相同**
  （`npz=IDENTICAL log=IDENTICAL`），即最终树原样复现参考。
  **这一遍的 `log=IDENTICAL` 后来被判定为无效证据**（那一列拿
  `$dir/compare.log` 与它自己的副本比，不可能失败）；`npz=IDENTICAL` 仍然成立，
  它比的是两次不同抓取产生的两个 inode。详见 §参考验证：三进程协议 的更正，
  以及该目录下的 `LOG-COLUMN-CORRECTION.txt`。
- **驱动修好后按新格式再跑一遍**（`artifacts/perf-round1-rerun/reference-final2/`，
  10-27 04:57，后缀 `pre-final2`）：同样 **8/8 rc=0、860 张量、`OVERALL_RC=0`**。
  8 份 `slowq.npz` 相对 `pre-final2` 备份**逐字节相同**（`cmp` 逐个确认，不是从
  汇总表的 `IDENTICAL` 读出来的）。日志一侧要分开说：8 份里 **7 份逐字节相同**，
  SmolLM2-135M 那份差**恰好一行**——`[compare] took 0.0 s` vs `0.1 s`，即墙钟，
  与上一代 `pre-final` 比较时差的是同一行同一个模型（该模型一次 `compare` 的墙钟
  本来就在 0.1 s 量级，印到一位小数就在 0.0／0.1 之间摆动）。滤掉 `[compare] took ` 之后 **8/8 相同**，
  这就是 `log_no_timing=IDENTICAL` 的含义，也是它为什么必须叫这个名字而不能叫
  `log=`。把两份 `summary.txt` 的 `log=`／`log_no_timing=` 列去掉后 `diff`，
  **其余每一列逐个相符**。这一遍的四个列全部从**本次运行自己产生的**日志解析，
  头部记下备份后缀。

三遍都用**独立输出目录**（`integration-final/`、`reference-final/`、
`reference-final2/`）与**各自的新备份后缀**（`pre-final`、`pre-final2`），旧的
`integration-fixed/`、`reference-fixed/`、`reference-final/` 与更早的
`pre-review`／`pre-round1` 备份都保留未覆盖，所以「重跑前 vs 重跑后是否一致」
能从字节比出来，而不是靠记忆。参考验证那两遍还顺带修了 `run_reference.sh` 的
两个缺陷：把跨实现 `[numerics]` 与内部 `[internal]` 两个 rel_L2 混成一列，
以及上面那个 `log` 列自比（均详见 changelog）。

#### 参考验证：五代 npz 逐字节相同

抓取前先备份，`run_reference.sh` 的第二个参数就是备份后缀，
且**只在备份不存在时才写**（重跑驱动不会用"审查后"的版本顶掉"审查前"的基线）。
于是同一个模型目录下现在有五代 `slowq.npz`：

| 代 | 文件名 | 抓取时间 | 对应源码 |
| --- | --- | --- | --- |
| 1 | `slowq-pre-round1.npz` | 2026-10-21 00:17:36 | 推理加速第 1 轮**之前** |
| 2 | `slowq-pre-review.npz` | 2026-10-23 21:07:56 | 第 1 轮**之后**、审查修复**之前** |
| 3 | `slowq-pre-final.npz` | 2026-10-26 15:32:10 | 审查修复**之后** |
| 4 | `slowq-pre-final2.npz` | 2026-10-27 02:49:44 | 性能重采 + 最终树重跑**之后** |
| 5 | `slowq.npz` | 2026-10-27 04:57:01 | `run_reference.sh` 的 `log` 列缺陷修好**之后**（本节） |

（时间取自 Qwen3-0.6B，八个模型的五代顺序相同。）五代的 **inode 互不相同**
（该模型是 `78241413` / `82256335` / `83374176` / `83702986` / `72352198`），
所以 `cmp` 报出的 `IDENTICAL` 不是拿一个文件跟它自己比。
复核方式不是读汇总表，而是逐模型 `shasum -a 256 slowq*.npz`：
**八个模型各 5 个文件、合计 40 份，每个模型只出现 1 个不同的哈希。**

| 模型 | `slowq.npz` 字节 | npz vs 第 2 代 | npz vs 第 1 代 | `compare.log` vs 第 2 代 | vs 第 1 代 |
| --- | --- | --- | --- | --- | --- |
| SmolLM2-135M-Instruct | 3672443 | **相同** | **相同** | 相同 | 只差计时行 |
| SmolLM2-360M-Instruct | 3949317 | **相同** | **相同** | 相同 | 只差计时行 |
| Qwen2.5-0.5B-Instruct | 11405332 | **相同** | **相同** | 相同 | 相同 |
| Qwen2.5-Coder-0.5B-Instruct | 11204459 | **相同** | **相同** | 相同 | 相同 |
| Qwen3-0.6B | 11422233 | **相同** | **相同** | 相同 | 相同 |
| Qwen3-1.7B | 11746361 | **相同** | **相同** | 相同 | 相同 |
| Qwen3-Embedding-0.6B | 332479 | **相同** | **相同** | 相同 | 相同 |
| SmolVLM-256M-Instruct | 17561422 | **相同** | **相同** | 相同 | 相同 |

「只差计时行」是逐行 `diff` 出来的，两个模型各 1 处，内容分别是
`[compare] took 0.0 s` ↔ `0.1 s`——**比较器自己的耗时**，不是被比较的数值。

> **上表 `compare.log` 那两列现在只能当线索，不能当证据。** 第 3 代（`reference-final/`）
> 由修复前的 `run_reference.sh` 产生，而那个驱动**从不写** `$dir/compare.log`：
> 它重定向到 `$OUT/<model>-compare.log`。所以做这张表时磁盘上的 `compare.log`
> 还是 15:32 那一代（`compare-pre-final.log` 与 `compare-pre-final2.log` 的 mtime
> 都是 `2026-10-23 21:07:56`，即 `cp -p` 保留下来的源文件时间，两代备份其实是
> 同一份内容的两次拷贝）。"vs 第 2 代 = 相同"因此和那个 `log=IDENTICAL` 一样，
> 是拿一个文件跟它自己的副本比出来的。**vs 第 1 代那列不是自比**——它比的是
> 10-23 与 10-21 两个真正不同的世代，"只差计时行"当时是真的；但它描述的是
> **做表那一刻磁盘上的那一代日志**，第 5 代重写 `compare.log` 之后要重算，
> 重算结果见本节下面「vs 第 1 代」那一段。
>
> 这一列已由第 5 代重新建立：`reference-final2/` 的驱动自己 `cp` 了
> `$dir/compare.log`（Qwen3-0.6B 的 mtime 现在是 `2026-10-27 04:57:01`，inode 仍是
> `72352093`，即在原文件上重写），逐模型 `cmp` 的结果是 **7/8 份与第 4 代
> 逐字节相同**，SmolLM2-135M 那份差**恰好一行**计时（`0.0 s` ↔ `0.1 s`），
> 滤掉 `[compare] took ` 之后 **8/8 相同**。

**「vs 第 1 代」那列同样要按第 5 代重算，重算结果与上表不同。**
上表那两格是拿 15:32 那一代日志与 `compare-pre-round1.log` 比出来的（6/8 原始相同）。
现在磁盘上的 `compare.log` 已是第 5 代，重新逐模型 `cmp`／`diff`：

| 比较 | 原始逐字节 | 滤掉 `[compare] took ` 行 |
| --- | --- | --- |
| 第 5 代 vs 第 1 代（`compare-pre-round1.log`） | **7/8** | **8/8** |
| 第 4 代 vs 第 1 代 | 6/8 | 8/8 |
| 第 3 代 vs 第 1 代 | 6/8 | 8/8 |
| 第 2 代 vs 第 1 代 | 6/8 | 8/8 |

第 5 代只剩 **SmolLM2-360M-Instruct** 一份不同，差异是**恰好一行**
`[compare] took 0.1 s` ↔ `[compare] took 0.0 s`（`diff` 输出 `185c185`，两侧都是
188 行）；SmolLM2-135M 在这一代回到了与第 1 代逐字节相同。

**"墙钟卡在 0.0／0.1 的舍入边界上"这句话现在是观测，不是推测**——把第 185 行
逐代抄出来就是证据（两个模型都在 `185c185` 上、都只差这一行）：

| 模型 | 第 1 代 | 第 2／3／4 代 | 第 5 代 |
| --- | --- | --- | --- |
| SmolLM2-135M-Instruct | `took 0.0 s` | `took 0.1 s` | `took 0.0 s` |
| SmolLM2-360M-Instruct | `took 0.1 s` | `took 0.0 s` | `took 0.0 s` |

同一个模型在不同世代里印过 `0.0` 也印过 `0.1`，说明真值就在 `0.05 s` 附近、
每一次运行自己决定它落在哪一边；SmolLM2-135M 因此**翻回去过两次**。
另外六个模型的四份备份（第 1–4 代）**逐字节全同**，第 5 代也与第 1 代全同——
**跨世代唯一会动的东西就是这一行秒表**，这正是「逐字节相同」必须带世代、带计数、
带例外的原因：它是一个关于文件字节的断言，而文件里含有一只秒表。
`slowq.npz` 那半句不受影响，五代 8/8 全部逐字节相同。

复算命令（不读任何汇总表；第一段是逐模型判定，第二段把上面那张逐代表打出来，
列序为 第 1／2／3／4／5 代。**不要用固定行号取那行秒表**——八个模型的日志行数不同，
`185` 只对两个 SmolLM2 成立）：

```bash
for d in artifacts/validation/*/; do
  cmp -s "$d/compare.log" "$d/compare-pre-round1.log" \
    && echo "$(basename "$d") raw=IDENTICAL" || echo "$(basename "$d") raw=DIFFERS"
  diff "$d/compare-pre-round1.log" "$d/compare.log"
done

for d in artifacts/validation/*/; do
  printf '%-34s' "$(basename "$d")"
  for g in pre-round1 pre-review pre-final pre-final2; do
    printf ' %s' "$(grep -h '^\[compare\] took ' "$d/compare-$g.log" | sed 's/.*took //')"
  done
  printf ' %s\n' "$(grep -h '^\[compare\] took ' "$d/compare.log" | sed 's/.*took //')"
done
```

[推理加速第 1 轮实测](推理加速-第1轮-2026-10-23.md) §2.2 那行原来写的是
「新 `compare.log` 与 pre-round1 备份**逐字节相同**」——**那句已按上表更正**，
现在由 `make_report.py` 的 `reference_identity_cell()` 从磁盘算出，
印成「`slowq.npz` … **8/8 逐字节相同**；`compare.log` **7/8 逐字节相同**——
SmolLM2-360M-Instruct-slowq 差**恰好 1 行**…滤掉该行后 **8/8**」。
守着它的是 `tests/test_perf_tooling.py` 的
`test_the_pre_round1_identity_row_counts_the_models_it_actually_covers`
（红跑：`falsification/preround1-blanket-byte-identity.log`，**不需要注入**，
缺陷就是生成器当时印的那句字面量）。

`slowq.npz` 逐字节相同意味着**SlowQ 侧输出的每一个张量的每一个字节都没变**，
而 `ref.npz` 本轮没有被重跑（参考实现一行没改，它也是比较的锚点）。
两侧都没动，所以「8 个模型的最差实测值」表**按构造**与上一节那张表相同——
这比重新算一遍再声明相等更强。仍然独立算了一遍作为交叉核对：
从新日志的 `[numerics]` 段按第 4 列（`rel L2`）取每个模型的最大值，
八个模型的最差 rel_L2、对应 max_abs 与**张量名**都与上一节的表逐项一致
（最差 rel_L2 仍是 Qwen2.5-0.5B 的 `emoji-and-mixed-script.decode2.logits` 2.719e-05，
最差 max_abs 仍是 SmolLM2-360M 的
`emoji-and-mixed-script.full_prefill.layer31.layer_output[0]` 1.562e-02）。

其余可核对项：8 份日志各有一行
`[declared tolerances] rel_l2_logits=0.001, max_abs_logits=0.05, rel_l2_hidden=0.001,
max_abs_hidden=0.05, rel_l2_vector=0.001, max_abs_vector=0.001, rel_l2_weights=0,
max_abs_weights=0`——**容差一项没动**；
`MISMATCH` 与 `INVALID` 在 8 份日志里合计 **0 处**；
向量模型 8 行 `input=ok`、`embedding_input_format slowq=2 ref=2 [ok]`、
`decode_steps slowq=0 ref=0 [ok]`；SmolVLM 的
`prefill_chunk_size slowq=100 ref=100`、`decode_steps slowq=3 ref=3`。

**为什么修复能不动一个字节：** 四处修复都不在数值路径上。
`DocumentIndex` 加的是返回前的拷贝（拷的是同一批 float32）；
`VisionEngine._model_identity_changed` 只决定**查不查缓存**，命中与否算出的是同一份特征；
`prefill()` 的输入校验前移是在**拒绝**非法输入，合法输入走的运算一步没变；
`Session._run` 改的是 `turns` 何时自增。
「逐位相同」这个结论是拿 `np.load` 出来的**原始数组**比字节得到的，
不是从日志打印的聚合误差推出来的——聚合误差相同不能证明张量逐位相同。

#### 两个「最差 rel_l2」不是同一个量

`compare.log` 里有两段都印 `rel_l2`，**维度不同，不能混用**：

| 段 | 比的是 | 门限 | 本轮八个模型的最差值 |
| --- | --- | --- | --- |
| `[numerics] N expected tensors, SlowQ vs Transformers` | **跨实现**，逐张量，空白分隔四列（`max abs` / `mean abs` / `rel L2` / `argmax`） | logits 与 hidden 各 `rel_l2 1e-3`、`max_abs 5e-2` | **2.719e-05**（Qwen2.5-0.5B） |
| `[internal] SlowQ full prefill vs chunked prefill` | **SlowQ 对自己**，逐样本，带字面量 `rel_l2=` 前缀 | 沿用 logits 门限 `1e-3`（`validate_reference.py` 的 `check_internal`，注释写明分块会改归约顺序所以不要求逐位相同） | **1.750e-05**（Qwen3-1.7B） |

`tools/perf/run_reference.sh` 汇总时 `grep -o 'rel_l2=[0-9.e+-]*'`，
**只匹配得到第二段**，所以 `reference-fixed/summary.txt` 那一列
`worst_rel_l2` 是内部一致性的值：Qwen2.5-0.5B 印 `0.000e+00`
（它的分块 prefill 与整块逐位相同），而该模型跨实现的最差 rel_L2 是 `2.719e-05`。
两个量差 4 个数量级，列名却一样。**性能采集期间没有改**：`source_identity.tree_hashes`
覆盖 `tools/perf/*.{py,sh}`，采集中途改工具会让各步记录的 tooling 摘要不一致。
采集全部 `rc=0` 之后已拆成 `worst_numerics_rel_l2` 与 `worst_internal_rel_l2`
两列并重跑（`reference-final/`）；本节上面所有引用都取自 `[numerics]` 段。

**同一列还藏着第二个缺陷，是拆完列之后才发现的**：`grep -o` 那一步取的日志根本不是
本次运行产生的那份，而 `log=IDENTICAL` 是拿它与自己刚 `cp` 出来的备份比的。
两个缺陷叠在一起时，`reference-fixed/summary.txt` 里 Qwen2.5-0.5B 那一行印的
`0.000e+00` 既是**错的量**（内部一致性，不是跨实现），又来自**错的世代**。
修好后重跑在 `reference-final2/`，四个列全部由本次运行自己的日志解析，
`summary.txt` 头部记下备份后缀与比较规则；两份被取代的 `summary.txt` 原样保留，
旁边各放一份 `LOG-COLUMN-CORRECTION.txt` 说明哪一列当时没在测任何东西。

`[internal]` 段与「容差与实测最差值」一节里那个更紧的
`rel_l2 < 1e-5` **不是一回事**：后者由 `tests/real_models.py` 的 `assert_same_numerics`
在真实模型集成测试里执行，`compare` 的 `[internal]` 段用的是 `1e-3`。
Qwen3-1.7B 的 `1.750e-05` 因此不违反任何门限，但也**不能**拿去和 `1e-5` 那条比。

#### 已补（原「待补」两项均已完成）

- **快速测试已落盘、逐文件收集数已重算。**
  `.venv/bin/python -m pytest -q -m "not integration"` 在最终树上跑出
  **991 passed, 62 deselected**，写入 `artifacts/perf-round1-rerun/quick-fixed.log`；
  §快速测试 的增量表按 `--collect-only -q` 逐文件重算为 826 → **991**（+165），
  收集总数 888 → **1053**。当时故意不在性能采集期间跑（一个约 10 s 的全量 pytest
  会和 `core_ab.py` 抢 CPU），采集结束后才补跑，所以它跑在最终树上、与计时不冲突。
  **此后又给 `run_reference.sh` 的 `log` 列自比缺陷补了 3 项测试**（都在
  `test_perf_tooling.py`，131 → 134），同一棵树重跑得到 **994 passed,
  62 deselected**，增量表与收集总数已按 826 → **994**（+168）、888 → **1056**
  重算；上面那个 991 是**那一次**的结果，日志留在 `quick-fixed.log.prev-991`。
  **再此后又给 §2.2 那句"与 pre-round1 备份逐字节相同"补了 1 项测试**
  （134 → 135），同一棵树重跑得到 **995 passed, 62 deselected**，
  增量表与收集总数已按 826 → **995**（+169）、888 → **1057** 重算，
  即本文 §快速测试 现在印的那一组；994 那一代日志留在 `quick-fixed.log.prev-994`。
  三个计数各自对应**那一次**的树，日志三代都留着，没有互相覆盖。
- **性能重采已完成。** `run_recollection.sh`（16:20:15 启动）五个模型全部 `rc=0`，
  原始 JSON 落在 `artifacts/perf-round1-rerun/`。本文与
  [推理加速第 1 轮实测](推理加速-第1轮-2026-10-23.md) 里的端到端倍率
  **现在由重采后的数据派生**（那份文档整篇由 `make_report.py --data artifacts/perf-round1-rerun`
  生成），不再只描述修复前那棵树的观察值；每棵被测树的源码身份见该文档 §1.5。

### 二次审查修复后重跑（2026-10-28）

四个问题（复现脚本覆盖证据、视觉缓存身份漏掉「同权重不同配置」、剔除异常轮次后
未重算 AB/BA 均衡、验证失败仍返回 0）修好之后，用**当前树**重跑了真实模型集成与
参考验证。快速层同样在最终树上重跑过，**1042 passed, 62 deselected, rc=0**，
日志 `artifacts/perf-run-20261028-185623/quick-fixed.log`，
逐文件增量见 §快速测试；四个修复各自的反证（回退→变红→恢复）见
§反证：四个问题各自的最小回归。
两个驱动都由一个编排脚本串起来跑，**串行**，输出目录是 `out_guard`
新建的 `artifacts/perf-run-20261028-185623/`：

```console
=== start 2026-10-28T10:56:44Z ===
--- run_integration.sh artifacts/perf-run-20261028-185623/integration-round2
...
OVERALL_RC=0
INTEGRATION_RC=0
--- run_reference.sh artifacts/perf-run-20261028-185623/reference-round2 round2-fix
...
OVERALL_RC=0
REFERENCE_RC=0
=== end 2026-10-28T12:07:08Z ===
```

（原文见 `artifacts/perf-run-20261028-185623/driver.log`。）

**集成：8/8 通过，208 passed、0 failed、`INTEGRATION_RC=0`。** 逐模型
（`integration-round2/<model>.log`）：Qwen3-0.6B 27 passed / 35 skipped（19.55 s）、
Qwen2.5-0.5B 26/36（18.17 s）、SmolLM2-135M 26/36（5.49 s）、
**SmolVLM-256M 26/36（35.80 s）**、Qwen3-Embedding-0.6B 25/37（11.57 s）、
Qwen2.5-Coder-0.5B 26/36（15.42 s）、**Qwen3-1.7B 26/36（3935.49 s，见下）**、
SmolLM2-360M 26/36（11.40 s）。问题 2 改的是 `vlm.py`，
**SmolVLM-256M 就是它唯一能跑到的真实模型**，那一格 26 passed 是这次重测的重点。

**这一批日志里有一处不一致，写下来而不是抹平：**八个模型的 `deselected` 计数不是同一个数——
前七个都是 **1034 deselected**，最后跑的 SmolLM2-360M 是 **1042 deselected**
（`grep -ho "[0-9]* deselected" artifacts/perf-run-20261028-185623/integration-round2/*.log | sort | uniq -c`
→ `7 × 1034`、`1 × 1042`）。也就是说**这批集成测试跑的过程中，快速层的收集数长了 8 项**
（当时正在编辑 `tests/test_perf_tooling.py`），整批不是在同一棵树上跑的。
它**不影响这批集成结论**，理由不是「差得少」而是「差的那一半根本没被收集」：
集成驱动带 `-m integration`，`tests/test_perf_tooling.py` 全部落在快速层、被 deselect 掉，
所以它的增删只改 `deselected` 那一格，改不到 `passed`；八个模型的 **208 passed / 0 failed
是逐项相同的**，而集成测试真正 import 的 `src/slowq/` 15 个模块由
`reference-round2/source-identity.txt` 记录、与当前树**逐字节相同**（见下）。
**受影响的只有「这批日志能不能当作快速层计数的证据」——不能**，
快速层的权威计数以本轮结束时那次单独、干净的重跑为准（见 §快速测试）。

**参考验证：8/8 `numpy_rc=0 compare_rc=0`、860 个张量、`REFERENCE_RC=0`。**
`summary.txt` 里八个模型全部 `npz=IDENTICAL log_no_timing=IDENTICAL`
（备份后缀 `round2-fix`）。最差值：

| 模型 | 张量 | worst_numerics_rel_l2 | worst_internal_rel_l2 |
| --- | --- | --- | --- |
| SmolVLM-256M-Instruct | 82 | 8.587e-06 | 3.126e-06 |
| Qwen3-Embedding-0.6B | 82 | 4.935e-06 | 4.061e-06 |
| SmolLM2-135M-Instruct | 116 | 2.536e-06 | 1.078e-06 |
| Qwen3-0.6B | 116 | 7.986e-06 | 7.090e-06 |
| Qwen2.5-Coder-0.5B-Instruct | 116 | 1.067e-05 | 0.000e+00 |
| SmolLM2-360M-Instruct | 116 | 1.195e-05 | 3.645e-06 |
| Qwen3-1.7B | 116 | 1.595e-05 | 1.750e-05 |
| Qwen2.5-0.5B-Instruct | 116 | 2.719e-05 | 0.000e+00 |

**两列各有各的门限，都不是 2e-4**（本项目没有一个"统一容差"，各场景的门限见
[容差与实测最差值](#容差与实测最差值)）：`worst_numerics_rel_l2` 是跨实现比较，
门限 `rel_l2 < 1e-3`，上表最差 2.719e-05 落在门限的 **1/37**；
`worst_internal_rel_l2` 是 `compare` 的 `[internal]` 段——SlowQ 的分块 prefill 对它
自己的整块 prefill——门限沿用 logits 的 `1e-3`，上表最差 1.750e-05 落在 **1/57**。
两个 `0.000e+00` 不是"小到印不出来"，是分块与整块**逐位相同**
（Qwen2.5-0.5B-Instruct 与 Qwen2.5-Coder-0.5B-Instruct）。
另外那个更紧的内部一致性门限 `rel_l2 < 1e-5` / `max_abs < 1e-3`
由 `tests/real_models.py` 的 `assert_same_numerics` 执行，用在真实模型集成测试里，
**不是**上表这两列的门限；而 `2e-4` 是 `tests/test_model.py` 与 `tests/test_session.py`
里 `CHUNK_TOLERANCE` 的 `rtol`，只用于微型合成模型上的分块一致性单元测试，
与上表的真实模型 860 项无关。**容差一项没有放宽过。**
`npz=IDENTICAL` 是拿 `artifacts/validation/<model>/slowq.npz` 与
`slowq-round2-fix.npz` 比**原始字节**，不是比日志里印的聚合误差；
`log_no_timing=IDENTICAL` 是两侧都去掉 `[compare] took <n> s` 那一行之后比，
那一行测的是机器不是数值，所以这一列的名字里就带着 `no_timing`，
不让人把过滤后的比较读成逐字节比较。

**这一批跑的是哪棵树，是算出来的，不是记下来的。**
`artifacts/perf-run-20261028-185623/reference-round2/source-identity.txt`
（JSON，尽管扩展名是 `.txt`）现场记录了 15 个 `src/slowq/*.py`
与 20 个 `tools/perf/*` 的 sha256。把那份记录逐文件与**当前**工作区比：
**35 个里 34 个逐字节相同**，唯一不同的是 `tools/perf/make_report.py`
（记录值 `1f47999a…` → 现在 `92061cd0…`）——那是报告生成器，本轮在那次运行之后又改了
（`--paired-data`、「测得稳定收益」档位的取消、以及 §10 期望值改从新运行根读取），
**它不是被测运行时的一部分**，集成测试与参考验证的数值都不经过它。
即：**`src/slowq/` 的 15 个模块与当前树逐字节相同**，
上面那 208 passed 与 8/8 IDENTICAL 描述的就是现在这份代码。

**Qwen3-1.7B 的 3935.49 s 与宿主机内存压力同时出现；本轮没有把它归因为确定原因。**
同一棵树、同一批命令里其余七个模型都在 5.5–35.8 s，只有它 65 分钟。
**这一格在历史记录里本来就是双峰的**：快的三次是 **57.17 s**（2026-10-21）、
**58.74 s**（`artifacts/perf-round1-rerun/integration-final-driver.log:28`）、
**115.37 s**（2026-10-23 opt12 轮）；慢的两次是 **3828.08 s** 与 **4036.71 s**，
而本文 §「Qwen3-1.7B 的异常耗时与宿主机内存压力相符」已经把那两次归到同一个原因上。
**本次 3935.49 s 落在那两个慢值的区间里**，不是一个新的量级——
所以这里说的不是「本轮把它测慢了」，是「本轮也没能把它测快」。现场证据：`top` 显示 PhysMem 23G/24G 已用、
压缩器 7.8–11G、未用 86–149M，`com.apple.Virtualization` 7149M、
Codex Renderer 1132M、AliLangClient 876M、WindowServer 813M、DingTalk 731M、
CloudShell 556M；CPU 90% 空闲，swapins 19641 / swapouts ~104728。
**并且做了标定，标定只支持它支持的那一句**：同一时段一个 SmolLM2-135M 的
`core_ab.py` 进程实测 **28.001 s**，历史中位 27.2 s——即那个窗口里
**小模型的计时没有被打折**，所以这不是「整机所有测量都同样变慢」。
这与「被拖慢的是把 Qwen3-1.7B 那 6.4 GiB 权重装进内存这件事」相符，
但**相符不等于归因**：内存压力是被观测到的，没有被操纵过，
也没有那次运行窗口内的时间序列（`driver.log` 只有起止时间戳与逐模型标题）。
要把它变成结论，得在测量窗口内同步采 `vm_stat`／压缩器字节数／RSS 时间序列，
或者主动施加内存压力做受控对照——本轮都没有做。
**正确性结论与耗时无关**（26 passed 是 26 passed），
但这一批里的任何耗时都不能与之前那批直接比大小。

**问题 3 的配对重测是另一次运行**，不在上面这个目录里：
`artifacts/perf-round2-paired/`，由
`bash tools/perf/run_noise_floor.sh artifacts/perf-round2-paired 6` 产生
（5 个模型 × 6 轮 × base/opt 两侧 = 60 个独立进程，AB/BA 交替，模型为主循环）。
判读结果写在
[推理加速第 1 轮实测](推理加速-第1轮-2026-10-23.md) §7.2/§7.3，
那一节由 `make_report.py --paired-data artifacts/perf-round2-paired` 派生。
**已发布的 `artifacts/perf-round1-rerun/noise-floor/` 一份未改**——
它是被撤回的「70/73」背后的原始证据，重测加在它旁边，不替换它。

**重测判读结果（60/60 个进程写完、`NOISE_RC=0`，`--paired-data` 重生成两遍字节相同）：**
73 个（模型 × 量）里 **64 个达到「有限轮次方向一致」判据**、
**5 个 `尚不能确认收益或回退`**（逐轮增益跨过 1.00，全部是 Qwen3-1.7B）、
**4 个 `顺序不均衡或未记录，尚不能确认收益或回退`**、**0 个 `有效轮次不足`**。
那 4 个正是问题 3 要抓的那一类：**采集时 6 轮是均衡的（3 先 base、3 先 opt），
污染判据剔掉 1–2 轮之后才不均衡**（保留 5/6 时 base 先跑 3 或 2、保留 4/6 时 base 先跑 3），
上一版对它们印「顺序均衡：是」并把增益计进分子，现在一律不判方向、不计入分子，
弃用的轮次号与理由（spike／level）逐条列在 §7.3。
**采集时的均衡是逐模型核过的，不是从驱动的设计推出来的**：
5 个模型各 6 轮，每轮两个进程的 `label` 都带 `-pos1`／`-pos2`，
`base_first` 各为 3，交替模式 `B O B O B O`。
被撤回的「70/73 达到稳定收益判据」由 **64/73 达到「有限轮次方向一致」** 取代；
「稳定收益」这个档位本身已取消，判据四条见 §7.3。

### 反证：四个问题各自的最小回归（2026-10-28）

**每个修复都先被回退、证明对应的测试会变红，再恢复。**
脚本与全部日志在 `artifacts/perf-round2-falsification-20261028-233602/`
（脚本副本 `falsify_round2.sh` 与日志放在一起）；
每次回退都用**文本锚点**而不是行号（锚点必须在文件里**唯一**，否则脚本直接失败退出），
每次恢复都用 `shasum -a 256` 前后比对，逐条记在 `restore-hashes.txt`。

| 日志 | 回退了什么 | 选择器选中 | 结果 |
| --- | --- | --- | --- |
| `p1-out-guard.log` | `out_guard._conflict()` 开头插 `return None`（防护完全失效） | 21 | **9 failed**, 12 passed |
| `p2c-both-halves.log` | `_model_identity_changed` 的**两个**条件同时回退到「只比权重指纹」 | 12 | **4 failed**, 8 passed |
| `p3-order-balance-final.log` | 删掉 `INSUFFICIENT`／`UNORDERED` 两个改判分支 | 8 | **6 failed**, 2 passed |
| `p4-integration-rc.log` | `run_integration.sh` 的 `exit $FAIL` → `exit 0` | 4 | **2 failed**, 2 passed |
| `p4-recollection-rc.log` | `run_recollection.sh` 的 `exit $OVERALL` → `exit 0` | 3 | **2 failed**, 1 passed |
| `green-targeted-final.log` | 什么都不回退（修复在位） | 287 | **287 passed** |

**问题 2 那一行值得单独说，因为第一次跑错了。**
`_model_identity_changed` 跳过作废的条件是**两个都成立**：
`self._identity_model() is model`（对象同一）与 `identity == self._identity`（配置同一）。
第一次只回退了后一半（把五元组改回 `(model.weights.fingerprint(),)`），
**12 项全绿**——那不是「修复不必要」，那是**回退没有回到修复前的状态**：
所有相关测试都写 `engine.model = other`（换对象），对象那一条单独就抓住了它们。
所以补了两个对照：`p2a-object-clause-only.log`（只回退对象那一条）与
`p2b-tuple-only.log`（只回退五元组），**两份都是 12 passed**。
结论是精确的：**这 12 项里，两条检查各自都足以让它们变红；只有同时回退到
「只比权重指纹」这个修复前的状态，才有 4 项变红**——
`test_a_swapped_vision_config_over_the_same_weights_does_not_serve_the_old_features`、
`test_a_swapped_preprocessor_config_refreshes_the_preprocessor_and_the_entry`、
`test_build_prompt_preprocesses_with_the_swapped_models_settings`、
`test_a_config_swap_rewarms_and_then_hits_again`。
另外 8 项在修复前后都绿：它们是上一轮「换权重」那批覆盖，不是本轮问题 2 的。
**五元组那一条不是冗余的**，它管的是对象没换、配置被就地改掉的情形
（对象那条看不见）；只是**现有测试没有单独钉住它**——这是一处覆盖缺口，
记在交付报告「剩余限制」里，本轮没有为它加测试（加了就要构造「同一对象就地改配置」
的夹具，超出四个问题的范围）。

**恢复是可验证的，不是「应该恢复了」。**`restore-hashes.txt` 里
`out_guard.py`、`vlm.py`（三次）、`run_integration.sh`、`run_recollection.sh`
的哈希与当前工作区**逐个相同**；
`make_report.py` 那一行记的是反证当时的 `b8ca95be…`，
它在本轮之后又改了（§9.1 与 §7.3 的轮数区间改成从数据派生），当前是
`92061cd02c71cff4d40072d79113ad879553aa3b45021b88cd9768bb780a0f90`，
所以 `p3` 那一项**在最终树上重跑过一遍**（`p3-order-balance-final.log`，
同样 6 failed；`green-targeted-final.log` 287 passed 也是最终树上的）。

## 真实模型集成测试

```bash
SLOWQ_MODEL_DIR=models/Qwen3-Embedding-0.6B-slowq python -m pytest -m integration -q
```

一个环境变量指定目录，**manifest 里记录的任务决定哪些模块运行**，没有第二个开关需要同步：

| 模块 | 任务过滤 | 覆盖 |
| --- | --- | --- |
| `test_integration_reference.py` | 无（任何目录都跑） | 与固定参考的比较、离线 `inspect` |
| `test_integration_real_model.py` | `text-generation` | 确定性、流式、上下文边界、分块 prefill、cached decode、v0.1 固定输出回归 |
| `test_integration_embedding.py` | `text-embedding` | 向量维度与归一化、pooling 位置、query 前缀、分块等价、相似度、无 lm_head 投影 |
| `test_integration_vlm.py` | `image-text-generation` | 双塔权重完整且互不重叠、PNG 摘要、预处理逐字段复现、crop 排布、双向视觉注意力、greedy 复现、特征消融 |

每个目录收集 62 项，其余按任务跳过并打印原因。**请检查 skipped 列表，
不能把部分通过当作完整集成验证**：一个文本生成目录跑出来的 26 passed / 36 skipped
说明另外两个任务的模块根本没运行。

### 2026-10-23 推理加速第 1 轮的实际执行结果

同一台机器、同一条命令，**逐模型单独设置** `SLOWQ_MODEL_DIR`。本轮跑了两次：
优化 1+2 之后一次（`artifacts/perf-round1/integration-opt12/`），
优化 1~4 全部完成后再一次（`artifacts/perf-round1/integration-opt4/`）。
两次都是 **208 passed、0 failed**，逐目录结果完全一致：

| 模型 | opt12 轮 | opt4 轮 | opt12 耗时 | opt4 耗时 |
| --- | --- | --- | --- | --- |
| SmolLM2-135M-Instruct | 26 passed, 36 skipped | 26 passed, 36 skipped | 5.56 s | 5.98 s |
| SmolLM2-360M-Instruct | 26 passed, 36 skipped | 26 passed, 36 skipped | 11.26 s | 10.85 s |
| Qwen2.5-0.5B-Instruct | 26 passed, 36 skipped | 26 passed, 36 skipped | 17.49 s | 17.52 s |
| Qwen2.5-Coder-0.5B-Instruct | 26 passed, 36 skipped | 26 passed, 36 skipped | 14.17 s | 14.48 s |
| Qwen3-0.6B | 27 passed, 35 skipped | 27 passed, 35 skipped | 12.73 s | 19.37 s |
| Qwen3-Embedding-0.6B | 25 passed, 37 skipped | 25 passed, 37 skipped | 11.31 s | 10.97 s |
| SmolVLM-256M-Instruct | 26 passed, 36 skipped | 26 passed, 36 skipped | 35.87 s | 35.73 s |
| Qwen3-1.7B | 26 passed, 36 skipped | 26 passed, 36 skipped | **115.37 s** | **3828.08 s** |
| 合计 | **208 passed** | **208 passed** | | |

日志里的 `deselected` 数还顺手对上了快速测试的增量：opt12 轮是 `727 deselected`
（685 + 优化 1、2 新增的 42），opt4 轮是 `826 deselected`（727 + 优化 3、4 新增的 99）。
这是从两个独立来源得到的同一个 826，不是把同一个数字抄了两遍。

**Qwen3-1.7B 的 115.37 s 与 3828.08 s 是同一个测试、同一台机器、同一份权重，差 33 倍。**
这个对照恰好补上了上一轮留下的空白——那时只有 4036.71 s 一个观测值，
文档里写的是"与宿主机内存压力相符但尚未做对照测量"。现在有对照了：

- 3828.08 s 那一次，宿主机 `PhysMem: 23G used (3560M wired, **10G compressor**), 137M unused`，
  `vm.compressor_bytes_used` 约 9.3 GiB；pytest 进程逻辑 RSS 5.98 GiB，
  而 `top` 报它的常驻只有 126 MiB——**工作集被压缩出去了，每访问一次就要解压一次**。
  CPU 时间 15 分钟 / 墙钟 64 分钟 = 24%，与上一轮记录的 24% 完全一致，
  说明它不是算得慢，是等内存。
- 115.37 s 那一次是在没有其他负载的窗口里跑的。
- 同一台机器上除 Qwen3-1.7B 之外的 7 个模型，两轮耗时差都在 1.5 倍以内
  （最大是 Qwen3-0.6B 的 12.73 → 19.37 s），因为它们的权重只有 0.5 ~ 2.3 GiB，
  压不进压缩器。**只有 6.4 GiB 权重的那个模型会被这个机制放大。**

宿主机那 10 GiB 不是 SlowQ 造成的：当时占用最大的是 `com.apple.Virtualization`（4.2 GiB）、
Codex Renderer（1.3 GiB）、DingTalk（0.95 GiB）、WindowServer（0.9 GiB）。
这些不是本项目的进程，也不该由测试去动。

**这不是一个已完成的受控实验**，必须说清楚它缺什么：变量（宿主机内存压力）是被观测到的，
不是被操纵的；两次运行也不是相邻交替的，中间隔了几小时和大量其他模型加载。
能下的结论只有一条：**Qwen3-1.7B 的集成测试耗时由宿主机内存状态支配，
115 s 和 3828 s 都在同一份代码上产生过，所以这个耗时不能用来判断代码快慢。**
正确性结论不受影响——两轮的 passed / skipped 数逐项相同。

### 2026-10-26 审查修复后的实际执行结果

同一台机器、同一条命令、同样**逐模型单独设置** `SLOWQ_MODEL_DIR`，
源码是审查四处修复都落地之后的树（`integration-fixed/environment.txt` 里的
`git status --porcelain` 与逐文件 sha256 现场记录）。
`bash tools/perf/run_integration.sh artifacts/perf-round1-rerun/integration-fixed`
（**记录**：这条命令今天会被 `out_guard` 拒绝，因为那个目录已是已发布证据；
重跑请不带参数，见 §「审查修复后重跑（2026-10-26）」里的复现块）：

| 模型 | 结果 | 耗时 |
| --- | --- | --- |
| SmolLM2-135M-Instruct | 26 passed, 36 skipped | 5.68 s |
| SmolLM2-360M-Instruct | 26 passed, 36 skipped | 12.27 s |
| Qwen2.5-0.5B-Instruct | 26 passed, 36 skipped | 16.81 s |
| Qwen2.5-Coder-0.5B-Instruct | 26 passed, 36 skipped | 14.32 s |
| Qwen3-0.6B | 27 passed, 35 skipped | 17.49 s |
| Qwen3-Embedding-0.6B | 25 passed, 37 skipped | 11.00 s |
| SmolVLM-256M-Instruct | 26 passed, 36 skipped | 35.58 s |
| Qwen3-1.7B | 26 passed, 36 skipped | **57.17 s** |
| 合计 | **208 passed、0 failed、288 skipped** | |

**逐目录的 passed / skipped 数与 2026-10-23 那两轮完全相同**——审查轮新增的测试
全部落在快速层（最终树上快速层已从 826 增至 **995**，新增项集中在
`test_perf_tooling.py` 与 `test_model/embedding/session/vlm.py` 的快速用例里），
没有一项进入集成层；四处修复也没有让任何一项集成测试变红。

`deselected` 是 **945**：这次集成跑在 15:24，而 16:13 才往
`tests/test_perf_tooling.py` 加了 8 项身份区间测试（945 + 8 = 953，即当时 16:17
快速层的计数；最终树又增到 991，再补 `run_reference.sh` 那 3 项后是 994，
再补 §2.2 逐字节那 1 项后是 **995**）。
**这个差值本身就是"执行记录早于最终源码"的独立证据**，
与上一节按 sha256 逐文件比对得到的结论一致，不是同一个数字抄了两遍。

**Qwen3-1.7B 这次是 57.17 s**，比 2026-10-23 那两轮的 115.37 s 与 3828.08 s 都快。
`environment.txt` 记录该窗口 `vm.swapusage: total = 0.00M used = 0.00M`、
load average `2.05 2.16 2.35`。这与"耗时由宿主机内存状态支配"的假设**方向一致**，
但**不构成对它的验证**：变量仍然是被观测到的、不是被操纵的，
而且这台机器同时有 CloudShell（101.8% CPU）与 Codex Renderer（92.2%）在跑，
load 2.05 并不是空闲。能说的只有：同一份代码在这个模型的集成测试上
产生过 57.17 s、115.37 s、3828.08 s 三个值，**所以这个耗时不能用来判断代码快慢**。
上一节列出的"要补"（在测量窗口内同步采集压缩器字节数与 RSS 时间序列，
或者直接操纵内存压力做受控实验）本轮仍然没有做。

### 2026-10-21 发布前修复后的实际执行结果

macOS arm64 / Apple M5 Pro，`pytest -m integration`，**逐模型单独设置** `SLOWQ_MODEL_DIR`
（一个模型的通过结果不能代替另一个）。耗时是 pytest 自己报告的会话时长：

| 模型 | 结果 | 该轮耗时 | 转换后目录 `du -sh` |
| --- | --- | --- | --- |
| SmolLM2-135M-Instruct | 26 passed, 36 skipped | 6.84 s | 517 MiB |
| Qwen3-Embedding-0.6B | 25 passed, 37 skipped | 12.07 s | 2.2 GiB |
| SmolLM2-360M-Instruct | 26 passed, 36 skipped | 14.62 s | 1.4 GiB |
| Qwen2.5-Coder-0.5B-Instruct | 26 passed, 36 skipped | 15.48 s | 1.9 GiB |
| Qwen3-0.6B | 27 passed, 35 skipped | 19.18 s | 2.2 GiB |
| Qwen2.5-0.5B-Instruct | 26 passed, 36 skipped | 20.59 s | 1.9 GiB |
| SmolVLM-256M-Instruct | 26 passed, 36 skipped | 39.90 s | 985 MiB（双塔） |
| Qwen3-1.7B | 26 passed, 36 skipped | **4036.71 s** | **6.4 GiB** |

8 个模型合计 **208 passed, 0 failed**。Qwen3-0.6B 多出的 1 项是 v0.1 遗留的
固定输出回归，它显式检查 checkpoint 来源；Qwen3-Embedding-0.6B 少 1 项是因为
参考比较里没有 logits argmax 段。

**这张表是发布前修复轮的运行结果。** 之后的验证工具修复轮只改了
`tools/validate_reference.py`、它的测试和文档，没有改运行时代码，因此没有重跑这些
集成测试；那一轮重跑的是 8 个已有 artifact 的 `compare`（860 tensors 全部 PASS），
以及复审方独立重跑的 7 个模型（182 passed、0 failed，Qwen3-1.7B 未重跑）。
逐模型日志与 sha256 索引见
`artifacts/repair-review-20261021/`。

**所有 skipped 都是任务过滤或 checkpoint 身份过滤，没有一条是"缺少验证证据"。**
逐条读过 `-rs` 打印的原因：文本生成目录跳过 `test_integration_embedding.py` 与
`test_integration_vlm.py`（"is a converted text-generation checkpoint, and this module
tests text-embedding"），向量目录跳过另外两个模块，非 Qwen3-0.6B 的目录跳过那条
显式检查来源的 v0.1 回归。

#### Qwen3-1.7B 的异常耗时与宿主机内存压力相符

上一轮它在 6 passed、约 10 分钟处被中断，当时既没有失败也没有跑完。本轮单独复跑，
**26 passed、0 failed，跑完了**，耗时 4036.71 s。`/usr/bin/time -l` 与 `--durations=0`
给出了下面这些数字：

```
4037.09 real       125.71 user       845.35 sys
     6906134528  maximum resident set size      # 6.43 GiB
832.08s  test_an_answer_is_the_tokenizers_own_decoding_of_the_returned_ids
715.07s  test_a_fixed_seed_reproduces_a_sampled_answer
687.66s  (setup) test_integration_reference.py  # 模块级 fixture，重跑一次 NumPy 抓取
630.66s  test_greedy_generation_is_deterministic
```

这些数字记录了本次运行的内存占用与耗时，但不能单独确定异常耗时的原因：

1. **峰值 RSS 6.43 GiB ≈ 权重整体积 6.4 GiB。** 权重是
   `np.load(..., mmap_mode="r")` 建立的只读映射，**页面驻留由操作系统管理**：
   建立映射不等于权重全部常驻，RSS 本来可以远低于 6.4 GiB。峰值 RSS 还包含进程的
   其他内存，不能仅凭它接近权重体积，就判定全部权重页驻留、反复换页或没有富余物理页。
2. **`sys` 845 s 是 `user` 126 s 的 6.7 倍。** 本次记录的内核 CPU 时间明显较高，
   与缺页处理等系统开销相容，但没有细分测量，不能直接归因为换页。
3. **real 4037 s，而 `user + sys` 为 971 s，约为墙钟时间的 24%。**
   这不是整机 CPU 利用率；两者差额可能包含 I/O 等待、调度等待等，不能全部计为 I/O。

测量时宿主机 swap 已用 5.9–7.4 GiB / 7–8 GiB，可用物理页一度只剩约 60 MiB，
当时也运行着其他桌面应用，但没有测量各进程对内存压力的贡献，不能据此排除测试进程的影响。

**结论：Qwen3-1.7B 的完整正确性回归已通过（26 passed、0 failed）；异常耗时与严重
内存压力、页面调入调出相符，但尚未通过资源充足环境下的对照测量排除其他性能因素。**
没有为此缩短测试、减少 token 数或放宽任何断言。

接入批次记录到同一套测试约 62.7 s（见[历史验证记录](#历史验证记录)），
这是**历史观测，不是资源充足环境下的受控对照，也不是后续运行的耗时承诺**。
有效换页带宽、每个 decode step 的页面调入量都没有单独测量；
“权重工作集被反复换出导致变慢”仍是待验证的机制假设，不能用它预测一定恢复到 62.7 s。

上一轮只留下“约 10 分钟、6 passed 后中断”的不完整结果。
本轮 `test_greedy_generation_is_deterministic` 耗时 630.66 s；
两次运行的时间量级接近，本身不能证明中断位置或异常耗时原因。

后续可在其他测试条件一致、内存资源更充足的环境中复跑，记录系统内存压力与耗时，
并与这里的 4036.71 s 做对照测量。

离线检查也在这一批里：每个模型都有一条测试阻断 socket 后运行 `slowq inspect --json`
以及对应任务的命令（`generate` / `embed` / `vlm`），要求返回 0 且输出的 `source`
与 manifest 逐字一致。

## 历史验证记录

| 批次 / 平台 | 快速测试 | 真实模型集成 | 备注 |
| --- | --- | --- | --- |
| 2026-10-21 v0.2 验证工具修复轮，macOS arm64 / Apple M5 Pro | **685** | **未重跑**（沿用上一轮的 208） | 只改了验证工具、它的测试和文档；8 个已有 artifact 重新 `compare`，860 tensors 全 PASS |
| 2026-10-21 v0.2 发布前修复轮，macOS arm64 / Apple M5 Pro | 665 | 208（8 模型） | 命令与输出见上；Qwen3-1.7B 耗时 4036.71 s，正确性回归已通过，异常耗时与宿主机内存压力、页面调入调出相符但未做对照测量。8 份日志留档在本机 `artifacts/repair-review-20261021/developer-integration-logs/` |
| 2026-10-21 v0.2 接入批次，macOS arm64 / Apple M5 Pro | 646 | 208（8 模型） | 该批次实测的耗时基线（当时没有观测到严重内存压力），Qwen3-1.7B 62.7 s |
| 2026-10-19 v0.1 发布收尾，macOS arm64 / Apple M5 Pro | 480 | 23 | 仅 Qwen3-0.6B |
| v0.1 上一轮修复，Linux x86_64 / Xeon Platinum 8575C | 462 | 23 | 仅 Qwen3-0.6B |

macOS 参考抓取使用 PyTorch 2.14.1、Transformers 4.51.3、FP32 eager attention、CPU、线程 4
（SmolVLM 为 8）。v0.1 在 Linux 上重算的是 **SlowQ NumPy 侧**，比较对象是 macOS 抓取的固定参考，
116 项通过，最差 `rel_l2=1.011e-05`、`max_abs=1.159e-04`；PyTorch 参考端没有在 Linux 重跑。

**v0.2 新增的 7 个模型没有 Linux 真实权重验证记录。** 上表的 Linux 数字只覆盖 Qwen3-0.6B，
不表示新增模型也在那里完成了真实权重验证；本版本的无权重测试另见
[当前验证入口](#current-validation)。跨平台真实模型验证方法与那一轮的实测记录见
[架构说明 · 历史批次记录](architecture.md#16-历史批次记录)。

历史数字用于说明已验证的范围；当前改动是否通过，应以本地执行上述命令的结果为准。
