# 测试与参考验证

本文说明 SlowQ v0.2 怎么验证：跑什么命令、比什么、门限是多少、实测结果如何。
所有命令从项目根目录执行。安装与模型下载需要网络；准备好依赖和本地模型后，
测试与参考比较可以完全离线运行。

返回 [README](../README.md)；模型清单与支持范围见[模型支持](models.md)；
实现细节见[架构说明](architecture.md)。

## 两层测试

| 层 | 命令 | 需要网络 | 需要真实权重 | 数量 |
| --- | --- | --- | --- | --- |
| 快速测试 | `pytest -m "not integration"` | 否 | 否 | 685 |
| 真实模型集成 | `SLOWQ_MODEL_DIR=<dir> pytest -m integration` | 否 | 是 | 每模型收集 62，按任务过滤 |

快速测试用临时构造的微型模型和 tokenizer，覆盖算子、KV cache、采样、流式输出、
config 家族规则、转换器、验证工具和依赖边界。`.[dev]` 只增加 pytest，不安装 PyTorch。

## 快速测试

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -m "not integration" -q
```

**2026-10-21 验证工具修复轮在 macOS arm64 / Apple M5 Pro 上的实际输出：**

```
685 passed, 62 deselected in 6.42s
```

那 62 项是真实模型集成测试，被 `-m "not integration"` **deselect**（根本不收集）。
不加标记过滤直接跑 `python -m pytest -q`，它们会被收集后**skip**：

```
685 passed, 62 skipped in 4.10s
```

skip 原因是没设 `SLOWQ_MODEL_DIR`。两种写法通过的快速测试都是 685 项，
差别只在 pytest 报告这 62 项的方式；引用输出时要与命令对应。

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

**v0.2 新增的 7 个模型没有 Linux 验证记录。** 上表的 Linux 数字只覆盖 Qwen3-0.6B，
不表示后续新增测试或新增模型在那里执行过。跨平台验证方法与那一轮的实测记录见
[架构说明 · 历史批次记录](architecture.md#16-历史批次记录)。

历史数字用于说明已验证的范围；当前改动是否通过，应以本地执行上述命令的结果为准。
