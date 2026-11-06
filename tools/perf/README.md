# `tools/perf/` — 第 1 轮推理加速的测量工具

这里是产生 `docs/推理加速-第1轮-2026-10-23.md` 全部数字的代码。**工具入版本管理，
原始数据不入**：数据留在 `artifacts/`（被 `.gitignore` 忽略），模型留在 `models/`。

所有命令都在仓库根目录执行，解释器固定为 `.venv/bin/python`。

## 五条不能破的规矩

1. **严格串行。** 两个基准同时跑会争抢同一批核和同一条内存带宽，那样两个数都没有
   意义。所有 `run_*.sh` 都是顺序执行，不要用 `&` 或 `xargs -P` 并行化它们。
2. **A/B 的那一对必须相邻。** 这台机器会节流，页缓存只装得下约四份 Qwen3-1.7B 权重，
   隔一小时跑的两次看到的不是同一台机器。`run_round1.sh` 因此在**每个模型内部**
   交替跑 base / opt，而不是先跑完五个 base 再跑五个 opt。
3. **新实验写新目录。** `artifacts/perf-round1/` 是已发布报告的原始证据，
   `run_round1.sh`、`run_round1_longctx.sh` 和 `run_noise_floor.sh` 都拒绝往那里写。
   重测产生的是**新的证据**，不是对旧证据的修订——删掉旧数字再把新数字当作同一条
   序列，会让更正无法审计。
4. **用"改坏了看它红"验证测试时，改回来要 `rm` 掉 `__pycache__`。** CPython 的
   字节码失效判据是**源文件 mtime + size**。`cp -p` 会把 mtime 一起还原，所以一次
   等长替换（`0.0295` → `0.0290`）在同一秒内还原后，mtime 与 size 都没变，
   改坏那版编出来的 `.pyc` 会被当成有效缓存继续用——测试于是"还原之后仍然失败"。
   本轮在核对 `test_the_identity_docstring_bands_cover_every_run` 时实际踩到了一次。
   还原后 `rm -f src/slowq/__pycache__/<模块>.cpython-*.pyc`，或者干脆
   `PYTHONDONTWRITEBYTECODE=1` 跑这类变异验证。
   变异对象是**已验证过的工具**时不要就地改：把它复制到 `/tmp/falsify/` 下，
   用 pytest 插件在 `pytest_collection_finish` 里换掉加载路径（插件要从
   `session.items[i].module` 取模块，不能靠 autouse fixture——fixture 逐测试运行，
   在那个 hook 之后）。就地改等于撤销修复，中途失败还会把工作区留在坏状态，
   而 `tools/perf/` 未进任何提交，`git` 也还原不回来。
5. **汇总表的每一列都必须出自本次运行自己产生的文件，过滤要在列名上写明。**
   `run_reference.sh` 早先重定向到 `$OUT/<model>-compare.log`，却从
   `$dir/compare.log` 解析全部四列——那个文件它从来不写（`compare` 对抓取目录只读），
   于是汇总表描述的是磁盘上碰巧躺着的上一代，而并排的 `npz=` 描述本次运行，
   两列读起来像同一句话。`log=IDENTICAL` 更糟：它比的是那个文件与脚本几行前从它
   `cp` 出来的备份，**一个拿文件和它自己的副本比的列不可能失败**。
   还有：滤掉某几行再比，就不能把列叫 `log=`；`log_no_timing=` 这个名字本身就是
   证据的一部分，读者不必去脚本里找那行 `grep -v`。同理，`summary.txt` 的头部要
   写下备份后缀，否则 `IDENTICAL` 无法归属，事后谁也复算不了。
   三条测试守着这一条（`tests/test_perf_tooling.py` 里
   `…reference_summary_is_derived_from_the_log_this_run_produced`、
   `…a_tagged_reference_summary_reproduces_from_its_own_logs`、
   `…a_superseded_reference_summary_is_kept_and_says_why`），红跑在
   `artifacts/perf-round1-rerun/falsification/reference-log-self-comparison.log`。
   **同一条规矩也管文档里的散文。** `make_report.py` 早先把
   「新 `compare.log` 与 pre-round1 备份**逐字节相同**」当字面量印进 §2.2：
   8 份里其实只有 7 份逐字节相同，SmolLM2-360M-Instruct-slowq 差恰好一行
   `[compare] took`（比较器给自己计的秒表），滤掉后 8/8；而同一句话对中间三代
   是 6/8。**这个计数依赖"说的是哪一代"，写死必然过时。** 现在那一格由
   `reference_identity()` / `timing_only_difference()` / `reference_identity_cell()`
   从 `artifacts/validation/*/` 现算，抓取产物不在树上时改印「未在本地复算」。
   第四条测试 `…the_pre_round1_identity_row_counts_the_models_it_actually_covers`
   独立重算一遍并要求该行按 `；` 分句后逐句带上计数与例外模型名，红跑在
   `falsification/preround1-blanket-byte-identity.log`（这份不需要注入，
   缺陷就是那句字面量本身）。

## 两棵树

| 树 | 怎么来 | 身份 |
| --- | --- | --- |
| `base` | `baseline_tree.py` 从 git 修订 `8968e39` 用 `git archive` 物化 | 逐文件 sha256 与该修订核对；快照目录只是缓存 |
| `opt` | 每次测量所用的工作区 | 逐文件 sha256；第 1 轮采集时尚未提交，验收后代码提交为 `c5bfc42` |

上表区分测量时的源码身份与后续提交。v0.3.0 的版本和 CI 收尾不重写历史 JSON，
也不把旧性能数字标作新版本重测结果；新测量仍记录实际工作区的逐文件哈希。

```bash
# 物化（已存在则只做哈希校验），并把校验过的路径打到 stdout 供脚本取用
BASE_SRC=$(.venv/bin/python tools/perf/baseline_tree.py --print-dest)

# 换基线修订：
.venv/bin/python tools/perf/baseline_tree.py --revision <sha> --dest <新目录>
```

`baseline_tree.py` **不会覆盖已有快照**：目录非空且源码哈希对不上就直接失败并列出
差异文件，而不是重新物化。想让某个目录被重写成另一个修订，先把它移走。

## 源码身份

测量 JSON 里的 `source_identity` 块由 `source_identity.py` 生成，把数字挂到代码而不是
挂到日期上。

**注意时间顺序**：这个块是后补的，`artifacts/perf-round1/` 里已发布的那批 JSON 全部
**早于**它——那个目录下 70 个 JSON，带身份块的是 0 个
（`grep -rl '"source_identity"' artifacts/perf-round1 --include='*.json' | wc -l`）。
那些数字只能靠日期和"这是当时的代码"这句话挂靠，无法用哈希核对。
这正是必须用修好的源码重新采集（写进新目录）的原因之一。

单独查一棵树的身份：

```bash
.venv/bin/python tools/perf/source_identity.py --src .            # 全部逐文件哈希
.venv/bin/python tools/perf/source_identity.py --src . --digest-only   # 只要短 digest
```

- `source` — `src/slowq/*.py` 每个文件的 sha256，**按树内相对路径为键**，
  所以两份不同位置的同一修订哈希相同。
- `tooling` — `tools/perf/` 每个脚本的 sha256，让 harness 版本跟着它产生的数字走。
- `harness` — 实际运行的那个脚本的路径与 sha256。
- `git` — HEAD、是否 dirty、哪些文件有改动。**只有检出根才会被描述**：
  物化出来的基线树位于本仓库 `artifacts/` 内部，问 git 只会得到外层检出的状态，
  那是错的树的身份，所以它返回 `repository: false` 并用 `BASELINE-REVISION` 标签代替。
- `digest` — 只覆盖 `source` + `tooling` 的短哈希。刻意排除 `git` 块，
  因为那里含检出路径和未跟踪文件列表，两台源码相同的机器会给出不同值。

## 测量 harness

| 脚本 | 测什么 | 关键参数 |
| --- | --- | --- |
| `core_ab.py` | 文本生成。三个套件：`micro`（引擎分块 prefill / 单次 forward prefill / 单 token decode）、`workloads`（与 `tools/benchmark.py` 逐字相同的五个对话负载，走 `Engine.generate`）、`chat`（固定四轮对话，冷路径逐轮独立请求 vs 有 `Engine.session` 时的热路径，并逐轮比对 token id） | `--src --model --suite --threads --repeat --warmup --decode-steps --prefill-chunk-size --max-context --no-chat` |
| `vision_cache_ab.py` | 图片特征缓存的开/关，SmolVLM-256M | `--image --image-other --cache-bytes --max-new-tokens` |
| `document_index_ab.py` | 预编码文档索引 vs 每次重算，Qwen3-Embedding-0.6B | `--corpus-size --queries` |
| `projection_probe.py` | 单独给一次词表投影定价（优化 1 的归因）。`forward` 就是 `encode` 加一次投影，所以对同一批 id 分别计时再相减 | `--model --lengths --ab-dir` |
| `sigmoid_check.py` | sigmoid 新旧写法计时 + 逐位相同检查（优化 2 的归因）。模型路径写死在脚本里 | `--tokens --repeat --warmup` |
| `identity_cost.py` | 给图片特征缓存每次查询前那次**模型身份检查**定价，并把它的替代方案——真的去 sha256 全部权重——一并计价。分三层计时（`fingerprint` ⊂ `identity_tuple` ⊂ `identity_check`），所以"把配置也盖进来"的成本不藏在一个总数里；另外对每张验证图跑 `disabled` / `miss` / `hit` 三条缓存路径，并把三条路径的特征都与绕过缓存直接算的结果逐位比（`bit_identical` / `max_abs_difference`）。`vlm.py` 两处 docstring 里的数字全部出自它 | `--model --image --calls --key-calls --path-calls --skip-weight-hash` |

三个 `*_ab.py` 都能原样跑在两棵树上：`core_ab.py` 只用基线修订就存在的 API
（唯一的例外是 `chat` 套件，它用 `hasattr(engine, "session")` 探测，缺失时把热路径
报成 unsupported 而不是崩掉）；另两个则依赖基线**没有**的 `image_cache_bytes` /
`index_documents`，在基线树上把这些路径报成 unsupported，只跑关缓存的那条——
那条就是"之前"的数字。每个脚本开头都断言 `slowq.__file__` 确实在 `--src`
下面——开发 venv 里有一个指向主检出的 editable 安装，测错树比测不出更糟。

**`--repeat` 对 decode 不是"重复一次前向"。** prefill 的每次重复确实是一次独立的
同样输入的前向；decode 的一次重复是一个**窗口**：把同一段 prompt 重新 prefill 进一个
全新的 KV cache，再连续跑 `--decode-steps` 步单 token 前向，计时的边界是整段窗口，
样本数是窗口数。一个窗口内部的连续 token 走的是同一个不断增长 cache 里的同一条轨迹，
把它们当成独立重复实验会让"离散度"随步数变小而不是反映测量噪声。
早先那版 decode 探针只跑一个窗口，所以报告里凡是来自它的行都标成 `1（单窗口）`、
离散度标成 `无（单窗口）`——一个观测值没有离散度，这和离散度为零是两件事。

## 驱动脚本

**每个驱动的 `[out-root]` 都是可选的，默认值是 `out_guard.py fresh-root` 现造的一个
唯一新目录**（`artifacts/perf-run-<UTC 时间戳>`），并打到 stderr。已发布的那两个根
（`artifacts/perf-round1`、`artifacts/perf-round1-rerun`）**被拒绝**：它们是已发布报告
背后的原始证据，重测只能加在旁边，不能替换。显式给路径时会先规范化——尾斜杠、相对
路径、绝对路径、符号链接（含悬空链接）都解析到同一个真实目标再判重，所以
`artifacts/perf-round1-rerun/`、`./artifacts/perf-round1-rerun` 和一个指向它的软链
都被同样拒绝。冲突**在昂贵测量开始之前**检查，实际写 JSON 时 `write_json` 再挡一次；
脚本**不会**把旧产物删掉、移走或改名来让自己跑通。

编排脚本（`run_recollection.sh`）可以先造好一个运行根，子步骤共享它——
`out_guard_reserve` 接受一个**已存在的空目录**，所以子步骤在编排根下面建自己的
新子目录不会被误拒；但每个子步骤只写自己的产物。

```bash
# 步骤 1~4：逐模型相邻交替的 A/B、opt 树线程/分块扫描、图像缓存 A/B、文档索引 A/B
bash tools/perf/run_round1.sh [out-root]          # 默认：新造的唯一目录

# 补跑：micro + workloads 在 --max-context 2048 下重测（long-prompt-short-answer
# 的提示词有 1662 token，1024 的窗口下两棵树都只能记 skipped）。
# 必须是 --suite all：只跑 workloads 就没有窗口内的 decode 自检，
# 拿 ab/ 里几小时前的 micro 去当自检是跨进程比值，机器漂移会被算进"异常"。
bash tools/perf/run_round1_longctx.sh [out-root]

# 跨进程离散度实验：多轮、AB/BA 顺序均衡、每轮两个独立进程
bash tools/perf/run_noise_floor.sh [out-root] [rounds] [models...]   # 默认 4 轮

# 单独的 A/B / 扫描驱动，run_round1.sh 内部也调它们
bash tools/perf/run_ab.sh <src-dir> <out-dir> <label> [models...]
bash tools/perf/run_sweep.sh <src-dir> <out-dir> <label> [models...]

# 8 个模型的真实权重集成测试，一个模型一个 pytest 进程
bash tools/perf/run_integration.sh [log-dir]

# 用当前源码重抓 8 个模型的参考验证 numpy 半边并 compare；抓取协议从每个模型自己的
# reference.json 里读回来，所以它不可能和当初抓参考实现时用的协议走岔。
# 抓取前把 artifacts/validation/<model>/ 下的 slowq.npz 与 compare.log 备份成
# *-$backup-tag*（只在备份不存在时才写，重跑不会用"之后"的版本顶掉"之前"的基线），
# summary.txt 的 npz= / log_no_timing= 两列就是对这一代备份比出来的。
bash tools/perf/run_reference.sh [log-dir] [backup-tag]     # backup-tag 默认 pre-review
```

**退出码就是结论。**`run_integration.sh`、`run_reference.sh` 与 `run_recollection.sh`
都是"每个模型/每一步都跑完，再把累计失败当成自己的退出码"：打印 `OVERALL_RC=1`
然后 `exit 0` 等于把结论写进一行没人读的状态里——`&&` 链、CI 作业和一句 `$?`
都会看到成功。三者在遇到失败时**都继续跑完剩下的模型/步骤**（一趟就知道八个里哪几个
失败，是这些脚本存在的理由），但最后一定以非零退出。`run_integration.sh` 还把
"解释器不存在"（`rc=127`，pytest 根本没启动）算作失败，不算作一次空通过。

**记录文件要先取得所有权，才允许追加。**`set -C` 只管 `>`，而
`run_recollection.sh` 的两份记录（`recollection-environment.txt`、
`recollection-status.txt`）在收尾时是用 `>>` 再写一次的：追加对已存在的文件**总是成功**。
于是"预检通过 → 另一次运行抢先把环境文件写成非空 → 本次 `>` 被 noclobber 拒绝"这条路径上，
旧代码只判 `[ ! -s ]`，别人那份非空文件看起来就像自己写成功了，
收尾的 `## machine at the end` 追加进**别人的证据**，进程还打印 `re-collection done` 返回 0。
状态文件同理：首次创建失败会累计，但后续四条步骤记录照样追加进冲突文件。
现在每份记录文件都由一次显式 `: >` 排他创建、**读它的返回码**，只有创建成功的那一份
（`OWN_ENVIRONMENT` / `OWN_STATUS`）才允许后续追加；两份里任意一份没拿到写入权，
最终一律非零退出——四个测量步骤全过也不例外，没有状态记录的运行不是完成的运行。
`: >` 被拒绝时不写任何字节，冲突文件的内容与 `st_mtime_ns` 都不变。

`identity_cost.py` 的四段计时**有顺序**：fingerprint → 像素 key → `encode_image` →
全部权重的 sha256。后两步是故意会扰动机器的（一次 2.5~4.1 s 的视觉前向、一次把
978 MiB 或 6.4 GiB 拉过页缓存的扫描），所以放在最后；负载平均值在这两段**之前**
就采好了，写进 `machine.before` / `machine.after`。它也记两套哈希：`*_sha256`
是审计线索，`*_code_sha256`（对 docstring 与行号不敏感）才是"这份数据还描述当前
代码吗"的判据——因为它测出来的数字要写进 `vlm.py` 自己的 docstring，
只认原始哈希就会陷入"改注释⇒数据失效⇒重测⇒再改注释"的循环。

`run_sweep.sh` 的七个配置是 `t-default / t1 / t4 / t8`（线程）和
`c64 / c256 / c512`（prefill 分块），输出名 `${model}-${label}-${tag}.json`。

`run_noise_floor.sh` 是**唯一**能给"这个差是不是真的"提供依据的驱动。它回答的问题是：
同一棵树、同一份配置，两个**独立进程**能差多远。进程内那七次重复的 min–max 很紧
（常常 ±0.5%），但那只说明一个进程内部自洽，看不见进程之间的漂移；把紧区间当成
分辨率，会让 2–3% 的差看着像结论。奇数轮数会被拒绝——多出来的那一对 AB 会把顺序
重新变成混淆项。顺序位置写进 `--label`（`noise-r3-base-pos1`），`make_report.py`
据此**核对**首/次是否真的各半，而不是相信这个循环按写的那样跑了。
产物交给 `make_report.py` 的 `paired_quantities` / `paired_noise_section`：
进程内离散与进程间离散分开列，逐轮配对增益按是否跨过 1.00 判为
收益 / 回退 / **尚不能确认收益或回退**。没有 `noise-floor/` 目录时，报告改印
`incidental_noise_section`——把已有的两次偶然同配置运行如实标成"不是实验"。

**顺序均衡是按"剔完之后还剩哪些轮"重算的，不是按采集时的轮数。**污染判据
（`spike` / `level`）按「量 × 轮 × 侧」逐格生效，所以一行真正用来算增益的轮集
通常比采集到的那个小；4 轮 2 AB + 2 BA 剔掉一轮就变成 3 轮 1 AB + 2 BA，
此时有一棵树系统性地排在第二个进程、系统性地继承热起来的 page cache，
**用它算出的方向不是测量的方向**。本表因此印 `有效轮次`（真正算出配对增益的轮数 /
采集轮数）与 `顺序均衡（保留轮次）`，并且只要后者不是「是」就**不给方向**：

| `结论` | 触发条件 |
| --- | --- |
| `收益` / `回退` | 有效配对 ≥ 2、保留轮次 AB/BA 均衡、且没有任何一轮跨过 1.00 |
| `尚不能确认收益或回退` | 逐轮增益跨过 1.00（幅度判不出方向） |
| `顺序不均衡或未记录，尚不能确认收益或回退` | 保留轮次首/次不是各半，**或**标签里根本没有位置信息（缺失与非法都算） |
| `有效轮次不足，尚不能确认收益或回退` | 有效配对 < 2。这一条**先**判：只剩 1 轮时 `1 × 2 ≠ 1` 也是"不均衡"，先判均衡会把一个样本量问题报成驱动 bug |

后两档与 `尚不能确认收益或回退` 一样，**不计入**任何"达到方向一致判据"的计数，
也不计入 `收益` / `回退`。`PAIRED_SUMMARY` 里那几个计数与表里逐行印的结论
由脚本自己断言一致（行数、每档条数、明细列表长度三项都对得上），
所以 §7.2 / §9.1 引用的数字不可能与读者看到的表格分家——
先前那句「70/73 达到稳定收益判据」正是这样活下来的：表格已经不支撑它，计数还在。
弃用的轮次与理由**保留在报告里**，原始 JSON 一个都不删。

## 生成报告

```bash
# 只读地看一遍（正文与表格都落进临时目录，正式产物一份不动）
.venv/bin/python tools/perf/make_report.py \
    --data artifacts/perf-round1-rerun \
    --paired-data artifacts/perf-round2-paired \
    --out /tmp/render/report-tables.md --doc /tmp/render/doc.md

# 正式更新交付文档与当前派生表（这是一次发布动作，不是测试）
.venv/bin/python tools/perf/make_report.py \
    --data artifacts/perf-round1-rerun \
    --paired-data artifacts/perf-round2-paired \
    --out docs/推理加速-第1轮-测量表.md \
    --doc docs/推理加速-第1轮-2026-10-23.md
```

写两份：`--out`（纯表格，默认 `<data>/report-tables.md`）和 `--doc`
（表格 + 叙述，默认 `docs/推理加速-第1轮-2026-10-23.md`，叙述也在同一脚本的
`write_doc` 里）。**报告里没有人手转录的数字**，叙述中引用的区间同样由脚本从同一批
JSON 算出。输入缺失的小节会打印一行显式的 `MISSING`，而不是被静默丢掉——
一次没测完的采集不能生成一份看起来完整的报告。

**四个路径参数各自独立**，因为「读哪个数据根」和「写到哪」是两种权限。
表头会印出这次调用实际用的 `--data` / 配对根 / 配对根的来源 / `--doc`，
§1.3 还会把四个路径逐行列成一张表——正文与表格由**同一次调用**写出，所以逐项一致、
可以互相核对。

`--out` 不给时跟着 `--data` 走，而**它落在已发布证据目录里时会被拒绝**
（`artifacts/perf-round1/`、`artifacts/perf-round1-rerun/`）。那两个
`report-tables.md` 是已发布报告的派生证据，和它们旁边的原始 JSON 一起**原样保留**：
它们各自读的是自己那一次的数据根，行数与当前正文不同属于**来源不同**，
不是某一处数字写错、也不该手改数字去对齐。要给本轮生成表格，显式 `--out`。
显式 `--out` 是打字那个人自己的发布动作，脚本不替他二次猜测。

`pytest` **不会**再写任何正式产物。`tests/test_perf_tooling.py` 里有一个 autouse
fixture 对 `PROTECTED_PRODUCTS`（两个根的 `report-tables.md`、交付正文、
以及本轮新建的交付表格 `docs/推理加速-第1轮-测量表.md`，共四项）取
内容 + mtime 指纹，**哪个测试改写了它就点名哪个测试**；所有渲染都走
`render_report()`，正文与表格全落进 `tmp_path`。这条规矩早先不存在，
于是八个按两个已发布根参数化的测试各自调 `report.main(["--data", root])`、
把 `report.DOC` 指到 `tmp_path` 却没有指 `report.OUT`，一次完整快速测试把
`perf-round1/` 那份写 8 遍、`perf-round1-rerun/` 那份写 9 遍——而且**不带
`--paired-data`**，历史表格因此读的是旧配对实验（65 项收益），
旁边那份正文读的是新配对实验（64 项）。所以：在证据目录里看到这两个文件
mtime 变新**不再是预期行为**，看到了就是有人绕过 `render_report()` 跑了生成器。

四项里有两项在 `.gitignore` 掉的 `artifacts/` 下，所以**指纹对比本身不能依赖它们在不在**：
`product_fingerprint()` 对不存在的路径返回 `None`，`None == None`，
一份从没被搬过来的产物和一份被测试删掉的产物在对比里长得一样。两件事因此分开做——
autouse fixture 只做前后对比（不在就是不在了，没东西可改写，副本里照样跑），
"这四项确实都在"是**发布验收**，落在
`test_every_protected_product_in_a_full_checkout_is_fingerprintable`，
用 `protected_products_missing` 这个 `skipif` 门控，和这个文件里
`ab_artifacts_missing` / `identity_artifacts_missing` 等既有约定一致：
门控，不是删断言。保护机制本身（删除、改写、内容相同但 mtime 变了）
用 `tmp_path` 造的假产物驱动，两种环境都真跑。

**同一个进程里第二次 `main()` 从空状态开始。**`main()` 只重新赋值它自己
`global` 的那五个名字，另外三个不是它赋的：`SIGMOID_CORRECTION` 只在
`<data>/sigmoid/` 存在时才覆盖；`PAIRED_SUMMARY` 只在 `paired_noise_section` 里赋值，
而那个函数在根目录下没有配对数据时**提前 return**；`PAIRED_BETWEEN` 按模型累加、从不清空。
于是一个进程里连着调两次 `main()`（notebook、长驻调用方、测试），
第二次的正文会引用第一次的结论：§7.2 印「N 个模型 × M 轮 …已经跑过」，
而同一次调用写出的表格头部注释是「配对实验（§7.3）根目录：（未运行）」。
现在 `main()` 在解析完参数、解析任何路径**之前**先调 `_reset_run_state()`，
把七个输入派生的全局量一起放回导入时的值。默认值写在声明旁边、由这一个函数统一复位，
不在 `main()` 里再抄一份——抄第二份正是这个缺陷的形状。
**每次独立进程运行的 CLI 从来不受影响**，已发布的 64／5／4 也不受影响。

`--paired-data` 是给 §7.3 那张配对表单独指一个运行根：重测**必须**写进新目录
（已发布的根拒绝写入），可报告的 `--data` 仍然指向原来那一轮，
所以两者要能分开指。**给了这个参数就只允许读那一个来源，没有回退**：
它早先是一串候选按顺序试（先 `PAIRED_ROOT/noise-floor`，再 `<data>/noise-floor`），
于是拼错路径或新实验还没采集时，脚本照常生成**上一轮**的结论，
而报告抬头写的是操作员指定的那个目录。现在参数在**渲染任何一节之前**判掉，
不可用就打印具体原因并非零退出——所以参数写错不会先覆盖正式文档再报错。
五种「不可用」分别点名：目录不存在 / 那个位置是个文件 / 缺 `noise-floor/` /
JSON 读不出来 / 读不出任何配对 JSON。不给这个参数时行为不变：读
`<data>/noise-floor/`，没有就显示「未运行」。
它**只**影响配对表：`--data` 下其余每一节的来源都不动，
免得一次重测被读成"整份报告都重新采集过了"。

## 数据布局

`artifacts/perf-round1/` 是已发布那一轮的原始证据，**不要改写**：

```
baseline-tree/          8968e39 的物化快照 + BASELINE-REVISION 标签
ab/                     步骤 1：5 模型 × {base,opt}，micro+workloads+chat
ab-longctx/             补跑：同 5 模型 × 2 树，--max-context 2048。
                        **这一批当时是 --suite workloads 跑的，JSON 里 micro 为空**，
                        所以报告无法给它窗口内自检，25 行全部标 **待重测**；
                        驱动已改成 --suite all --no-chat，重跑会写进新目录
baseline/               本轮之前那棵树的线程/分块扫描
sweep-opt/              优化后那棵树的同一批扫描，列与 baseline/ 对齐
projection/             一次词表投影的净代价，5 模型
sigmoid-check.json      sigmoid 新旧写法计时 + 逐位相同检查
vision/                 图像特征缓存 A/B
documents/              预编码文档索引 A/B
integration-opt12/      8 模型集成测试日志
integration-opt4/
run_round1.log          两次运行的完整 stderr
run_round1_longctx.log
report-tables.md        与报告表格部分同源
```

已发布那一轮里**没有** `noise-floor/`：跨进程配对实验是这次审查之后才有的驱动。
它只会出现在新的输出根下（默认 `artifacts/perf-round1-rerun/noise-floor/`），
`make_report.py` 按目录是否存在决定印哪一节，不会拿别处的数字顶替。

`artifacts/perf-round1-rerun/` 是审查之后那一批**新证据**的根，同样不改写：

```
sigmoid/                更正后的 sigmoid 归因：sigmoid-check.json（最新）+
                        sigmoid-check-run1/2/3.json（保留的历史观测，区间端点出自它们）
identity/               identity_cost.py 的产物：SmolVLM-256M-Instruct-slowq-run{1,2,3}.json
                        与 Qwen3-1.7B-slowq-run1.json。vlm.py 两处 docstring 引的
                        全部数字出自这四个文件；第二个模型是用来证明代价跟着
                        manifest 条目数走、不跟着张量字节数走的
integration-fixed/      修复后源码的 8 模型集成测试日志 + environment.txt
                        （uname / sw_vers / numpy 配置 / load / swap / git HEAD 与
                        status --porcelain），因为"跑在什么机器上"是结论的一部分
reference-fixed/        修复后源码重抓的参考验证：source-identity.txt、逐模型
                        *-numpy.log / *-compare.log、summary.txt。summary 里
                        npz=IDENTICAL 表示新抓的 slowq.npz 与被替换掉的那份
                        **逐字节相同**；替换前那份由 run_reference.sh 备份成
                        artifacts/validation/<model>/slowq-<backup-tag>.npz
reference-final/        最终树重跑，backup-tag=pre-final。summary.txt 用拆开后
                        的 worst_numerics_rel_l2 / worst_internal_rel_l2 两列，
                        但四个列仍从上一代日志解析、log= 是自比 —— 见旁边的
                        LOG-COLUMN-CORRECTION.txt，文件本身原样保留未改
reference-final2/       驱动修好后按新格式再跑，backup-tag=pre-final2。
                        四个列全部出自本次运行自己的 <model>-compare.log，
                        summary.txt 头部记下 backup-tag 与比较规则
quick-fixed.log         快速套件的落盘记录；头部记命令、时间、git HEAD 与
                        make_report.py / run_reference.sh 的 sha256。前两代
                        留在 quick-fixed.log.prev-994 与 .prev-991 —— 故意都
                        不叫 quick*.log，否则 make_report.py 的
                        quick_suite_expectation() 按名字排序会先取到它们，
                        把被取代的计数喂给文档 §10
falsification/          24 份"改坏应变红"日志。前 19 份是就地改实现再 cp 还原
                        （每份都配 cmp 还原确认与变绿确认）；第 20–23 份改用
                        /tmp/falsify/ 下的副本 + pytest 插件注入，头部记复制路径、
                        驱动命令、插件路径与仓库那份文件跑前跑后的 sha256；
                        第 24 份 preround1-blanket-byte-identity.log 不需要注入
                        —— 缺陷就是生成器当时印的那句字面量
```

`integration-fixed/` 与 `reference-fixed/` 是**修复后最终源码**的执行记录，
和 `artifacts/perf-round1/integration-opt*` 那批（修复前）不是同一条证据。
