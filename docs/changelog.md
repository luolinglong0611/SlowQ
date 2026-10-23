# 变更记录

[返回 README](../README.md)

## v0.2.0

从"只支持 Qwen3-0.6B 的文本生成"扩展到 **8 个模型、3 类任务**（文本生成、文本向量、
图文生成）。本轮改动、验证状态与未完成项见 [v0.2 接入报告](v0.2-多模型接入报告.md)，
逐个模型的状态见[模型支持](models.md)。

### 模型与任务抽象

- 把"架构家族"与"任务类型"拆成两条独立的轴。家族由 `config.json` 的 `model_type`
  决定，规则集中在 `config.py` 的 `TEXT_FAMILIES`（qwen3 / qwen2 / llama / idefics3）；
  任务写进转换后目录的 manifest，**不从 config 猜** ——
  `Qwen/Qwen3-Embedding-0.6B` 公布的 `model_type` 与 `architectures` 和聊天模型取值一致。
- 删除代码里按模型名分支的地方。原来写死"有 Q/K RMSNorm""QKV 有 bias""head_dim 可推导"
  的断言，改成按家族声明：qwen2 的 Q/K/V bias 恒开、O 投影恒关（硬编码在官方
  `Qwen2Attention.__init__`，config 里没有字段）；llama 的三种 bias 都从 config 读；
  qwen3 要求 `head_dim` 必须显式给出，缺失时报 `ConfigError` 而不是回退到
  `hidden_size / heads`（Qwen3-0.6B 的 head_dim 是 128，而 `1024/16 = 64`）。
- sliding window 按**实际启用状态**判断：两个 Qwen2.5-0.5B checkpoint 的 config 里都有
  `sliding_window: 32768` 且 `use_sliding_window: false`，此时跑全注意力；
  既不因为字段存在就拒绝加载，也不会把真正启用的滑动窗口当成全注意力。
- 新增算子：`layer_norm`、`gelu_tanh`、`conv2d_patchify`、`pixel_shuffle`、
  `l2_normalize`；`linear` 增加可选 bias。视觉塔的注意力传 `mask=None`（非因果，
  且单图路径不产生 padding）；真出现部分 padding 的 patch 网格时报错而不是屏蔽。

### manifest 与离线目录

- manifest `format_version` 升到 2，新增 `task` 与 `model_family` 两个键，两者都必须存在。
  版本 1（只有 v0.1 写过）仍可读，缺键时按 v0.1 的唯一可能取值处理为文本生成；
  其他版本号被拒绝，不会静默降级。**v0.1 转好的 Qwen3-0.6B 目录不需要重新转换。**
- 转换器新增 `--task auto|text-generation|text-embedding|image-text-generation`。
  `auto` 读文件而不是读仓库名：sentence-transformers 导出的三个标记文件
  （`modules.json`、`config_sentence_transformers.json`、`1_Pooling/config.json`）
  表示向量模型，嵌套 `vision_config` 加 processor 文件表示图文模型；
  三个标记文件只出现一部分时报错而不是猜。显式指定的 task 仍会与文件证据交叉校验。
- 每个 repo 解析并记录**自己当时**的 commit，不再套用 v0.1 为 Qwen3-0.6B 固定的 revision。
- 图文目录额外保存 `preprocessor_config.json`、`processor_config.json`、`chat_template.json`
  与 `added_tokens.json`，使离线预处理不需要网络。
- 并非所有任务都需要 `generation_config.json` 或 chat template；缺文件时按任务判断是否致命。

### 新任务：文本向量

- 新增 `src/slowq/embedding.py`：`EmbeddingEngine`、`EmbeddingModel`、`EmbeddingResult`、
  `format_query`、`last_token_pool`、`cosine_similarity`、`similarity_matrix`。
- 编码路径与官方 snippet 一致：query 加 `Instruct: …\nQuery:…` 前缀（`Query:` 后无空格），
  document 不加；`add_special_tokens=True`；保留官方因果注意力（不是双向）；
  取最后一个有效 token 的 final hidden state（不是平均池化）；L2 归一化；
  输出官方全维度 1024。
- **不做全词表投影，也不做自回归生成。** KV cache 按当前文本长度分配，不按 `max_length` 预分配。
- 超过 `max_length`（默认 8192）时报错，不静默截断。
- 新增 `slowq embed` 子命令：`--query` / `--document` 可重复，逐条编码并输出两两余弦相似度。

### 新任务：图文生成

- 新增 `src/slowq/vlm.py`：`VisionEngine`、图像预处理、ViT 视觉塔、pixel shuffle、
  connector、图像特征注入、图文 prefill 与复用 KV cache 的文本 decode。
- 完整实现官方单图切分：resize 到最长边 2048 再向上取整到 512 的倍数，切成 512×512 的
  rows×cols 个 crop **加 1 个全局帧**，每帧经 pixel shuffle 后是 64 个 image token，
  crop 之间插入 `<row_r_col_c>` 标签。"只支持一张图"不等于跳过这张图产生的多个 crop。
- 视觉塔是与文本塔不同的一套：LayerNorm（weight + bias）而不是 RMSNorm，
  fc1/fc2 + GELU-tanh 而不是 SwiGLU，q/k/v/out_proj 都带 bias，
  全连接**非因果**注意力，位置来自学习到的 `position_embedding`，没有 RoPE。
- 新增可选 extra `slowq[vision]`（Pillow），只有 `slowq.vlm` 会用到，且只在真正读图时导入。
  **纯文本安装仍然是 numpy + tokenizers + Jinja2 三个运行时依赖。**
- 新增 `slowq vlm` 子命令；`GenerationMetrics` 增加 `vision_encode_seconds`，
  只有图文请求会打印这一行，文本请求的输出不变。
- 图像缺失或读不出、占位符数量对不上、扩展后超过 `max_context`，都给出明确错误，
  不静默截断或忽略图像。一次请求只接受一张图，第二张报错。

### CLI 与工具

- `slowq inspect` 按 manifest 记录的任务分派，三种任务的目录都能查。
- `Engine.from_directory` 拒绝向量与图文目录，`EmbeddingEngine.from_directory` 拒绝
  文本生成目录，两边都在错误信息里指出正确入口。
- `tools/validate_reference.py` 重构为多家族多任务：抽样层按家族与层数选择
  （不再固定第 27 层），抽样权重张量跟着家族走（qwen3 取 q/k norm，qwen2 取 q_proj.bias，
  llama 两者都不取），artifact 目录按模型名分开（`artifacts/validation/<模型目录名>/`），
  artifact 格式升到 3。新增图文任务的逐段验证：官方 processor 的 pixel 张量与 mask、
  视觉塔输出、connector 输出、图文融合位置、最终 logits 与 cached decode。
- 参考侧的图像由 NumPy 侧生成并存成 PNG，参考侧按 sha256 校验，
  两侧证明看到了同一份字节；图像不下载也不入库。

### 测试

- 新增 `tests/real_models.py` 作为真实权重测试的共享管道；集成测试拆成四个模块，
  按 manifest 里的任务过滤：`test_integration_reference.py`（无任务过滤）、
  `test_integration_real_model.py`、`test_integration_embedding.py`、`test_integration_vlm.py`。
- 新增快速测试 `tests/test_embedding.py`、`tests/test_vlm.py`；`tests/conftest.py` 增加
  按家族构造微型 checkpoint 的夹具，覆盖有/无 QKV bias、有/无 Q/K norm、
  head_dim 推导 vs Qwen3 的独立 head_dim、tied/untied 权重、chat template 与停止符、
  缓存位置与上下文边界、full 与 chunked prefill 等价、向量输入格式与归一化、
  视觉 patch 顺序 / mask / connector / image token 注入、错误路径与新模块的依赖边界。
- v0.1 的检查没有被删掉，负向用例保留。
- `tools/benchmark.py` 仍然只测文本生成；向量与图文的耗时通过各自 CLI 的 JSON 输出测量。

### 发布前修复（审查后）

审查发现 5 个问题，全部修掉并各自补了能证明原缺陷的回归测试。

- **向量分块等价测试此前是空转的。** 它构造了带小 `prefill_chunk_size` 的引擎，
  然后调 `model.encode_ids(ids)` —— 分块大小没有传下去，7 个用例全部退化成
  一次 50 token 的整块 prefill。现在每个用例都断言 decoder **实际看到的分块长度**，
  覆盖 `chunk=1`、不能整除的长度和最后一个短块，既比数值也比分块确实发生。
- **manifest 的 `model_family` 此前不与 `config.json` 的 `model_type` 交叉校验。**
  把 SmolLM2 目录的 manifest 改成 `qwen3` 也照样加载。要说清楚的是，**解码规则一直
  来自 `config.json`**：实测该目录的 `config.family` 仍是 `llama`、`qk_norm` 仍是
  `False`，所以权重不会因此被算错，也不会多出 Q/K RMSNorm。真正的缺陷是这个目录
  从此自相矛盾而没有任何一方拒绝它 —— `slowq inspect` 会照 manifest 报出错误的家族，
  而 `config.json` 被换成另一家族时权重确实会被按错误规则解码。现在不一致时抛
  `ConfigError`，说明两边各声称什么、并指出重新转换的命令；
  版本 1 的 manifest 没有这个键，仍然可读，不会被要求补上它当年没写过的字段。
  图文目录**不走这个函数**：`slowq.vlm` 有自己的 `load_config_and_weights`，
  在自己的加载器里分别校验 `config.json` 的 `model_type` 与 manifest 的
  `model_family`，两侧都必须等于 `idefics3`。
- **向量参考侧此前把 SlowQ 拼好的字符串当成输入。** 参考侧对 SlowQ 记录的
  `text` 做分词，于是一个写错的 query 前缀会被两侧同时继承，比较照样 `PASS`。
  现在参考侧从原始 query / document、instruction 和角色**自己重建**输入，
  不调用被测的 `format_query`，也不把 SlowQ 的最终文本当作事实来源；
  官方格式 `Instruct: {instruction}\nQuery:{query}`（`Query:` 后无空格）与
  "document 不加任何前缀"都由参考侧独立断言。抓取配置新增
  `embedding_input_format=2`：**只影响向量任务的证据**，旧向量 artifact 会被
  明确拒绝并提示重抓 `ref`，生成与图文的 artifact 版本不变、无需重抓。
- **删除了 `ops.padding_attention_mask`。** 它对 `np.array([True, False])` 抛
  `IndexError`。先确认了运行时、工具、测试与文档里都没有调用者 ——
  视觉塔走非因果注意力且单图路径不产生 padding，真出现部分 padding 的 patch
  网格时报错而不是屏蔽 —— 所以按"不留无人使用且已知有缺陷的代码"处理，
  而不是修好一个没有调用者的函数。
- **图文请求的校验此前发生在视觉编码之后。** `Engine.stream_ids` 是生成器函数，
  它的校验要到第一次 `next()` 才执行，而 `VisionEngine.stream_prompt` 是普通函数，
  于是"要 0 个新 token"这种非法请求会先把视觉塔跑完一遍才报错。校验逻辑抽成
  `Engine._validate_request`，两条路径共用同一份实现，图文路径在编码图像**之前**调用它。

### 验证工具修复（第二轮审查后）

只改验证工具、测试与文档，没有动前向计算、采样行为、数值容差或支持范围。

- **`compare` 此前会对缺失的输入字段给假 PASS。** 向量 artifact 的比较用
  `ours_entry.get(field) == theirs_entry.get(field)` 判 `raw` / `is_query` / `text`，
  于是两侧都缺同一个字段时两边都是 `None`，`None == None` 成立——
  "两边都没说自己嵌入了什么"被判成"两边说的一样"，打印 `input=ok` 并以退出码 0 通过。
  现在逐侧逐字段判"键在不在 / 是不是 null / 类型对不对"，不合格的记 `input=INVALID`、
  写进 problems、退出码非 0，错误信息带上哪一侧、哪个样例、哪条文本、哪个字段。
  `is_query` 用 `isinstance(value, bool)` 判而不是和 `True` 比相等：Python 里 `bool`
  是 `int` 的子类，JSON 的 `1` 会被当成"是 query"。原有的数值、tokenizer、权重摘要、
  pooling 与格式版本检查一条没减。
- **生产的参考抓取函数此前没有测试真的执行过。** `_reference_embedding_capture`
  是"参考侧自己重建输入"这条修复的核心，但把它的函数体第一句换成 `raise AssertionError`，
  当时那 91 项测试全绿。现在 4 条用例直接调用它：mock 的只是它外围的 torch 接口、
  tokenizer（记录被喂了哪些字符串）与一个返回 `(batch, seq, hidden)` 的假模型，
  没有在测试里另写一份"正确的抓取"；8 条官方字符串逐字写在测试里，
  不调用被测的 `reference_embedding_texts`。

### 结果

- 快速测试：发布前修复轮 **665** 项通过，验证工具修复轮 **685** 项通过
  （无网络、无真实权重；+20 全在 `tests/test_validate_reference.py`，91 → 111）。
- 8 个模型的真实权重集成测试合计 208 项通过、0 失败，逐目录单独设置 `SLOWQ_MODEL_DIR`
  ——**发布前修复轮的结果**，验证工具修复轮没有重跑，8 份原始日志已留档在
  `artifacts/repair-review-20261021/developer-integration-logs/`（sha256 与审查方索引一致）。
  其中 Qwen3-1.7B 该轮耗时 4036.71 s。**完整正确性回归已通过；异常耗时与严重内存压力、
  页面调入调出相符，尚未通过资源充足环境的对照测量排除其他性能因素。**
  权重是 `np.load(..., mmap_mode="r")` 加载的，页面驻留由操作系统管理，
  映射不等于常驻，所以峰值 RSS 6.43 GiB 是观测值而不是结构性推论；
  `sys` 845 s 是 `user` 126 s 的 6.7 倍、CPU 利用率 24% 与之一致。
  62.7 s 那个量级是**接入批次**的实测，不是修复轮的。
- 8 个模型的官方参考数值比较全部 `PASS`：6 个生成模型各 116 项，向量与图文各 82 项，
  合计 860 个张量，greedy argmax 位置全部一致。
  参考环境 transformers 4.51.3 / torch 2.14.1 / eager / FP32 / CPU。
  向量那一侧的证据在发布前修复轮按新的 `embedding_input_format=2` **重新抓取**；
  验证工具修复轮对同一批 artifact **重新比较**，8 / 8 仍 `PASS`、仍是 860 个张量。
- 干净环境安装验证（发布前修复轮）：wheel → 两个新建 venv（无 Pillow 时 660 passed / 5 skipped，
  装上 `[vision]` 后 665 passed / 0 skipped），`torch` / `transformers` 始终不存在，
  `pip check` 干净，三类任务各跑通一条真实离线命令。
- 全部在 macOS arm64 / Apple M5 Pro 上执行；v0.2 新增模型没有 Linux 验证记录。

## v0.1.0

首个实验版本支持 Qwen3-0.6B 的 CPU / FP32 推理。以下为公开提交前完成的主要修复。
详细验证方法见[测试与数值验证](validation.md)。

### 权重转换与下载

- 转换先在暂存目录完成并自检，再安装到目标路径；默认拒绝覆盖非空目录。
- 接受预先创建的空输出目录，安装时拒绝覆盖后来写入其中的文件。
- 下载缓存绑定模型仓库与完整 commit；记录未完成状态，中断后可重跑命令补齐文件。
- 缓存文件按大小和可用的 SHA-256 校验，无来源标记的目录仍会被拒绝。
- 跨分片的 tied weights 分别读取各自分片进行比较。
- 截断的 checkpoint 给出张量、分片和字节范围错误，避免清理 mmap 时掩盖原始异常。

### 生成与数值验证

- 流式生成结束时，将解码器 flush 的尾部文本一并交付，保持流式与完整输出一致。
- `DebugProbe` 按指定位置捕获 logits，支持非末尾位置。
- 数值比较检查预期张量集合、形状和有限性；缺失数据、空集合、NaN / Inf 都会失败。
- tokenizer 样例与权重摘要也检查预期集合，双方同时遗漏同一项时仍会报错。
- 集成测试重新计算当前 tokenizer 输出和权重摘要，不复用历史布尔结论。

### 基准与发布文件

- 分开测量 full / chunked prefill，报告实际前向次数。
- `--threads 4` 与 `--threads=4` 都在导入 NumPy 前设置线程环境变量。
- 补齐 MIT LICENSE，修正转换依赖说明，并配置真实的 GitHub 仓库地址。

数值容差保持不变；测试结果及平台范围见[验证记录](validation.md)。
更完整的早期修复复盘保留在[首次提交的 README](https://github.com/luolinglong0611/SlowQ/blob/b5548dca2faa8e85a078ebdec7d120d4940e3d7c/README.md)中。
