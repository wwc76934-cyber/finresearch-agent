# v2.0 PRD：可行性与合理性审查，以及落地计划

审查日期 2026-11-18。基线 = 仓库 `44634f0`（v1.2）。PRD 文件：`c6f4e1b1…_finresearch_agent_prd_v2_0_enterprise.md`。

## 1. 与仓库现状不符的地方

| PRD 的说法 | 实际 | 处理 |
|---|---|---|
| 基线 175 题 n=3 准确率 85.1% ± 0.6% | 离线 B4 96.0%（168/175），在线 75 题 88.0%。没有 85.1% 这个数 | 零退化门槛改为：离线 B4 ≥ 168/175，PIT 泄漏 0 |
| v1.2 `mypy 0 error` | 仓库从未配置过 mypy。首次运行：513 个错误，28 个文件（`factors.py` 一个文件 405 个，全是 pandas 类型） | 加 mypy 配置：新模块严格，27 个旧模块列入债务名单并记数。CI 加 `mypy packages apps` |
| 当前是"同步等待 42 秒" | v1.1 起已有 SSE 节点事件流（`/api/research/{job}/events`），前端已按节点点亮 | Epic 3 改为扩展：新增 `POST /api/research/stream`，补齐 8 类事件，不是从零做 |
| 用 AKShare / 新浪拉数 | 本机网络上 AKShare、东财 push2 被拦 | 只用已验证可用的三个源：东财 F10 datacenter、巨潮、Yahoo。全市场名单取自巨潮 `szse_stock.json`（6,189 只 A 股） |
| 同申万二级行业市值前 5 名同行实时拉取 | 没有可用的行业成分接口（push2 被拦） | 同行 = 本地库里同行业的公司；新公司的同行集合会偏小，报告要写明 |
| 全部在线数据"严守 PIT" | 东财对每个历史期只给**最新值**。公司事后重述，早期 as_of 就能看到重述值。8 家核心公司用 PDF 逐项核对过，新公司没有 | 在线入库的数据在报告里带固定告警，不声称已核对。不隐藏 |
| `data_mode` 默认 `hybrid_live` | 数百个测试和评测脚本直接构造 `ResearchRequest`，会碰网络 | 默认读环境变量 `FINRESEARCH_DATA_MODE`（默认 hybrid_live）；评测 runner 与 `conftest` 强制 `offline_golden` |
| 耗时目标：财报 ≤3.5 s、行情 ≤2.0 s、公告 ≤3.0 s | 这些是公共接口，没有 SLA。实测并行拉取约 4–5 s（财报+行情+公告并发） | 如实报测量值；超时降级为 `partial`，不阻塞 |
| 金融业模板（银行/保险/券商） | 现有指标（毛利率、存货等）对金融业没有意义 | 只做识别和告警：写入 `template=bank` 并提示，不伪造金融业比率 |
| 子图内 ReAct-Lite 最多 2 轮 | 每多一次模型调用就吃掉 Step 4 刚省下的时间 | 只在 `full` 路由、且确定性检测到证据缺口时触发；离线不运行，只能在线验证 |
| 复杂题集通过率 ≥80%，由 pytest 验证 | 离线只能验证结构，不能验证模型的开放式推理 | 离线测试管线；通过率用在线运行补测，并标明哪些是离线、哪些是在线 |
| 多股对比用笛卡尔积扇出 | 单个 `RunContext` 里的证据、结论、血缘都按一个 ticker 编号 | 在同一个 `RunContext` 里算所有公司的确定性指标，由 comparator 节点汇成矩阵；只对主标的跑完整子图 |

## 2. 合理的地方

* 三种 `data_mode` 与"评测只读冻结快照"是对的，它保住了 175 题可复现。
* 假设驱动的子问题树、观点记忆与自动复盘有价值，且复盘可以做成纯确定性（证伪条件本来就是 MetricRecord 阈值）。
* 记忆必须在 `offline_golden` 下关闭（读和写都关），否则前一次运行会影响评测。PRD 没写，这里加上。

## 3. 任务清单（按 PRD 的步骤，依赖关系不变）

| 步 | 内容 | 验证 | 状态 |
|---|---|---|---|
| 1 | 全 A 股名单、`ReadThroughIngestor`、`ingestion_watermarks`、`data_mode`、mypy 配置与 CI | `tests/test_live_read_through.py`；真实接口 `RUN_LIVE_TESTS=1`（300308.SZ、601899.SH） | ✅ 本次完成 |
| 2 | `SubHypothesis` 规划器、ReAct-Lite 子图、`tickers` 多股与 comparator | `tests/test_complex_and_multiticker.py` | 未开始 |
| 3 | `research_theses` 表、`ThesisMemoryStore`、跨期复盘章节 | `tests/test_thesis_memory.py` | 未开始 |
| 4 | `POST /api/research/stream`（8 类事件）、前端全 A 股搜索、多股选择、同步进度条、复盘卡片 | TTFT 实测；前端构建 | 未开始 |
| 5 | 20 题复杂开放与多股对比集（评测 195 题）、文档 | 离线回归 + 说明在线部分 | 未开始 |

## 4. Step 1 实测

* 全 A 股名单：6,189 只，缓存在 `data/a_share_catalog.json`；按代码、简称、拼音首字母、文本中提取。
* 对 `601899.SH` 与 `300308.SZ`（都不在原 8 家里）的首次穿透：约 4.2–4.8 s，写入 444 条财报事实、734 根日线、6 篇公告；第二次运行 0.4 s（水位线命中）。
* PIT：`published_at` 取东财 `NOTICE_DATE`，`available_at` 取其后第一个交易日 09:30；缺失或不晚于报告期末时退回法定截止日并标 `pit_estimated`；财报期末从未被当作发布时间。
* 回归：117 个测试通过（含 11 个新测试；2 个真实接口测试默认跳过），离线 B4 仍是 168/175。
* mypy：`mypy packages apps` 通过；债务名单 27 个旧模块（首次运行共 513 个错误）。

## 5. 已知限制（Step 1）

* 新公司的财报值未经 PDF 核对（见上）。
* 公告只入库带关键词的非定期报告，最多 6 篇、只读前 4 页。整本定期报告（Tier-B）按需解析还没做，对应异常追查会落到 `open_questions`。
* pgvector 后端下，在线新增的片段没有增量写入向量表；该后端会退回内存索引并给出警告。
* 北交所公司没有行情源（Yahoo 不支持），只入库财报和公告。
