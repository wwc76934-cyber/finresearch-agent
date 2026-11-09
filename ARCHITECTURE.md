# ARCHITECTURE

## 1. 四层结构

```text
apps/api_server        FastAPI + SSE：研究任务、血缘、评测、数据质量 API
apps/web_workspace     三栏工作台 + /evaluation + /data-health（原生 JS + ECharts）
packages/agent_runtime LangGraph 主图、4 个子图、辩论、风险、双层 Verifier、基线 B0–B2
packages/rag_engine    结构感知切片、BM25/Dense/RRF/Rerank、路由、Self-RAG
packages/data_platform 8 个子模块：registry ingestion schema quality catalog lineage versioning serving(+calculators)
packages/evaluation_platform datasets runner evaluators tracing comparison error_analysis regression rag_eval
```

依赖方向只向下：`agent_runtime → rag_engine → data_platform`，`evaluation_platform` 依赖前三者，前三者只从 `evaluation_platform` 引入 `tracing`。

## 2. 主图（LangGraph StateGraph）

```text
START → plan ──Send×N──▶ collect ──▶ gather ──(HIGH 异常且 depth<2 且预算够)──▶ replan ──Send×≤2──▶ collect
                                       │
                                       └──▶ debate → risk → inject(混沌) → verify ──(有失败)──▶ rewrite ─┐
                                                                             │◀─────────────────────────┘
                                                                             └──▶ report → END
```

| 节点 | 文件 | 作用 |
|---|---|---|
| plan | `agents/planner.py` | 解析问题为 `Ask` 列表（指标/异常/事件/同行/不可回答），生成 `ResearchPlan` 子任务与预算；LLM 在线时由 LLM 解析，离线用正则规则；LLM 给出未可见报告期会被改回最新可见期 |
| collect | `agents/collectors.py` | 用 `Send` 并行执行子图：fundamental / market / news / investigate。每个子图是独立编译的 `StateGraph(SubState)` |
| gather | `graph.py` | Map 阶段汇合点（LangGraph superstep 保证所有 Send 完成后执行一次） |
| replan | `graph.py` | `RouteAfterCollection`：存在未处理的 `HIGH` 级 `AnomalyFlag`、`replan_depth < max_replan_depth(2)`、已用 tokens 未超预算 → 派生最多 2 个追查子任务 |
| debate | `agents/debate.py` | 证据绑定式多空审议，最多 2 轮，无新证据即停；Bear 每轮最多 2 次反证检索；Manager 输出证伪条件 |
| risk | `agents/risk.py` | 独立于多空结论的六维规则评分 |
| inject | `agents/faults.py` | 评测用故障注入（假数字 / 泄漏未来公告 / 工具报错） |
| verify / rewrite | `agents/verifier.py` | Tier-1 + Tier-2；失败只重写失败的 Claim（局部回滚）；第 3 次仍失败即熔断降级为 `UncertaintyStatement` |
| report | `agents/report.py` | 只收录 VERIFIED / PARTIALLY_SUPPORTED 论断，生成 Markdown、ECharts 配置与引用表 |

Checkpointer：`MemorySaver`，`thread_id = run_id`。持久化 checkpointer（`langgraph-checkpoint-postgres`）可直接替换 `build_graph(checkpointer=...)`。

## 3. Context Engineering：指针式状态

* 父图 `GraphState` 只保存 `SubTaskSummary`（摘要 + `evidence_ids` + `metric_ids`），不保存原始表格或工具返回。
* 子图私有状态 `SubState.scratch` 存放原始报表与中间结果，子图结束即丢弃。
* 证据与指标保存在运行级 `RunContext.evidence`（EvidenceStore）与 `FinancialCalculator.store`（MetricStore）中，下游节点按 id 解引用。
* `context_isolation=False` 的消融会把所有子图的原始输出拼进下游上下文，用于测量 tokens 差异（实测下游上下文 tokens 增加到 6.7 倍）。

`RunContext` 不放进 LangGraph 状态，而是按 `run_id` 注册在进程内字典里：状态可序列化、可 checkpoint，重对象不进入序列化。

## 4. 计算与推理解耦

* 所有数字来自 `data_platform/calculators.py` 产出的 `MetricRecord`（带输入 fact_id、公式版本、状态码）。
* 论断文本由“渲染函数”从 MetricRecord 生成；回滚时重新渲染即可修复被篡改的数字。
* LLM 只用于：问题解析、摘要、Manager 分歧概括、Tier-2 蕴含判断、执行摘要。凡是 LLM 输出的文本，校验函数都要求不含数字。
* 经济上无意义的比率返回状态码而不是数字：`NEGATIVE_EARNINGS_NA`、`ZERO_DENOMINATOR_NA`、`NEGATIVE_EQUITY_NA`、`INSUFFICIENT_DATA`、`RESTATEMENT_BREAKS_SPLIT`。

## 5. 双层 Verifier

| 层 | 检查 | 失败类型 |
|---|---|---|
| Tier-1（确定性，毫秒级，0 token；耗时写在 trace 的 `run_deterministic_validators` 结果里） | 每个数字须等于所引 MetricRecord 在其显示精度下的值，或逐字出现在所引原文中 | `NUMBER_MISMATCH` / `WILD_NUMBER` |
| | 所引证据 `available_at ≤ as_of` | `PIT_VIOLATION` |
| | 证据 id 存在于本次运行 | `UNKNOWN_EVIDENCE` / `MISSING_CITATION` |
| | 同一论断混用母公司与合并口径且未标注 | `SCOPE_MIX` |
| Tier-2（LLM Judge；离线启发式） | SUPPORTED / PARTIALLY_SUPPORTED / CONTRADICTED / UNSUPPORTED | `UNSUPPORTED` / `CONTRADICTED` |

回滚动作：剔除违规证据并删去其引文、按渲染函数重写数字、去掉母公司口径证据、弱化推断措辞。每个论断最多 2 次重写。

## 6. Agentic RAG

* 解析：markdown + `<!-- page N -->`（MinerU/Docling/pdfplumber 的输出格式）。表格整表保留，跨页续表合并，页码记录为区间。
* 父子切片：父 = 一个小节/整表；子 = 段落或按行分组的表格（表头重复）。每个切片头部注入 `[代码 | 报告期 | 章节路径 | 表名 | 口径 | 单位 | 页码]`。
* 检索：BM25、Hashing 稠密向量（离线替身，可换 bge）、RRF(k=60)、词覆盖交叉打分（可换 bge-reranker）。`ticker` 与 `available_at ≤ as_of` 始终作为前置过滤；口径与章节类型可选。
* 路由：问“是多少”走结构化算子，问“为什么”走文档检索。
* Self-RAG：命中只有表格或无解释句时判定 `INSUFFICIENT_EXPLANATION`，补同义词并改章节过滤，最多 2 次。

## 7. 可观测性

`evaluation_platform/tracing.py` 的 `Tracer` 记录每个节点的耗时、工具调用（入参/结果/错误）、LLM 用量、异常标记，用 `contextvars` 保证并行子图的调用归属正确；同一事件流推给 API 的 SSE。运行结束写入 `agent_executions`。

## 8. 与 PRD 的差异

| PRD 要求 | 当前实现 | 原因 |
|---|---|---|
| React 18 + TS + Tailwind | 原生 JS + ECharts 单页 | 不需要构建链即可运行；页面结构与功能按 PRD 第 8 章实现 |
| PostgreSQL 16 | SQLite 默认，docker-compose 提供 PG16 | 零依赖本地运行；SQLAlchemy 模型两者通用 |
| PGVector/Chroma + bge | 内存索引 + Hashing 稠密向量 | 离线无模型下载；`DenseEncoder`/`Reranker` 协议是替换点 |
| 真实 A 股 Golden Snapshot | 6 家虚构公司 | 当前环境无法访问行情/公告源；`scripts/ingest_live.py` 提供 AKShare 行情接入 |
| Human-in-the-Loop 审批卡片 | 未实现 | 预算超限时只会跳过重规划，不会暂停等待审批 |
| PDF 导出 | Markdown 导出 + 浏览器打印样式 | — |
| OpenTelemetry | 自研 Tracer | 字段与 OTel span 对齐，可加 exporter |
| P2：同行横向对比矩阵、因子 IC、Reflection 记忆、定时监控 | 只有同行 PE 中位数对比 | 属 P2 |
