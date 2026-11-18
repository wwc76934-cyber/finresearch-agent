# FinResearch Agent

A 股多智能体投研系统，含金融数据中台（PIT + 血缘）与 Agent 评测中台（B0–B4 基线 + 消融）。v2.0 起支持全 A 股按需在线拉取、假设驱动的开放问题、多公司对比、跨期观点复盘和流式进度。

> 数据：**8 家真实 A 股**（宁德时代、亿纬锂能、比亚迪、隆基绿能、贵州茅台、五粮液、美的集团、格力电器；东方财富三表 + 巨潮原始财报 PDF + Yahoo 日线），以及 6 家**虚构“实验室”公司**（99000x，用于故障注入）。仅供研究，不构成投资建议。

## 快速开始

```bash
uv venv .venv && uv pip install -e ".[dev]"      # 或 pip install -e ".[dev]"
python scripts/bootstrap.py                       # 入库两套快照 + 质量校验 + 切片索引（SQLite: data/finresearch.db）
python scripts/run_research.py 300750.SZ 2025-12-31 --pdf   # 宁德时代 B4 研报（Markdown + JSON + PDF）→ reports/
python -m uvicorn apps.api_server.main:app --host 127.0.0.1 --port 8000   # 打开 http://127.0.0.1:8000
python scripts/run_experiments.py                 # 离线矩阵：18 架构 × 195 题 + 检索评测 → reports/eval/EVALUATION_generated.md
python -m pytest -q                               # 约 170 个测试（含回归集、真实数据、在线穿透、流式接口）
mypy packages apps                                # 新模块严格；旧模块的类型债务见 pyproject.toml
```

真实快照已提交在 `benchmarks/real_snapshot/`。要从头重建（需联网，约 200MB 原始缓存、解析约 1 小时）：

```bash
python scripts/fetch_real.py --pdfs
python scripts/build_real_snapshot.py
```

Windows 下把 `python` 换成 `.venv\Scripts\python.exe`。

### 前端（React + TypeScript + Tailwind）

```bash
cd apps/web_react && npm install && npm run build   # 生成 dist/，由 FastAPI 在 http://127.0.0.1:8000 提供
npm run dev                                          # 开发模式 http://127.0.0.1:5173，API 代理到 8000
```

未构建时 `/` 只显示构建提示（原生 JS 版 `web_workspace` 已在 v1.2 删除）；API 文档在 `/docs`。前端含：投研工作台（含 Trace 时间线）、评测中台、数据质量、因子回测。

### v2.0：全市场、开放问题、对比、记忆、流式

| 能力 | 怎么用 | 细节 |
|---|---|---|
| 全 A 股在线穿透 | 工作台里搜任意 A 股；或 `ResearchRequest(data_mode="hybrid_live")` | 本地没有（或本地数据早于 as_of）就按需拉取并按披露日入库；`offline_golden` 只读冻结快照，所有基准和测试用它 |
| 同业对比 | 问题里点名 2–4 家，或在选择器里选多家 | 共同的最新报告期，所有数字来自计算器；银行与制造业混比时毛利率标"不适用" |
| 开放问题 | 直接问"海外业务、客户集中度、资本开支转化……" | 规划器拆成子假设，子图最多 2 个检索短语 + 1 次改写去找年报原文，找不到就写进"待解答" |
| 观点记忆与复盘 | 对同一公司在更晚的时点再研究一次 | 上次的证伪条件会被重新检验：`CONFIRMED` / `WEAKENED` / `FALSIFIED` |
| 流式进度 | `POST /api/research/stream` | 首个事件在任何计算开始之前发出；8 类事件见 `apps/api_server/stream.py` |
| 同业与北交所行情 | `python scripts/akshare_collect.py industry`、`... prices --tickers 430047.BJ` | 在能访问 AKShare 的网络里运行，输出放进 `data/akshare_cache/` |

数据模式环境变量：`FINRESEARCH_DATA_MODE=offline_golden|hybrid_live|online_force_refresh`（默认 hybrid_live）。设计、取舍和已知限制见 `docs/V2_REVIEW_AND_PLAN.md`。

### Demo 录屏

![demo](reports/demo/demo.gif)

```bash
LLM_OFFLINE=1 python -m uvicorn apps.api_server.main:app --host 127.0.0.1 --port 8000
node scripts/record_demo.mjs http://127.0.0.1:8000 reports/demo   # 需本机 Chrome 或 Edge
```

流程：选 990001 → 小预算触发审批卡 → 批准 → 动态追加追查 → 研报 → 证据血缘抽屉 → 评测中台 → 数据质量。录制使用真实 API、离线模型。

### 接入真实 LLM

在 `.env` 设置 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL_STRONG`（任意 OpenAI 兼容接口）。不设置时全部 Agent 走确定性模板路径，结果可复现。无论哪种模式，财务数字都只来自确定性算子。

### Docker（PostgreSQL 16 + pgvector + OpenTelemetry）

三个开关，默认都不需要任何服务（单元测试和 CI 用默认值）：

| 变量 | 默认 | 生产值 | 作用 |
|---|---|---|---|
| `STORAGE_BACKEND` | `sqlite` | `postgres` | 关系库。显式的 `DATABASE_URL` 优先 |
| `VECTOR_BACKEND` | `memory` | `pgvector` | 稠密检索：`halfvec(2048)` + HNSW，元数据预过滤（ticker、`available_at`、口径、章节）在 SQL 里完成，再与内存 BM25 做 RRF |
| `TRACING_BACKEND` | `json` | `otel` | 额外产出 OpenTelemetry span（`research_run` → `langgraph.node.*` → `tool.call.*` / `llm.completion.*` / `verifier.check`）。设了 `OTEL_EXPORTER_OTLP_ENDPOINT` 就发到该地址，否则留在内存 |

```bash
cp .env.example .env
docker compose up -d db                          # 只起数据库（pgvector/pgvector:pg16）
STORAGE_BACKEND=postgres VECTOR_BACKEND=pgvector python scripts/bootstrap.py   # 入库 + 写向量
docker compose up -d --build                     # 数据库 + API + 前端，http://127.0.0.1:8000
docker compose --profile observability up -d     # 加 Phoenix，http://127.0.0.1:6006
python scripts/check_pg_parity.py                # pgvector 与内存检索对照 → reports/eval/pg_parity.json
```

## 目录

| 路径 | 内容 |
|---|---|
| `packages/data_platform` | Registry、Ingestion、Schema、Quality、Catalog、Lineage、Versioning、Serving、Calculators |
| `packages/rag_engine` | 结构感知切片、BM25/Dense/RRF/Rerank、路由、Self-RAG |
| `packages/agent_runtime` | LangGraph 主图与子图、Planner、Fundamental/Market/News/Investigate、Debate、Risk、Verifier、Report、B0–B2 |
| `packages/evaluation_platform` | 140 题基准（golden/blind/chaos/lab）+ 独立 Oracle、Runner、Evaluators、Tracer、Comparison、Error Analysis、Regression、RAG eval |
| `apps/api_server` | FastAPI + SSE |
| `apps/web_react` | 投研工作台 + Trace 时间线 / 评测中台 / 数据质量 / 因子回测（React 19 + TS + Tailwind + ECharts） |
| `benchmarks/` | Golden Snapshot、`tasks/golden_v1.jsonl`(30)、`tasks/chaos_v1.jsonl`(50)、`regression/` |
| `reports/cases/` | 3 份 B4 案例研报（Markdown + 完整 JSON 含 trace） |

## 文档

* [ARCHITECTURE.md](ARCHITECTURE.md)：状态机、上下文隔离、Verifier、与 PRD 的差异
* [DATA_CONTRACTS.md](DATA_CONTRACTS.md)：三时间戳 PIT、契约、血缘、快照、演示数据场景
* [EVALUATION.md](EVALUATION.md)：基线与消融结果、解读、局限
* [INTERVIEW_PLAYBOOK.md](INTERVIEW_PLAYBOOK.md)：架构决策、踩坑与量化亮点
