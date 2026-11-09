# FinResearch Agent

A 股多智能体投研系统，含金融数据中台（PIT + 血缘）与 Agent 评测中台（B0–B4 基线 + 7 项消融）。

> 仓库自带的数据是 **6 家虚构演示公司**（代码 99000x，名称“演示…”），用于离线复现与评测。它们不是行情数据，不构成投资建议。

## 快速开始（无需任何服务）

```bash
uv venv .venv && uv pip install -e ".[dev]"      # 或 pip install -e ".[dev]"
python scripts/build_golden_snapshot.py           # 生成演示快照（已提交，可跳过）
python scripts/bootstrap.py                       # 入库 + 质量校验 + 切片索引（SQLite: data/finresearch.db）
python scripts/run_research.py 990001.SZ 2025-12-31   # 跑一份 B4 研报 → reports/
python -m uvicorn apps.api_server.main:app --host 127.0.0.1 --port 8000   # 打开 http://127.0.0.1:8000
python scripts/run_experiments.py                 # 80 题 × 12 架构 + 检索评测 → reports/eval/ 与 EVALUATION.md 表格
python -m pytest -q                               # 35 个测试（含回归集）
```

Windows 下把 `python` 换成 `.venv\Scripts\python.exe`。

### 接入真实 LLM

在 `.env` 设置 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL_STRONG`（任意 OpenAI 兼容接口）。不设置时全部 Agent 走确定性模板路径，结果可复现。无论哪种模式，财务数字都只来自确定性算子。

### Docker（PostgreSQL 16）

```bash
cp .env.example .env    # 设置 POSTGRES_PASSWORD
docker compose up --build   # http://127.0.0.1:8000
```

## 目录

| 路径 | 内容 |
|---|---|
| `packages/data_platform` | Registry、Ingestion、Schema、Quality、Catalog、Lineage、Versioning、Serving、Calculators |
| `packages/rag_engine` | 结构感知切片、BM25/Dense/RRF/Rerank、路由、Self-RAG |
| `packages/agent_runtime` | LangGraph 主图与子图、Planner、Fundamental/Market/News/Investigate、Debate、Risk、Verifier、Report、B0–B2 |
| `packages/evaluation_platform` | 80 题基准 + 独立 Oracle、Runner、Evaluators、Tracer、Comparison、Error Analysis、Regression、RAG eval |
| `apps/api_server` | FastAPI + SSE |
| `apps/web_workspace` | 投研工作台 / 评测看板 / 数据质量 |
| `benchmarks/` | Golden Snapshot、`tasks/golden_v1.jsonl`(30)、`tasks/chaos_v1.jsonl`(50)、`regression/` |
| `reports/cases/` | 3 份 B4 案例研报（Markdown + 完整 JSON 含 trace） |

## 文档

* [ARCHITECTURE.md](ARCHITECTURE.md)：状态机、上下文隔离、Verifier、与 PRD 的差异
* [DATA_CONTRACTS.md](DATA_CONTRACTS.md)：三时间戳 PIT、契约、血缘、快照、演示数据场景
* [EVALUATION.md](EVALUATION.md)：基线与消融结果、解读、局限
* [INTERVIEW_PLAYBOOK.md](INTERVIEW_PLAYBOOK.md)：架构决策、踩坑与量化亮点
