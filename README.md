# FinResearch Agent

A 股多智能体投研系统，含金融数据中台（PIT + 血缘）与 Agent 评测中台（B0–B4 基线 + 7 项消融）。

> 数据：**8 家真实 A 股**（宁德时代、亿纬锂能、比亚迪、隆基绿能、贵州茅台、五粮液、美的集团、格力电器；东方财富三表 + 巨潮原始财报 PDF + Yahoo 日线），以及 6 家**虚构“实验室”公司**（99000x，用于故障注入）。仅供研究，不构成投资建议。

## 快速开始

```bash
uv venv .venv && uv pip install -e ".[dev]"      # 或 pip install -e ".[dev]"
python scripts/bootstrap.py                       # 入库两套快照 + 质量校验 + 切片索引（SQLite: data/finresearch.db）
python scripts/run_research.py 300750.SZ 2025-12-31 --pdf   # 宁德时代 B4 研报（Markdown + JSON + PDF）→ reports/
python -m uvicorn apps.api_server.main:app --host 127.0.0.1 --port 8000   # 打开 http://127.0.0.1:8000
python scripts/run_experiments.py                 # 离线矩阵：12 架构 × 140 题 + 检索评测 → EVALUATION.md 表格
python -m pytest -q                               # 56 个测试（含回归集、真实数据测试）
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

未构建时 `/` 自动退回免构建的原生 JS 版（也可访问 `/legacy`）。

### Demo 录屏

![demo](reports/demo/demo.gif)

```bash
LLM_OFFLINE=1 python -m uvicorn apps.api_server.main:app --host 127.0.0.1 --port 8000
node scripts/record_demo.mjs http://127.0.0.1:8000 reports/demo   # 需本机 Chrome 或 Edge
```

流程：选 990001 → 小预算触发审批卡 → 批准 → 动态追加追查 → 研报 → 证据血缘抽屉 → 评测中台 → 数据质量。录制使用真实 API、离线模型。

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
| `packages/evaluation_platform` | 140 题基准（golden/blind/chaos/lab）+ 独立 Oracle、Runner、Evaluators、Tracer、Comparison、Error Analysis、Regression、RAG eval |
| `apps/api_server` | FastAPI + SSE |
| `apps/web_workspace` | 投研工作台 / 评测看板 / 数据质量 |
| `benchmarks/` | Golden Snapshot、`tasks/golden_v1.jsonl`(30)、`tasks/chaos_v1.jsonl`(50)、`regression/` |
| `reports/cases/` | 3 份 B4 案例研报（Markdown + 完整 JSON 含 trace） |

## 文档

* [ARCHITECTURE.md](ARCHITECTURE.md)：状态机、上下文隔离、Verifier、与 PRD 的差异
* [DATA_CONTRACTS.md](DATA_CONTRACTS.md)：三时间戳 PIT、契约、血缘、快照、演示数据场景
* [EVALUATION.md](EVALUATION.md)：基线与消融结果、解读、局限
* [INTERVIEW_PLAYBOOK.md](INTERVIEW_PLAYBOOK.md)：架构决策、踩坑与量化亮点
