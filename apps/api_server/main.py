"""FastAPI gateway: research runs with SSE node streaming, lineage, evaluation and data-health APIs.

  uvicorn apps.api_server.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from sqlalchemy import Integer, cast, func, select  # noqa: E402

from agent_runtime import ARCHITECTURES, ResearchRequest, run_research  # noqa: E402
from data_platform import quality  # noqa: E402
from data_platform.db import (DocumentChunkRow, FinancialFactRow, ResearchRunRow, SecurityRow,  # noqa: E402
                              SourceDocumentRow, get_session_factory)
from data_platform.ingestion import GoldenFileProvider  # noqa: E402
from data_platform.lineage import LineageRecorder  # noqa: E402
from data_platform.versioning import list_snapshots  # noqa: E402

app = FastAPI(title="FinResearch Agent", version="0.1.0")
EVAL = ROOT / "reports" / "eval"
Session = get_session_factory()
JOBS: dict[str, dict] = {}


class ResearchIn(BaseModel):
    ticker: str
    as_of: datetime
    question: str = "全面研究公司基本面、估值与风险，形成包含多空审议与证伪条件的研究报告"
    mode: str = "deep"
    architecture: str = "B4"
    token_budget: int = Field(200_000, ge=1000, le=2_000_000)
    hitl: bool = True
    hitl_always: bool = False


@app.get("/api/securities")
def securities():
    with Session() as s:
        return [{"ticker": r.ticker, "name": r.name, "industry": f"{r.industry_sw_l1}/{r.industry_sw_l2}",
                 "is_demo": r.is_demo} for r in s.scalars(select(SecurityRow).order_by(SecurityRow.ticker))]


@app.get("/api/architectures")
def architectures():
    return ["B0", "B1", "B2"] + list(ARCHITECTURES)


@app.post("/api/research")
def start_research(body: ResearchIn):
    if body.architecture not in architectures():
        raise HTTPException(400, f"unknown architecture {body.architecture}")
    job_id = uuid.uuid4().hex[:12]
    q: queue.Queue = queue.Queue()
    JOBS[job_id] = {"queue": q, "status": "running", "run_id": None}

    def work():
        try:
            req = ResearchRequest(ticker=body.ticker, as_of=body.as_of, question=body.question, mode=body.mode,
                                  token_budget=body.token_budget, hitl=body.hitl, hitl_always=body.hitl_always)
            _settle(job_id, run_research(req, body.architecture, listener=q.put))
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an event
            JOBS[job_id]["status"] = "error"
            q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


def _settle(job_id: str, rep: dict) -> None:
    job = JOBS[job_id]
    if rep.get("status") == "awaiting_approval":
        job.update(status="awaiting_approval", run_id=rep["run_id"], approval=rep["approval"])
        return  # the SSE stream stays open; the approval_required event was already emitted
    job.update(status="done", run_id=rep["run_id"])
    job["queue"].put({"type": "done", "run_id": rep["run_id"]})


class ApprovalIn(BaseModel):
    action: str = Field(pattern="^(approve|skip|modify)$")
    query: Optional[str] = Field(None, max_length=80)
    extra_tokens: int = Field(0, ge=0, le=1_000_000)


@app.post("/api/research/{job_id}/approve")
def approve(job_id: str, body: ApprovalIn):
    from agent_runtime.graph import resume_research

    job = JOBS.get(job_id)
    if not job or job["status"] != "awaiting_approval":
        raise HTTPException(409, "job is not waiting for approval")
    job["status"] = "running"

    def work():
        try:
            _settle(job_id, resume_research(job["run_id"], body.model_dump()))
        except Exception as exc:  # noqa: BLE001
            job["status"] = "error"
            job["queue"].put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


@app.get("/api/runs/{run_id}/checkpoints")
def checkpoints(run_id: str):
    from agent_runtime.graph import checkpoint_history

    return checkpoint_history(run_id)


@app.get("/api/runs/{run_id}/export.pdf")
def export_pdf(run_id: str):
    from fastapi.responses import Response

    from agent_runtime.pdf_export import markdown_to_pdf

    return Response(markdown_to_pdf(_run(run_id).report["markdown"]), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{run_id}.pdf"'})


@app.get("/api/research/{job_id}/events")
def events(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")

    def gen():
        while True:
            try:
                ev = job["queue"].get(timeout=30)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
            if ev.get("type") in ("done", "error"):
                break
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/runs")
def runs(limit: int = 30):
    with Session() as s:
        rows = s.scalars(select(ResearchRunRow).order_by(ResearchRunRow.started_at.desc()).limit(limit))
        return [{"run_id": r.run_id, "ticker": r.ticker, "as_of": r.as_of, "architecture": r.architecture,
                 "latency_ms": r.latency_ms, "tokens": r.total_tokens, "question": r.request.get("question")} for r in rows]


def _run(run_id: str) -> ResearchRunRow:
    with Session() as s:
        r = s.get(ResearchRunRow, run_id)
        if r is None:
            raise HTTPException(404, "run not found (only persisted runs are stored)")
        return r


@app.get("/api/runs/{run_id}")
def run_report(run_id: str):
    return _run(run_id).report


@app.get("/api/runs/{run_id}/export.md", response_class=PlainTextResponse)
def export_md(run_id: str):
    return _run(run_id).report["markdown"]


@app.get("/api/runs/{run_id}/lineage/{entity_id}")
def lineage(run_id: str, entity_id: str):
    rep = _run(run_id).report
    g = LineageRecorder(run_id=run_id).upstream(entity_id)
    ev = rep["evidence"].get(entity_id)
    return {"lineage": g, "evidence": ev, "as_of": rep["as_of"],
            "pit_ok": ev is None or ev["available_at"] <= rep["as_of"]}


@app.get("/api/chunks/{chunk_id:path}")
def chunk(chunk_id: str):
    with Session() as s:
        c = s.get(DocumentChunkRow, chunk_id)
        if not c:
            raise HTTPException(404, "chunk not found")
        return {"chunk_id": c.chunk_id, "section_path": c.section_path, "text": c.text, "page": f"p{c.page_start}-{c.page_end}",
                "available_at": c.available_at, "document_id": c.document_id}


@app.get("/api/evaluation/summary")
def eval_summary():
    p = EVAL / "summary_latest.json"
    if not p.exists():
        raise HTTPException(404, "run scripts/run_experiments.py first")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/evaluation/results")
def eval_results(architecture: Optional[str] = None, failed_only: bool = False):
    p = EVAL / "latest.json"
    if not p.exists():
        raise HTTPException(404, "run scripts/run_experiments.py first")
    res = json.loads(p.read_text(encoding="utf-8"))["results"]
    res = [r for r in res if (architecture is None or r["architecture"] == architecture) and (not failed_only or not r["passed"])]
    return [{k: r.get(k) for k in ("task_id", "architecture", "run_id", "category", "passed", "error_category")} |
            {"scores": {k: v.get("score") for k, v in r["scores"].items() if isinstance(v, dict) and "score" in v}}
            for r in res]


@app.get("/api/data-health")
def data_health():
    provider = GoldenFileProvider(ROOT / "benchmarks" / "golden_snapshot")
    facts, reliable = provider.facts()
    issues = quality.run_all(facts, reliable)
    with Session() as s:
        per = s.execute(select(FinancialFactRow.ticker, func.count(), func.max(FinancialFactRow.period_end),
                               func.sum(cast(FinancialFactRow.is_restated, Integer)), func.sum(cast(FinancialFactRow.pit_unsafe, Integer)))
                        .group_by(FinancialFactRow.ticker)).all()
        docs = dict(s.execute(select(SourceDocumentRow.ticker, func.count()).group_by(SourceDocumentRow.ticker)).all())
    counts: dict[str, int] = {}
    for i in issues:
        counts[i.rule] = counts.get(i.rule, 0) + 1
    rules = ["uniqueness", "non_negative", "balance_identity", "unit_jump", "pit_reliability"]
    return {"coverage": [{"ticker": t, "facts": n, "latest_period_end": str(pe), "restated_facts": int(r or 0),
                          "pit_unsafe_facts": int(u or 0), "documents": docs.get(t, 0)} for t, n, pe, r, u in per],
            "quality": {"facts_checked": len(facts), "issues": counts, "rules": rules,
                        "pass_rate": 1 - len({i.entity_id for i in issues if i.severity == 'ERROR'}) / max(len(facts), 1),
                        "samples": [i.model_dump() for i in issues[:20]]},
            "snapshots": list_snapshots()[:20]}


# ------------------------------------------------------------------ scenario D: factor validation
class FactorIn(BaseModel):
    factor: str
    universe: str = Field("all", pattern="^(all|new_energy|consumer)$")
    as_of_date: str = Field("2025-12-31", pattern=r"^\d{4}-\d{2}-\d{2}$")
    lookback_quarters: int = Field(8, ge=1, le=12)
    forward_horizon_days: int = Field(60, ge=5, le=250)
    quantiles: int = Field(3, ge=2, le=5)
    report_period: Optional[str] = Field(None, pattern=r"^\d{4}Q[1-4]$")


@app.get("/api/factors")
def factor_list():
    from data_platform.factors import FACTOR_CN, FACTORS, get_panel

    panel = get_panel()
    return {"factors": [{"id": f, "name": FACTOR_CN[f]} for f in FACTORS], "universes": ["all", "new_energy", "consumer"],
            "panel_last_date": panel.dates[-1], "panel_companies": len(panel.tickers("all"))}


@app.post("/api/factors/test")
def factor_test(body: FactorIn):
    """Runs the same tool the agent uses. A request that reads data not yet public at the cross-section is refused (422)."""
    from data_platform.factors import PanelCoverageError, PITViolationError, evaluate_factor_hypothesis

    try:
        rep = evaluate_factor_hypothesis(body.factor, body.universe, body.as_of_date, body.lookback_quarters,
                                         body.forward_horizon_days, body.quantiles, report_period=body.report_period)
    except PITViolationError as exc:
        raise HTTPException(422, {"error": "PIT_VIOLATION", "message": str(exc)})
    except (PanelCoverageError, ValueError) as exc:
        raise HTTPException(422, {"error": type(exc).__name__, "message": str(exc)})
    return rep.model_dump()


@app.get("/api/otel/spans")
def otel_spans(run_id: Optional[str] = None):
    """OpenTelemetry spans kept in memory (TRACING_BACKEND=otel without an OTLP endpoint). Empty otherwise."""
    from evaluation_platform.tracing import finished_otel_spans, tracing_backend

    if tracing_backend() != "otel":
        return {"enabled": False, "spans": []}
    out = []
    for sp in finished_otel_spans():
        a = dict(sp.attributes or {})
        if run_id and a.get("run_id") not in (None, run_id):
            continue
        out.append({"name": sp.name, "start_ns": sp.start_time, "end_ns": sp.end_time, "attributes": a,
                    "parent": sp.parent.span_id if sp.parent else None, "span_id": sp.context.span_id})
    return {"enabled": True, "spans": out}


REACT = ROOT / "apps" / "web_react" / "dist"


@app.get("/")
def index():
    """The React build (apps/web_react: npm run build). Without a build, say how to make one."""
    if (REACT / "index.html").exists():
        return FileResponse(REACT / "index.html")
    return HTMLResponse("<h3>FinResearch Agent API is running.</h3><p>The web UI is not built: "
                        "<code>cd apps/web_react &amp;&amp; npm install &amp;&amp; npm run build</code>, then reload. "
                        "API docs: <a href='/docs'>/docs</a>.</p>")


if (REACT / "assets").exists():
    app.mount("/assets", StaticFiles(directory=REACT / "assets"), name="assets")
