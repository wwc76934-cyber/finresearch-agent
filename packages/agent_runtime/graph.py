"""Main LangGraph state machine (Supervisor + parallel subgraphs + re-planning +
debate + risk + two-tier verifier with partial rollback) and the run entry point."""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from datetime import datetime
from typing import Callable, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from data_platform.db import (ROOT, EvidenceRecordRow, ResearchClaimRow, ResearchRunRow, VerificationResultRow,
                              get_session_factory)
from data_platform.serving import PITDataService
from data_platform.versioning import create_snapshot

from .agents import debate as debate_mod
from .agents import faults, planner, report as report_mod, risk as risk_mod, verifier
from .agents.collectors import run_subgraph
from .context import RunContext, drop_context, get_context
from .llm import BaseLLM, estimate_tokens, run_cost_cny
from .state import ARCHITECTURES, ArchConfig, GraphState, ResearchRequest, SubTask


def _downstream_context(ctx: RunContext, state: GraphState) -> str:
    """Pointer-based: summaries + only the evidence they reference. Isolation off: everything."""
    if not ctx.config.context_isolation:
        return "\n".join(ctx.raw_outputs + state.get("shared_history", []) +
                         [f"{e.evidence_id}: {e.summary}\n{e.excerpt}" for e in ctx.evidence.values()])
    parts = []
    for s in state.get("summaries", []):
        parts.append(f"[{s['agent']}] {s['summary']} evidence={s['evidence_ids']}")
    return "\n".join(parts)


def _account(ctx: RunContext, node: str, state: GraphState) -> None:
    from .llm import LLMResult

    payload = _downstream_context(ctx, state)
    ctx.tracer.llm(node, LLMResult(None, estimate_tokens(payload), 0, 0.0, "context"), "strong")


def node_plan(state: GraphState) -> dict:
    ctx = get_context(state["run_id"])
    with ctx.tracer.node("planner") as span:
        p = planner.plan(ctx)
        ctx.plan = p
        span["summary"] = f"{len(p.subtasks)} subtasks, {len(p.asks)} asks ({p.rationale})"
    return {"plan": p.model_dump(), "replan_depth": 0}


def fan_out(state: GraphState):
    return [Send("collect", {"run_id": state["run_id"], "task": t}) for t in state["plan"]["subtasks"]]


def node_collect(payload: dict) -> dict:
    ctx = get_context(payload["run_id"])
    summ = run_subgraph(payload["run_id"], payload["task"])
    out = {"summaries": [summ.model_dump()]}
    if not ctx.config.context_isolation:
        out["shared_history"] = [n for n in ctx.raw_outputs]
    return out


def node_gather(state: GraphState) -> dict:
    ctx = get_context(state["run_id"])
    with ctx.tracer.node("gather") as span:
        span["summary"] = f"{len(state.get('summaries', []))} subtask summaries; anomalies={list(ctx.anomalies)}"
    return {}


def route_after_collection(state: GraphState) -> str:
    ctx = get_context(state["run_id"])
    cfg = ctx.config
    handled = set(state.get("handled_anomalies", []))
    high = [a for a in ctx.anomalies.values() if a["severity"] == "HIGH" and a["code"] not in handled]
    used = _used_tokens(ctx)
    if not (cfg.replanning and high and state.get("replan_depth", 0) < cfg.max_replan_depth):
        return "debate"
    over_budget = used >= ctx.request.token_budget
    if ctx.request.hitl and (over_budget or ctx.request.hitl_always):
        return "approval"   # Human-in-the-Loop: pause instead of silently skipping or overspending
    return "debate" if over_budget else "replan"


def _used_tokens(ctx: RunContext) -> int:
    return sum(s["input_tokens"] + s["output_tokens"] for s in ctx.tracer.spans)


def node_approval(state: GraphState) -> dict:
    """Pause the graph (LangGraph interrupt) and wait for a human decision.

    Resume value: {"action": "approve" | "skip" | "modify", "query": "...", "extra_tokens": int}
    """
    ctx = get_context(state["run_id"])
    handled = set(state.get("handled_anomalies", []))
    high = [a for a in ctx.anomalies.values() if a["severity"] == "HIGH" and a["code"] not in handled][:2]
    used = _used_tokens(ctx)
    payload = {"kind": "replan_approval", "reason": "over_budget" if used >= ctx.request.token_budget else "policy",
               "used_tokens": used, "token_budget": ctx.request.token_budget,
               "proposed": [{"code": a["code"], "message": a["message"], "query": a["suggested_query"]} for a in high]}
    ctx.tracer.tool("human_approval_request", {"proposed": [p["code"] for p in payload["proposed"]]})
    decision = interrupt(payload) or {}
    action = decision.get("action", "skip")
    with ctx.tracer.node("approval") as span:
        span["summary"] = f"human decision: {action}"
        if action == "modify" and decision.get("query"):
            ctx.query_override = str(decision["query"])[:80]
        if decision.get("extra_tokens"):
            ctx.request.token_budget += int(decision["extra_tokens"])
        if action == "skip":
            ctx.open_questions.append({"topic": f"异常追查被人工跳过：{[p['code'] for p in payload['proposed']]}",
                                       "reason": "human-in-the-loop 决定不追查"})
    return {"approval": action, "handled_anomalies": [p["code"] for p in payload["proposed"]] if action == "skip" else []}


def route_after_approval(state: GraphState) -> str:
    return "debate" if state.get("approval") == "skip" else "replan"


def node_replan(state: GraphState) -> dict:
    ctx = get_context(state["run_id"])
    handled = set(state.get("handled_anomalies", []))
    with ctx.tracer.node("replan") as span:
        high = [a for a in ctx.anomalies.values() if a["severity"] == "HIGH" and a["code"] not in handled][:2]
        override = getattr(ctx, "query_override", None)
        tasks = [SubTask(task_id=f"R{state.get('replan_depth', 0) + 1}-{i + 1}", agent="investigate",
                         objective=f"追查异常 {a['code']}", tool_budget=3, query=override or a["suggested_query"],
                         section_types=[a.get("section_type", "NOTES")], anomaly_code=a["code"]).model_dump()
                 for i, a in enumerate(high)]
        span["summary"] = f"derived {len(tasks)} investigation task(s): {[a['code'] for a in high]}"
        span["anomaly_flags"] = [a["code"] for a in high]
    return {"pending_tasks": tasks, "replan_depth": state.get("replan_depth", 0) + 1,
            "handled_anomalies": [a["code"] for a in high]}


def fan_out_replan(state: GraphState):
    return [Send("collect", {"run_id": state["run_id"], "task": t}) for t in state.get("pending_tasks", [])]


def node_debate(state: GraphState) -> dict:
    ctx = get_context(state["run_id"])
    if not ctx.config.debate:
        return {"debate": {}}
    with ctx.tracer.node("debate") as span:
        _account(ctx, "debate", state)
        d = debate_mod.run_debate(ctx)
        span["summary"] = f"{len(d['rounds'])} round(s), {len(d['falsification'])} falsification criteria"
    return {"debate": d}


def node_risk(state: GraphState) -> dict:
    ctx = get_context(state["run_id"])
    if not ctx.config.risk_agent:
        return {"risk": {}}
    with ctx.tracer.node("risk") as span:
        _account(ctx, "risk", state)
        r = risk_mod.evaluate_risk(ctx)
        span["summary"] = "; ".join(f"{m['dimension']}={m['score']}" for m in r["matrix"])
    return {"risk": r}


def node_inject(state: GraphState) -> dict:
    ctx = get_context(state["run_id"])
    if ctx.fault_state:
        with ctx.tracer.node("fault_injection") as span:
            span["summary"] = ", ".join(faults.inject(ctx)) or "none"
    return {}


def node_verify(state: GraphState) -> dict:
    ctx = get_context(state["run_id"])
    if not (ctx.config.tier1_verifier or ctx.config.tier2_verifier):
        ctx.pending_failures = []
        return {}
    with ctx.tracer.node("verifier") as span:
        _account(ctx, "verifier", state)
        t0 = time.perf_counter()
        ctx.pending_failures = verifier.verify(ctx)
        ctx.tracer.tool("run_deterministic_validators", {"claims": len(ctx.claims)},
                        result=f"{len(ctx.pending_failures)} failure(s) in {(time.perf_counter() - t0) * 1000:.1f}ms")
        span["summary"] = (f"failures={[(f.claim_id, f.error_type) for f in ctx.pending_failures]} "
                           f"downgraded={len(ctx.uncertainties)}")
    return {"verification_round": state.get("verification_round", 0) + 1}


def route_after_verify(state: GraphState) -> str:
    ctx = get_context(state["run_id"])
    return "rewrite" if getattr(ctx, "pending_failures", []) else "report"


def node_rewrite(state: GraphState) -> dict:
    ctx = get_context(state["run_id"])
    with ctx.tracer.node("rollback_rewrite") as span:
        n = verifier.rewrite(ctx, ctx.pending_failures)
        span["summary"] = f"rewrote {n} claim(s) (partial rollback)"
    return {}


def node_report(state: GraphState) -> dict:
    ctx = get_context(state["run_id"])
    with ctx.tracer.node("report") as span:
        _account(ctx, "report", state)
        rep = report_mod.build_report(ctx, state.get("debate", {}), state.get("risk", {}), ctx.config.name)
        span["summary"] = f"{sum(len(s['claim_ids']) for s in rep['sections'])} claims in report"
    return {"report": rep}


def build_graph(checkpointer=None):
    g = StateGraph(GraphState)
    for name, fn in [("plan", node_plan), ("collect", node_collect), ("gather", node_gather), ("replan", node_replan),
                     ("debate", node_debate), ("risk", node_risk), ("inject", node_inject), ("verify", node_verify),
                     ("rewrite", node_rewrite), ("report", node_report), ("approval", node_approval)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["collect"])
    g.add_edge("collect", "gather")
    g.add_conditional_edges("gather", route_after_collection,
                            {"replan": "replan", "debate": "debate", "approval": "approval"})
    g.add_conditional_edges("approval", route_after_approval, {"replan": "replan", "debate": "debate"})
    g.add_conditional_edges("replan", fan_out_replan, ["collect"])
    g.add_edge("debate", "risk")
    g.add_edge("risk", "inject")
    g.add_edge("inject", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"rewrite": "rewrite", "report": "report"})
    g.add_edge("rewrite", "verify")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


_GRAPH = None


def make_checkpointer():
    """Persistent checkpoints (SQLite file data/checkpoints.db) unless CHECKPOINTER=memory."""
    if os.environ.get("CHECKPOINTER", "sqlite") == "memory":
        return MemorySaver()
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:  # optional dependency
        return MemorySaver()
    path = ROOT / "data" / "checkpoints.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph(make_checkpointer())
    return _GRAPH


def checkpoint_history(run_id: str) -> list[dict]:
    """The persisted step-by-step state history of one run (newest first)."""
    out = []
    for snap in get_graph().get_state_history({"configurable": {"thread_id": run_id}}):
        out.append({"step": snap.metadata.get("step"), "next": list(snap.next),
                    "checkpoint_id": snap.config["configurable"].get("checkpoint_id"),
                    "summaries": len(snap.values.get("summaries", [])), "replan_depth": snap.values.get("replan_depth")})
    return out


def prepare(request: ResearchRequest, config: ArchConfig, db_url: Optional[str] = None,
            llm: Optional[BaseLLM] = None, listener: Optional[Callable] = None) -> RunContext:
    probe = PITDataService(request.as_of, "probe", db_url)
    sec = probe.resolve_security(request.ticker)
    if sec is None:
        raise ValueError(f"unknown security {request.ticker}")
    request = request.model_copy(update={"ticker": sec.ticker})
    tickers = [sec.ticker] + [p.ticker for p in probe.industry_peers(sec.ticker)]
    snap = create_snapshot(request.as_of, tickers, db_url)
    run_id = f"run_{datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:6]}"
    return RunContext(run_id, request, config, snap.snapshot_id, db_url, llm, listener)


def finalize(ctx: RunContext, rep: dict, started: datetime, latency_ms: float, persist: bool = True) -> dict:
    totals = ctx.tracer.totals()
    rep["trace"] = ctx.tracer.as_dicts()
    cost = run_cost_cny(rep["trace"])
    llm_calls = [c for s in rep["trace"] for c in s.get("llm_calls", []) if c.get("model") != "context"]
    totals["llm_calls"] = len(llm_calls)
    totals["llm_calls_failed"] = sum(1 for c in llm_calls if not c.get("online"))
    totals["llm_tokens"] = sum(c["in"] + c["out"] for c in llm_calls if c.get("online"))
    rep["totals"] = {**totals, "latency_ms": latency_ms, "cost_cny": cost, "llm_online": ctx.llm.online}
    if persist:
        ctx.lineage.flush()
        ctx.tracer.persist(ctx.db_url)
        with get_session_factory(ctx.db_url)() as s:
            s.add(ResearchRunRow(run_id=ctx.run_id, snapshot_id=ctx.snapshot_id, ticker=ctx.ticker, as_of=ctx.as_of,
                                 request=json.loads(ctx.request.model_dump_json()), architecture=ctx.config.name,
                                 status="done", started_at=started, finished_at=datetime.now(), latency_ms=latency_ms,
                                 total_tokens=totals["input_tokens"] + totals["output_tokens"], cost_cny=cost,
                                 report=json.loads(json.dumps(rep, default=str))))
            for c in ctx.claims.values():
                s.add(ResearchClaimRow(claim_id=f"{ctx.run_id}:{c.claim_id}", run_id=ctx.run_id, section=c.section,
                                       claim_type=c.claim_type, text=c.text, evidence_ids=c.evidence_ids,
                                       metric_ids=c.metric_ids, verification_status=c.status))
            for v in ctx.verification_log:
                s.add(VerificationResultRow(run_id=ctx.run_id, claim_id=v["claim_id"], tier=v.get("tier", 1),
                                            passed=v["passed"], error_type=v.get("error_type"),
                                            detail=v.get("fix_instruction", ""), attempt=v.get("attempt", 0),
                                            circuit_broken=v.get("circuit_broken", False)))
            for e in ctx.evidence.values():
                s.add(EvidenceRecordRow(evidence_id=f"{ctx.run_id}:{e.evidence_id}", run_id=ctx.run_id, kind=e.kind,
                                        source_ref=e.source_ref, summary=e.summary, excerpt=e.excerpt[:4000],
                                        available_at=e.available_at, payload=json.loads(e.model_dump_json())))
            s.commit()
    return rep


def run_research(request: ResearchRequest, architecture: str | ArchConfig = "B4", db_url: Optional[str] = None,
                 llm: Optional[BaseLLM] = None, listener: Optional[Callable] = None, persist: bool = True) -> dict:
    from .baselines import BASELINES

    name = architecture if isinstance(architecture, str) else architecture.name
    if isinstance(architecture, str) and architecture in BASELINES:
        return BASELINES[architecture](request, db_url=db_url, llm=llm, listener=listener, persist=persist)
    config = architecture if isinstance(architecture, ArchConfig) else ARCHITECTURES[architecture]
    ctx = prepare(request, config, db_url, llm, listener)
    ctx.started, ctx.t0, ctx.persist, ctx.arch_name = datetime.now(), time.perf_counter(), persist, name
    init = {"run_id": ctx.run_id, "request": json.loads(request.model_dump_json()), "snapshot_id": ctx.snapshot_id,
            "summaries": [], "handled_anomalies": [], "claim_ids": [], "shared_history": []}
    return _drive(ctx, init)


def _drive(ctx: RunContext, inp) -> dict:
    """Invoke or resume the graph. On a Human-in-the-Loop interrupt the RunContext is kept alive and a
    {"status": "awaiting_approval"} dict is returned; call resume_research(run_id, decision) to continue."""
    cfg = {"configurable": {"thread_id": ctx.run_id}, "recursion_limit": 60}
    done = False
    try:
        out = get_graph().invoke(inp, config=cfg)
        if "__interrupt__" in out:
            payload = out["__interrupt__"][0].value
            ctx.tracer._emit({"type": "approval_required", "run_id": ctx.run_id, **payload})
            return {"status": "awaiting_approval", "run_id": ctx.run_id, "approval": payload}
        rep = finalize(ctx, out["report"], ctx.started, (time.perf_counter() - ctx.t0) * 1000, ctx.persist)
        rep["architecture"] = ctx.arch_name
        rep["status"] = "done"
        done = True
        return rep
    finally:
        if done or not ctx.request.hitl:
            drop_context(ctx.run_id)


def resume_research(run_id: str, decision: dict) -> dict:
    """Continue a run paused for approval. decision: {"action": "approve"|"skip"|"modify", "query"?, "extra_tokens"?}"""
    ctx = get_context(run_id)  # KeyError if the process restarted: the in-memory stores are gone
    return _drive(ctx, Command(resume=decision))
