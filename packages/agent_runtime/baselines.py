"""Baselines B0-B2 for the evaluation platform (B3/B4 are graph configurations).

B0 Vanilla LLM: question only, no tools. Offline it can only decline.
B1 Naive RAG: fixed-size chunks, no PIT/metadata filter, numbers read from text.
B2 Single ReAct agent: one agent, shared context, fixed tool budget, no verifier.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Optional

from data_platform.schema import MetricRecord, StatementScope
from rag_engine.index import get_retriever

from . import tools
from .agents import faults, planner, report as report_mod
from .agents.collectors import answer_asks, answer_factor_asks
from .context import RunContext, drop_context, fmt
from .graph import finalize, prepare
from .llm import LLMResult, call_json, estimate_tokens
from .state import ArchConfig, ResearchRequest

OFF = dict(replanning=False, debate=False, risk_agent=False, tier1_verifier=False, tier2_verifier=False)


def _wrap(fn):
    def run(request: ResearchRequest, db_url=None, llm=None, listener=None, persist=True):
        name = fn.__name__.split("_")[0].upper()
        ctx = prepare(request, ArchConfig(name=name, **OFF), db_url, llm, listener)
        started, t0 = datetime.now(), time.perf_counter()
        try:
            debate = fn(ctx)
            faults.inject(ctx)
            with ctx.tracer.node("report"):
                rep = report_mod.build_report(ctx, debate, {}, name)
            rep = finalize(ctx, rep, started, (time.perf_counter() - t0) * 1000, persist)
            rep["architecture"] = name
            return rep
        finally:
            drop_context(ctx.run_id)
    return run


@_wrap
def b0_vanilla(ctx: RunContext) -> dict:
    with ctx.tracer.node("vanilla_llm"):
        sys_p = "你是金融分析师。回答用户的投研问题。输出JSON {\"answer\":\"...\",\"numbers\":{\"指标\":数值}}"
        data = call_json(ctx.llm, ctx.tracer, "vanilla", sys_p,
                         f"股票{ctx.ticker}，研究时点{ctx.as_of}。问题：{ctx.request.question}", "strong")
        asks = planner.rule_asks(ctx.request.question, None)
        if data is None:
            ctx.open_questions.append({"topic": "全部问题", "reason": "无外部数据且模型离线，无法作答"})
        else:
            # whatever the model says is recorded as an unverified inference with no evidence
            ctx.add_claim("answer", "model_inference", str(data.get("answer", ""))[:800], [], "vanilla")
            for a in asks:
                ctx.answers[a.ask_id] = {"ask": a.model_dump(), "metric_id": None, "value": None, "status": "LLM_ONLY"}
    return {}


NUM_NEAR = re.compile(r"\|\s*([^|]+?)\s*\|\s*([-\d,]+\.\d+)\s*\|")
METRIC_CN = {"operating_revenue": "营业收入", "net_profit": "归属于母公司股东的净利润", "operating_cost": "营业成本",
             "operating_cash_flow": "经营活动产生的现金流量净额", "accounts_receivable": "应收账款", "inventory": "存货"}


@_wrap
def b1_naive(ctx: RunContext) -> dict:
    with ctx.tracer.node("naive_rag") as span:
        retr = get_retriever(ctx.db_url, naive=True)
        hits = retr.search(ctx.request.question + " " + ctx.ticker, ticker=None, as_of=None, mode="bm25", top_k=5,
                           rerank=False, return_parents=False)
        ctx.tracer.tool("retrieve", {"query": ctx.request.question, "filters": "none"}, result=f"{len(hits)} chunks")
        evs = [ctx.evidence_from_hit(h, "naive") for h in hits]
        ctx.tracer.llm("naive", LLMResult(None, estimate_tokens("".join(e.excerpt for e in evs)), 0, 0, "context"), "strong")
        asks = planner.rule_asks(ctx.request.question, None)
        for a in asks:
            if a.kind == "unanswerable":
                continue  # naive pipelines do not know when to refuse
            if a.kind != "metric" or a.metric not in METRIC_CN:
                continue
            for e in evs:
                for name, num in NUM_NEAR.findall(e.excerpt):
                    if METRIC_CN[a.metric] in name or name in METRIC_CN[a.metric]:
                        v = float(num.replace(",", ""))  # unit unknown in a header-less chunk: read as 元
                        rec = MetricRecord(metric_id=f"m_naive_{a.ask_id}", ticker=ctx.ticker, metric=a.metric, value=v,
                                           unit="CNY", period=a.period or "?", statement_scope=StatementScope.CONSOLIDATED,
                                           formula="regex from chunk", input_ids=[e.source_ref], available_at=e.available_at,
                                           as_of=ctx.as_of)
                        ctx.calc.store[rec.metric_id] = rec
                        ctx.answers[a.ask_id] = {"ask": a.model_dump(), "metric_id": rec.metric_id, "value": v,
                                                 "status": "OK", "unit": "CNY", "display": fmt(rec)}
                        ctx.add_claim("answer", "disclosed_fact", f"{name}为{num}", [e], "naive")
                        break
                if a.ask_id in ctx.answers:
                    break
        for e in evs[:3]:
            ctx.add_claim("financials", "disclosed_fact", e.excerpt[:120].replace("\n", " "), [e], "naive")
        span["summary"] = f"{len(hits)} chunks, {len(ctx.answers)} answers"
    return {}


@_wrap
def b2_react(ctx: RunContext) -> dict:
    """One agent, one shared scratchpad, budget of 6 tool calls."""
    from .agents import collectors

    budget = 6
    with ctx.tracer.node("react_agent") as span:
        ctx.plan = planner.plan(ctx)
        steps = [("statements", lambda: collectors.f_fetch({"run_id": ctx.run_id, "task": {"agent": "react"}})),
                 ("metrics", lambda: collectors.f_compute({"run_id": ctx.run_id, "task": {"agent": "react"}})),
                 ("asks", lambda: {"claim_ids": answer_asks(ctx, ctx.plan.asks, "fundamental", ())
                                   + answer_factor_asks(ctx, ctx.plan.asks, "react")}),   # B2 holds every tool
                 ("market", lambda: collectors.m_valuation({"run_id": ctx.run_id, "task": {"agent": "react"}})),
                 ("anomalies", lambda: collectors.f_anomalies({"run_id": ctx.run_id, "task": {"agent": "react"}})),
                 ("events", lambda: collectors.n_events({"run_id": ctx.run_id, "task": {"agent": "react"}}))]
        done = []
        for name, step in steps:
            used = len(span["tool_calls"])
            if used >= budget * 2:  # each step issues ~2 tool calls; stop when the budget is spent
                break
            step()
            done.append(name)
            # shared scratchpad: every observation goes back into the next prompt
            ctx.tracer.llm("react", LLMResult(None, estimate_tokens("\n".join(ctx.raw_outputs)), 80, 0, "context"), "strong")
        span["summary"] = f"steps={done}"
    return {}


BASELINES = {"B0": b0_vanilla, "B1": b1_naive, "B2": b2_react}
