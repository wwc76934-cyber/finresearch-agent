"""Agent tool layer over data_platform + rag_engine. Every call is traced; every
returned number is a MetricRecord produced by deterministic code."""
from __future__ import annotations

import time
from datetime import date
from typing import Optional

from data_platform.calculators import AnomalyFlag
from data_platform.schema import MetricRecord
from data_platform.serving import label_to_end, period_label
from rag_engine.agentic import agentic_search

from .context import RunContext


class ToolError(RuntimeError):
    pass


def _call(ctx: RunContext, name: str, args: dict, fn):
    fault = ctx.fault_state.get("tool_error")
    if fault and fault.get("tool") == name and fault.get("remaining", 0) > 0:
        fault["remaining"] -= 1
        ctx.tracer.tool(name, args, error="ToolError: injected upstream timeout")
        raise ToolError(f"{name}: injected upstream timeout")
    t0 = time.perf_counter()
    out = fn()
    summary = (f"{len(out)} items" if isinstance(out, (list, dict)) else str(out)[:120]) if out is not None else "None"
    ctx.tracer.tool(name, args, result=summary, latency_ms=(time.perf_counter() - t0) * 1000)
    return out


def call_with_recovery(ctx: RunContext, name: str, args: dict, fn, retries: int = 2):
    """Self-healing wrapper: a failing tool is retried before the subgraph gives up."""
    last = None
    for _ in range(retries + 1):
        try:
            return _call(ctx, name, args, fn)
        except ToolError as exc:
            last = exc
    raise last  # type: ignore[misc]


def resolve_security(ctx: RunContext, query: str):
    return call_with_recovery(ctx, "resolve_security", {"query": query}, lambda: ctx.svc.resolve_security(query))


def check_data_coverage(ctx: RunContext) -> dict:
    return call_with_recovery(ctx, "check_data_coverage", {"ticker": ctx.ticker}, lambda: ctx.svc.coverage(ctx.ticker))


def latest_period(ctx: RunContext) -> Optional[date]:
    return ctx.svc.latest_period(ctx.ticker)


def get_financial_statements(ctx: RunContext, period: date, scope: str = "consolidated") -> dict[str, MetricRecord]:
    def run():
        out = {}
        for m in ("operating_revenue", "operating_cost", "net_profit", "operating_cash_flow", "accounts_receivable",
                  "inventory", "total_assets", "total_liabilities", "total_equity", "interest_bearing_debt",
                  "asset_impairment_loss", "rd_expense", "goodwill"):
            r = ctx.calc.raw(ctx.ticker, m, period, scope)
            if r:
                out[m] = r
        return out
    return call_with_recovery(ctx, "get_financial_statements",
                              {"ticker": ctx.ticker, "period": period_label(period), "scope": scope}, run)


def get_fact(ctx: RunContext, metric: str, period: date, scope: str = "consolidated") -> Optional[MetricRecord]:
    return call_with_recovery(ctx, "get_financial_statements",
                              {"ticker": ctx.ticker, "metric": metric, "period": period_label(period), "scope": scope},
                              lambda: ctx.calc.raw(ctx.ticker, metric, period, scope))


def calculate_financial_ratios(ctx: RunContext, period: date) -> dict[str, MetricRecord]:
    return call_with_recovery(ctx, "calculate_financial_ratios", {"ticker": ctx.ticker, "period": period_label(period)},
                              lambda: ctx.calc.ratios(ctx.ticker, period))


def calculate_growth_metrics(ctx: RunContext, period: date) -> dict[str, MetricRecord]:
    return call_with_recovery(ctx, "calculate_growth_metrics", {"ticker": ctx.ticker, "period": period_label(period)},
                              lambda: ctx.calc.growth(ctx.ticker, period))


def single_quarter(ctx: RunContext, metric: str, period: date) -> MetricRecord:
    return call_with_recovery(ctx, "calculate_growth_metrics",
                              {"ticker": ctx.ticker, "single_quarter": metric, "period": period_label(period)},
                              lambda: ctx.calc.single_quarter(ctx.ticker, metric, period))


def detect_financial_anomalies(ctx: RunContext, period: date) -> list[AnomalyFlag]:
    anns = [(d.document_id, d.title) for d in ctx.svc.documents(ctx.ticker, "announcement")]
    return call_with_recovery(ctx, "detect_financial_anomalies", {"ticker": ctx.ticker, "period": period_label(period)},
                              lambda: ctx.calc.detect_anomalies(ctx.ticker, period, anns))


def calculate_valuation_metrics(ctx: RunContext) -> dict[str, MetricRecord]:
    return call_with_recovery(ctx, "calculate_valuation_metrics", {"ticker": ctx.ticker},
                              lambda: ctx.calc.valuation(ctx.ticker))


def get_industry_peers(ctx: RunContext) -> MetricRecord:
    return call_with_recovery(ctx, "get_industry_peers", {"ticker": ctx.ticker},
                              lambda: ctx.calc.peer_valuation(ctx.ticker))


def calculate_risk_metrics(ctx: RunContext) -> dict[str, MetricRecord]:
    return call_with_recovery(ctx, "calculate_risk_metrics", {"ticker": ctx.ticker}, lambda: ctx.calc.risk(ctx.ticker))


def get_price_history(ctx: RunContext, days: int = 250) -> list:
    return call_with_recovery(ctx, "get_price_history", {"ticker": ctx.ticker, "days": days},
                              lambda: ctx.svc.prices(ctx.ticker)[-days:])


def get_company_announcements(ctx: RunContext) -> list:
    return call_with_recovery(ctx, "get_company_announcements", {"ticker": ctx.ticker},
                              lambda: ctx.svc.documents(ctx.ticker, "announcement"))


def retrieve_document_evidence(ctx: RunContext, query: str, section_types: Optional[list[str]] = None,
                               top_k: int = 4, doc_ids: Optional[list[str]] = None):
    cfg = ctx.config

    def run():
        if doc_ids:
            return ctx.retriever.search(query, ticker=ctx.ticker, as_of=ctx.as_of, top_k=top_k, doc_ids=doc_ids,
                                        section_types=section_types if cfg.rag_anchors_prefilter else None,
                                        mode="hybrid" if cfg.hybrid_rerank else "bm25", rerank=cfg.hybrid_rerank)
        res = agentic_search(ctx.retriever, query, ctx.ticker, ctx.as_of, top_k=top_k,
                             section_types=section_types if cfg.rag_anchors_prefilter else None,
                             mode="hybrid" if cfg.hybrid_rerank else "bm25", self_rag=cfg.self_rag)
        ctx.tracer.tool("retrieval_grader", {"query": query}, result=f"{res.grade} after {len(res.attempts)} attempt(s)")
        return res.hits
    return call_with_recovery(ctx, "retrieve_document_evidence", {"query": query, "sections": section_types}, run)


def latest_report_doc_ids(ctx: RunContext) -> list[str]:
    pe = latest_period(ctx)
    return [d.document_id for d in ctx.svc.documents(ctx.ticker) if d.period_end == pe and d.doc_type != "announcement"]


def retrieve_latest_first(ctx: RunContext, query: str, section_types: Optional[list[str]] = None, top_k: int = 3):
    """Prefer the latest visible periodic report; fall back to the whole PIT-filtered corpus."""
    ids = latest_report_doc_ids(ctx)
    hits = retrieve_document_evidence(ctx, query, section_types, top_k=top_k, doc_ids=ids) if ids else []
    return hits or retrieve_document_evidence(ctx, query, section_types, top_k=top_k)


def build_event_timeline(ctx: RunContext, docs: list) -> list[dict]:
    return call_with_recovery(ctx, "build_event_timeline", {"n": len(docs)},
                              lambda: [{"document_id": d.document_id, "title": d.title,
                                        "published_at": d.published_at.isoformat(),
                                        "available_at": d.available_at.isoformat()} for d in docs])


def query_evidence_store(ctx: RunContext, keyword: str) -> list:
    return call_with_recovery(ctx, "query_evidence_store", {"keyword": keyword},
                              lambda: [e for e in ctx.evidence.values() if keyword in e.summary or keyword in e.excerpt])


def validate_point_in_time(ctx: RunContext, evidence_ids: list[str]) -> list[str]:
    return [e for e in evidence_ids if e in ctx.evidence and ctx.evidence[e].available_at > ctx.as_of]


def trace_data_lineage(ctx: RunContext, entity_id: str) -> dict:
    return ctx.lineage.upstream(entity_id)


def period_from_label(label: str) -> date:
    return label_to_end(label)
