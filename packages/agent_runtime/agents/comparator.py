"""PeerComparisonAggregator (v2.0, scenario C): a cross-company matrix from deterministic metrics only.

All companies are read at the same as_of and compared on the latest report period that every one of them has
disclosed. Nothing here calls a model: each number comes from a MetricRecord and is cited, so Tier-1 verifies the
comparison sentences like any other claim. Items that are not comparable (different statement template, missing data,
negative-earnings PE) are named, not silently filled.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Callable, Optional

from data_platform.serving import period_label
from data_platform.schema import MetricRecord

from ..context import RunContext, fmt

MAX_TICKERS = 4
# dimension -> [(calculator family, metric key, display label, higher is better)]
DIMENSIONS: list[tuple[str, list[tuple[str, str, str, bool]]]] = [
    ("成长性", [("growth", "revenue_yoy", "营业收入同比", True), ("growth", "net_profit_yoy", "归母净利润同比", True)]),
    ("盈利能力", [("ratios", "gross_margin", "毛利率", True), ("ratios", "net_margin", "净利率", True),
                  ("ratios", "roe_annualized", "年化ROE", True)]),
    ("盈利质量", [("ratios", "ocf_to_net_profit", "经营现金流/净利润", True)]),
    ("财务结构", [("ratios", "debt_to_assets", "资产负债率", False)]),
    ("估值", [("valuation", "pe_ttm", "PE-TTM", False), ("valuation", "pb_lf", "PB-LF", False),
              ("valuation", "pe_percentile_3y", "PE近3年分位", False)]),
]
# margins and working-capital ratios assume an industrial income statement
NON_COMPARABLE_FOR_FINANCIALS = {"gross_margin"}
FINANCIAL = {"bank", "insurance", "broker"}     # manufacturing and consumer companies share the industrial income statement


def common_period(ctx: RunContext, tickers: list[str]) -> Optional[date]:
    ends = [ctx.svc.latest_period(t) for t in tickers]
    return None if any(e is None for e in ends) else min(e for e in ends if e is not None)


def _records(ctx: RunContext, ticker: str, pe: date) -> dict[str, dict[str, MetricRecord]]:
    return {"ratios": ctx.calc.ratios(ticker, pe), "growth": ctx.calc.growth(ticker, pe), "valuation": ctx.calc.valuation(ticker)}


def _name(ctx: RunContext, ticker: str) -> str:
    sec = ctx.svc.resolve_security(ticker)
    return sec.name if sec else ticker


def build(ctx: RunContext, tickers: list[str]) -> dict[str, Any]:
    """Create the comparison claims in `ctx` and return the matrix for the report. Safe to call once per run."""
    from .. import tools  # noqa: F401  (kept for symmetry with other agents; tool calls are traced below)

    tickers = tickers[:MAX_TICKERS]
    pe = common_period(ctx, tickers)
    out: dict[str, Any] = {"tickers": [], "period": None, "rows": [], "warnings": []}
    if pe is None:
        missing = [t for t in tickers if ctx.svc.latest_period(t) is None]
        ctx.open_questions.append({"topic": f"同业对比无法进行：{', '.join(missing)} 在该时点没有已披露的财务数据",
                                   "reason": "先让数据层拉取该公司（hybrid_live 模式）后再比较"})
        out["warnings"].append(f"no disclosed statements for {missing}")
        return out
    label = period_label(pe)
    out["period"] = label
    templates: dict[str, str] = {}
    for t in tickers:
        sec = ctx.svc.resolve_security(t)
        templates[t] = sec.template if sec else "manufacturing"
    out["tickers"] = [{"ticker": t, "name": _name(ctx, t), "template": templates[t]} for t in tickers]
    recs = {t: _records(ctx, t, pe) for t in tickers}
    ctx.tracer.tool("build_peer_matrix", {"tickers": tickers, "period": label}, result=f"{len(DIMENSIONS)} dimensions")

    mixed = len({tpl in FINANCIAL for tpl in templates.values()}) > 1
    if mixed:
        kinds = "、".join(f"{_name(ctx, t)}（{templates[t]}）" for t in tickers)
        msg = f"报表口径不同：{kinds}。毛利率、存货与应收类指标不可直接比较，对应项标注为不适用。"
        out["warnings"].append(msg)
        ctx.open_questions.append({"topic": "同业对比口径提示", "reason": msg})
    if any(ctx.svc.latest_period(t) != pe for t in tickers):
        msg = f"各公司最新披露期不同，已统一按共同的 {label} 比较。"
        out["warnings"].append(msg)
        ctx.open_questions.append({"topic": "同业对比报告期对齐", "reason": msg})

    for dim, metrics in DIMENSIONS:
        used: list[MetricRecord] = []
        parts_fn: list[Callable[[], str]] = []
        for t in tickers:
            def part(t: str = t, metrics: list[tuple[str, str, str, bool]] = metrics) -> str:
                cells = []
                for fam, key, lab, _ in metrics:
                    r = recs[t][fam].get(key)
                    na = mixed and templates[t] in FINANCIAL and key in NON_COMPARABLE_FOR_FINANCIALS
                    cells.append(f"{lab} {'不适用（口径不同）' if na else (fmt(r) if r is not None else '数据不足')}")
                return f"{_name(ctx, t)}：" + "，".join(cells)
            parts_fn.append(part)
            used += [r for fam, key, _, _ in metrics if (r := recs[t][fam].get(key)) is not None]
        if not used:
            continue

        def render(dim: str = dim, parts_fn: list[Callable[[], str]] = parts_fn) -> str:
            return f"{dim}（{label}）：" + "；".join(p() for p in parts_fn) + "。"
        evs = [ctx.evidence_from_metric(r, "comparator") for r in used]
        cl = ctx.add_claim("comparison", "derived_metric", render(), evs, "comparator")
        ctx.renderers[cl.claim_id] = render
        for fam, key, lab, _ in metrics:
            vals = {t: recs[t][fam].get(key) for t in tickers}
            out["rows"].append({"dimension": dim, "metric": lab, "values": {
                t: ("不适用" if (mixed and templates[t] in FINANCIAL and key in NON_COMPARABLE_FOR_FINANCIALS)
                    else (fmt(r) if r is not None else "数据不足")) for t, r in vals.items()}})
        # a lead sentence for the first metric of the dimension: names only, no number to mis-state
        fam, key, lab, high = metrics[0]
        ranked = [(t, r.value) for t in tickers if (r := recs[t][fam].get(key)) is not None and r.value is not None
                  and not (mixed and templates[t] in FINANCIAL and key in NON_COMPARABLE_FOR_FINANCIALS)]
        if len(ranked) >= 2:
            ranked.sort(key=lambda x: -x[1] if high else x[1])
            lead = ranked[0][0]
            word = "最高" if high else "最低"
            cl2 = ctx.add_claim("comparison", "derived_metric", f"{dim}：{lab}{word}的是{_name(ctx, lead)}。",
                                [ctx.evidence_from_metric(r, "comparator") for r in used if r.ticker == lead and r.metric == key][:1]
                                or evs[:1], "comparator")
            ctx.renderers[cl2.claim_id] = (lambda d=dim, l=lab, w=word, n=_name(ctx, lead): f"{d}：{l}{w}的是{n}。")
    return out
