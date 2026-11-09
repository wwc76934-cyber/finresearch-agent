"""Independent Risk Agent: six-dimension rule engine over MetricRecords and events."""
from __future__ import annotations

from .. import tools
from ..context import RunContext, fmt

DIMENSIONS = ["财务质量", "估值", "偿债与流动性", "经营与行业", "治理与合规", "市场波动"]


def _band(v, cuts, scores=(1, 2, 3, 4, 5)):
    for c, s in zip(cuts, scores):
        if v <= c:
            return s
    return scores[-1]


def evaluate_risk(ctx: RunContext) -> dict:
    pe = tools.latest_period(ctx)
    r = {**ctx.calc.ratios(ctx.ticker, pe), **ctx.calc.growth(ctx.ticker, pe)}
    v = ctx.calc.valuation(ctx.ticker)
    rk = tools.calculate_risk_metrics(ctx)
    ctx.tracer.tool("evaluate_risk_rules", {"ticker": ctx.ticker, "dimensions": DIMENSIONS})
    matrix = []

    def add(dim, score, rec, why):
        evs = [ctx.evidence_from_metric(x, "risk") for x in rec if x is not None]
        cl = ctx.add_claim("risk", "derived_metric", f"{dim}风险{score}/5：{why}", evs, "risk")
        matrix.append({"dimension": dim, "score": score, "rationale": why, "claim_id": cl.claim_id})

    ocf = r.get("ocf_to_net_profit")
    n_anom = len([a for a in ctx.anomalies.values() if a["severity"] == "HIGH"])
    if ocf and ocf.value is not None:
        s = min(5, _band(-ocf.value, [-1.2, -1.0, -0.8, -0.5]) + n_anom // 2)
        add("财务质量", s, [ocf], f"经营现金流/净利润{fmt(ocf)}，高危异常{n_anom}项")
    else:
        add("财务质量", 5, [ocf] if ocf else [], f"经营现金流/净利润{fmt(ocf) if ocf else '数据不足'}，高危异常{n_anom}项")
    pct, pe_rec = v.get("pe_percentile_3y"), v.get("pe_ttm")
    if pct and pct.value is not None:
        add("估值", _band(pct.value, [0.2, 0.4, 0.6, 0.8]), [pct], f"PE-TTM处于近3年{fmt(pct)}分位")
    elif pe_rec:
        add("估值", 4, [pe_rec], f"PE-TTM{fmt(pe_rec)}，估值锚失效")
    dta, idr = r.get("debt_to_assets"), r.get("interest_debt_ratio")
    if dta and idr and dta.value is not None and idr.value is not None:
        add("偿债与流动性", _band(idr.value, [0.1, 0.2, 0.3, 0.45]), [dta, idr],
            f"资产负债率{fmt(dta)}，有息负债率{fmt(idr)}")
    ry, gm = r.get("revenue_yoy"), r.get("gross_margin")
    if ry and ry.value is not None:
        s = _band(-ry.value, [-0.2, -0.05, 0.0, 0.1])
        if "GROSS_MARGIN_SHIFT" in ctx.anomalies:
            s = min(5, s + 1)
        add("经营与行业", s, [ry, gm], f"营收同比{fmt(ry)}，毛利率{fmt(gm) if gm else '数据不足'}")
    titles = [t["title"] for t in ctx.timeline] or [d.title for d in ctx.svc.documents(ctx.ticker, "announcement")]
    hits = [t for t in titles if any(k in t for k in ("立案", "差错", "关联交易", "召回", "处罚"))]
    evs = [e for e in ctx.evidence.values() if e.kind == "document" and any(h in e.excerpt or h[:6] in e.summary for h in hits)]
    score = 1 + min(4, 2 * len(hits))
    if hits and evs:
        cl = ctx.add_claim("risk", "disclosed_fact", f"治理与合规风险{score}/5：涉及公告{len(hits)}项（{'、'.join(hits)}）",
                           evs[:2], "risk")
        matrix.append({"dimension": "治理与合规", "score": score, "rationale": cl.text, "claim_id": cl.claim_id})
    else:
        matrix.append({"dimension": "治理与合规", "score": score, "claim_id": None,
                       "rationale": "快照期内未见治理类负面公告" if not hits else f"涉及公告{len(hits)}项，原文证据未入库"})
    if rk:
        vol, mdd = rk["volatility_1y"], rk["max_drawdown_1y"]
        add("市场波动", _band(vol.value, [0.2, 0.3, 0.4, 0.55]), [vol, mdd], f"年化波动率{fmt(vol)}，最大回撤{fmt(mdd)}")
    return {"matrix": matrix, "dimensions": DIMENSIONS}
