"""Parallel specialised subgraphs (Map stage) + anomaly investigation subgraph.

Each agent is its own compiled LangGraph StateGraph. Tool outputs and
intermediate notes stay in the subgraph's private `scratch`; only a
SubTaskSummary (ids + short text) is returned to the parent graph.
"""
from __future__ import annotations

import operator
import re
from datetime import date
from typing import Annotated, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from data_platform.catalog import cn_name
from data_platform.serving import label_to_end, period_label

from .. import tools
from ..context import RunContext, fmt, get_context
from ..llm import call_json
from ..state import Ask, SubTaskSummary
from ..tools import ToolError


class SubState(TypedDict, total=False):
    run_id: str
    task: dict
    period: str
    scratch: Annotated[list[str], operator.add]
    evidence_ids: Annotated[list[str], operator.add]
    claim_ids: Annotated[list[str], operator.add]
    anomalies: Annotated[list[dict], operator.add]
    summary: str


def _ctx(s: SubState) -> RunContext:
    return get_context(s["run_id"])


def _claim(ctx: RunContext, section: str, ctype: str, recs: list, render: Callable[[], str], agent: str,
           extra_evidence: Optional[list] = None):
    evs = [ctx.evidence_from_metric(r, agent) for r in recs if r is not None]
    evs += extra_evidence or []
    cl = ctx.add_claim(section, ctype, render(), evs, agent)
    ctx.renderers[cl.claim_id] = render
    return cl


def _period(ctx: RunContext, s: SubState) -> date:
    return label_to_end(s["period"]) if s.get("period") else tools.latest_period(ctx)


def _amount(rec, unit: Optional[str] = None) -> str:
    if rec is None or rec.value is None:
        return fmt(rec) if rec else "数据不足"
    if unit == "万元":
        return f"{rec.value / 1e4:.2f}万元"
    if unit == "元":
        return f"{rec.value:.2f}元"
    return fmt(rec)


# ======================= Fundamental =======================
def f_fetch(s: SubState) -> dict:
    ctx = _ctx(s)
    pe = _period(ctx, s)
    try:
        st = tools.get_financial_statements(ctx, pe)
    except ToolError as exc:
        ctx.open_questions.append({"topic": "财务报表获取失败", "reason": str(exc)})
        return {"scratch": [f"fetch failed: {exc}"], "period": period_label(pe)}
    raw = "\n".join(f"{k}={v.value:.2f} ({v.input_ids[0]})" for k, v in st.items())
    ctx.note(f"[fundamental raw statements]\n{raw}")
    return {"scratch": [raw], "period": period_label(pe)}


def f_compute(s: SubState) -> dict:
    ctx = _ctx(s)
    pe = _period(ctx, s)
    r = tools.calculate_financial_ratios(ctx, pe)
    g = tools.calculate_growth_metrics(ctx, pe)
    rev = ctx.calc.raw(ctx.ticker, "operating_revenue", pe)
    np_ = ctx.calc.raw(ctx.ticker, "net_profit", pe)
    lbl = period_label(pe)
    ids = []
    if rev and np_:
        ids.append(_claim(ctx, "financials", "disclosed_fact", [rev, g["revenue_yoy"], np_, g["net_profit_yoy"]],
                          lambda: (f"{lbl}累计营业收入{fmt(rev)}，同比{fmt(g['revenue_yoy'])}；归母净利润{fmt(np_)}，"
                                   f"同比{fmt(g['net_profit_yoy'])}。"), "fundamental").claim_id)
    if "gross_margin" in r and "net_margin" in r:
        ids.append(_claim(ctx, "financials", "derived_metric", [r["gross_margin"], r["net_margin"], r.get("roe_annualized")],
                          lambda: (f"{lbl}毛利率{fmt(r['gross_margin'])}，净利率{fmt(r['net_margin'])}"
                                   + (f"，年化ROE{fmt(r['roe_annualized'])}" if r.get("roe_annualized") else "") + "。"),
                          "fundamental").claim_id)
    if "ocf_to_net_profit" in r:
        ocf = r["ocf_to_net_profit"]
        ids.append(_claim(ctx, "financials", "derived_metric", [ocf, g["ocf_yoy"]],
                          lambda: (f"{lbl}经营现金流/净利润为{fmt(ocf)}，经营现金流同比{fmt(g['ocf_yoy'])}。"),
                          "fundamental").claim_id)
        if ocf.value is not None:
            if ocf.value < 0.8:
                ids.append(_claim(ctx, "financials", "model_inference", [ocf],
                                  lambda: f"推断：经营现金流/净利润仅{fmt(ocf)}，利润的现金含量偏低，盈利质量需打折看待。",
                                  "fundamental").claim_id)
            elif ocf.value > 1.0:
                ids.append(_claim(ctx, "financials", "model_inference", [ocf],
                                  lambda: f"推断：经营现金流/净利润为{fmt(ocf)}，利润具有较好的现金支撑。",
                                  "fundamental").claim_id)
    if "receivable_days" in r and "inventory_days" in r:
        ids.append(_claim(ctx, "financials", "derived_metric", [r["receivable_days"], r["inventory_days"]],
                          lambda: f"{lbl}应收账款周转天数{fmt(r['receivable_days'])}，存货周转天数{fmt(r['inventory_days'])}。",
                          "fundamental").claim_id)
    rq, nq = g.get("revenue_qoq"), g.get("net_profit_qoq")
    if rq is not None and rq.value is not None:
        ids.append(_claim(ctx, "financials", "derived_metric", [rq, nq],
                          lambda: f"{lbl}单季营收环比{fmt(rq)}，单季归母净利润环比{fmt(nq)}。", "fundamental").claim_id)
    if "debt_to_assets" in r and "interest_debt_ratio" in r:
        ids.append(_claim(ctx, "financials", "derived_metric", [r["debt_to_assets"], r["interest_debt_ratio"]],
                          lambda: f"{lbl}资产负债率{fmt(r['debt_to_assets'])}，有息负债率{fmt(r['interest_debt_ratio'])}。",
                          "fundamental").claim_id)
    # quarterly chart series (deterministic)
    series, d = [], pe
    for _ in range(8):
        rq = ctx.calc.single_quarter(ctx.ticker, "operating_revenue", d)
        nq = ctx.calc.single_quarter(ctx.ticker, "net_profit", d)
        cq = ctx.calc.single_quarter(ctx.ticker, "operating_cost", d)
        gm = (rq.value - cq.value) / rq.value if rq.value and cq.value is not None else None
        series.append({"period": period_label(d), "revenue": rq.value, "net_profit": nq.value, "gross_margin": gm,
                       "status": rq.status})
        m = d.month
        d = date(d.year, m - 3, {3: 31, 6: 30, 9: 30}[m - 3]) if m > 3 else date(d.year - 1, 12, 31)
    ctx.charts["quarterly"] = list(reversed(series))
    note = "; ".join(f"{k}={v.value}" for k, v in {**r, **g}.items())
    ctx.note(f"[fundamental metrics]\n{note}")
    return {"claim_ids": ids, "scratch": [note]}


def answer_asks(ctx: RunContext, asks: list[Ask], agent: str, kinds: tuple[str, ...]) -> list[str]:
    ids = []
    val_cache = None
    for a in asks:
        if a.kind != "metric" or not a.metric:
            continue
        m = a.metric
        is_market = m in ("pe_ttm", "pb_lf", "pe_percentile_3y", "pe_percentile_5y", "volatility_1y", "max_drawdown_1y",
                          "ps_ttm", "ev_ebitda")
        if (agent == "market") != is_market:
            continue
        pe = label_to_end(a.period) if a.period else tools.latest_period(ctx)
        rec = None
        if m.startswith("single_quarter:"):
            rec = tools.single_quarter(ctx, m.split(":")[1], pe)
            name = f"{period_label(pe)}单季{cn_name(m.split(':')[1])}"
        elif m in ("operating_revenue", "net_profit"):
            rec = tools.get_fact(ctx, m, pe, a.scope)
            name = f"{period_label(pe)}{'母公司' if a.scope == 'parent' else '合并'}口径累计{cn_name(m)}"
        elif is_market:
            if val_cache is None:
                val_cache = {**tools.calculate_valuation_metrics(ctx), **tools.calculate_risk_metrics(ctx)}
            rec = val_cache.get(m)
            name = cn_name(m)
        else:
            allm = {**ctx.calc.ratios(ctx.ticker, pe), **ctx.calc.growth(ctx.ticker, pe)}
            rec = allm.get(m)
            name = f"{period_label(pe)}{cn_name(m)}"
        if rec is None:
            ctx.answers[a.ask_id] = {"ask": a.model_dump(), "metric_id": None, "value": None, "status": "INSUFFICIENT_DATA"}
            ctx.open_questions.append({"ask_id": a.ask_id, "topic": f"{a.metric} {a.period}", "reason": "快照中无可用数据"})
            continue
        ctx.answers[a.ask_id] = {"ask": a.model_dump(), "metric_id": rec.metric_id, "value": rec.value,
                                 "status": rec.status, "unit": rec.unit, "display": _amount(rec, a.unit)}
        cl = _claim(ctx, "answer", "derived_metric" if rec.extra.get("kind") != "fact" else "disclosed_fact", [rec],
                    lambda rec=rec, name=name, unit=a.unit: f"{name}为{_amount(rec, unit)}。", agent)
        ctx.answers[a.ask_id]["claim_id"] = cl.claim_id
        ids.append(cl.claim_id)
    return ids


def f_asks(s: SubState) -> dict:
    ctx = _ctx(s)
    return {"claim_ids": answer_asks(ctx, ctx.plan.asks, "fundamental", ())}


def f_anomalies(s: SubState) -> dict:
    ctx = _ctx(s)
    pe = _period(ctx, s)
    flags = tools.detect_financial_anomalies(ctx, pe)
    out, ids = [], []
    for fl in flags:
        recs = [ctx.metric(m) for m in fl.metric_ids]
        ann_ev = []
        if not recs and fl.code == "RELATED_PARTY_OR_POLICY_CHANGE":
            hits = tools.retrieve_document_evidence(ctx, fl.suggested_query, ["ANNOUNCEMENT"], top_k=1)
            ann_ev = [ctx.evidence_from_hit(h, "fundamental") for h in hits]
        cl = _claim(ctx, "bear" if fl.severity == "HIGH" else "financials", "derived_metric", recs,
                    lambda fl=fl: f"异常信号[{fl.code}]：{fl.message}。", "fundamental", ann_ev)
        d = fl.model_dump() | {"claim_id": cl.claim_id, "evidence_ids": cl.evidence_ids}
        ctx.anomalies[fl.code] = d
        out.append(d)
        ids.append(cl.claim_id)
        if fl.severity == "MEDIUM" and fl.code != "RELATED_PARTY_OR_POLICY_CHANGE":
            # MEDIUM flags do not justify a re-plan (PRD 3.3), but the subgraph spends one targeted
            # lookup of its own budget to attach the disclosed explanation when there is one
            found = explain_anomaly(ctx, fl.code, fl.suggested_query, [fl.section_type], cl.evidence_ids,
                                    "fundamental", prefix="说明")
            ids += found
    ctx.tracer._cur() and ctx.tracer._cur()["anomaly_flags"].extend([f.code for f in flags])
    return {"anomalies": out, "claim_ids": ids}


EXPLAIN_LINE = re.compile(r"[^。\n]*(主要系|所致|原因)[^。\n]*。")


def f_explain(s: SubState) -> dict:
    ctx = _ctx(s)
    pe = _period(ctx, s)
    q = "营业收入 净利润 毛利率 经营活动现金流量净额 变动 原因 管理层讨论"
    hits = tools.retrieve_latest_first(ctx, q, ["MDA"], top_k=3)
    ids = []
    for h in hits:
        text = (h.parent or h.chunk).text
        m = EXPLAIN_LINE.search(text)
        if m:
            sentence = m.group(0)[-300:]
            ev = ctx.evidence_from_hit(h, "fundamental", focus=sentence)
            cl = ctx.add_claim("financials", "disclosed_fact", f"管理层讨论与分析披露：{sentence}", [ev], "fundamental")
            ids.append(cl.claim_id)
            break
    return {"claim_ids": ids}


def f_summarize(s: SubState) -> dict:
    ctx = _ctx(s)
    claims = [ctx.claims[c] for c in s.get("claim_ids", [])]
    scratch = "\n".join(s.get("scratch", []))
    sys_p = "你是基本面分析Agent。用不超过120字总结要点，只能引用给定 claims，不得新增数字。输出JSON {\"summary\":\"...\"}"
    user = scratch + "\n" + "\n".join(f"{c.claim_id}: {c.text}" for c in claims)
    tier = "light" if s["task"]["agent"] in ("market", "news") else "strong"  # PRD 3.2 model tiering
    data = call_json(ctx.llm, ctx.tracer, s["task"]["agent"], sys_p, user, tier,
                     validate=lambda d: isinstance(d.get("summary"), str))
    summary = data["summary"] if data else "；".join(c.text for c in claims[:3])[:200]
    return {"summary": summary}


# ======================= Market =======================
def _fv(v: dict, key: str) -> str:
    return fmt(v[key]) if v.get(key) is not None else "数据不足"
def m_valuation(s: SubState) -> dict:
    ctx = _ctx(s)
    v = tools.calculate_valuation_metrics(ctx)
    peer = tools.get_industry_peers(ctx)
    ids = []
    pe, pct = v.get("pe_ttm"), v.get("pe_percentile_3y")
    if pe is not None:
        if pe.status == "OK":
            ids.append(_claim(ctx, "valuation", "derived_metric", [pe, pct, v.get("pb_lf"), peer],
                              lambda: (f"截至{ctx.as_of.date()}，PE-TTM为{fmt(pe)}，近3年历史分位{fmt(pct)}，"
                                       f"PB-LF为{_fv(v, 'pb_lf')}；同行PE中位数{fmt(peer)}。"), "market").claim_id)
            if pct.value is not None:
                if pct.value >= 0.7:
                    ids.append(_claim(ctx, "bear", "model_inference", [pct],
                                      lambda: f"推断：估值处于近3年{fmt(pct)}分位的偏高区间，对业绩不及预期的容忍度低。",
                                      "market").claim_id)
                elif pct.value <= 0.3:
                    ids.append(_claim(ctx, "bull", "model_inference", [pct],
                                      lambda: f"推断：估值处于近3年{fmt(pct)}分位的偏低区间，具备一定安全边际。",
                                      "market").claim_id)
        else:
            ids.append(_claim(ctx, "valuation", "derived_metric", [pe, v.get("pb_lf"), v.get("ps_ttm")],
                              lambda: (f"PE-TTM{fmt(pe)}，改用PB-LF {_fv(v, 'pb_lf')}与PS-TTM {_fv(v, 'ps_ttm')}衡量估值。"),
                              "market").claim_id)
    ev, ps, p5 = v.get("ev_ebitda"), v.get("ps_ttm"), v.get("pe_percentile_5y")
    if ev is not None or (p5 is not None and p5.value is not None):
        parts = []
        if ev is not None:
            parts.append(lambda: f"EV/EBITDA为{fmt(ev)}")
        if ps is not None:
            parts.append(lambda: f"PS-TTM为{fmt(ps)}")
        if p5 is not None and p5.value is not None:
            parts.append(lambda: f"PE近5年分位{fmt(p5)}")
        ids.append(_claim(ctx, "valuation", "derived_metric", [x for x in (ev, ps, p5) if x is not None],
                          lambda parts=parts: "，".join(f() for f in parts) + "。", "market").claim_id)
    risk = tools.calculate_risk_metrics(ctx)
    if risk:
        ids.append(_claim(ctx, "valuation", "derived_metric", [risk["volatility_1y"], risk["max_drawdown_1y"]],
                          lambda: (f"近一年年化波动率{fmt(risk['volatility_1y'])}，最大回撤{fmt(risk['max_drawdown_1y'])}。"),
                          "market").claim_id)
    px = tools.get_price_history(ctx, 250)
    ctx.charts["price"] = [{"date": p.trade_date.isoformat(), "close": p.close} for p in px]
    if pct is not None and pct.extra:
        ctx.charts["pe_band"] = {"current": pe.value, **pct.extra}
    ctx.charts["peers"] = peer.extra.get("peer_values", {})
    note = f"valuation={ {k: x.value for k, x in v.items()} } peer={peer.value}"
    ctx.note(f"[market]\n{note}\nprices={[(p.trade_date.isoformat(), p.close) for p in px[-60:]]}")
    ids += answer_asks(ctx, ctx.plan.asks, "market", ())
    return {"claim_ids": ids, "scratch": [note]}


# ======================= News / Event =======================
def n_events(s: SubState) -> dict:
    ctx = _ctx(s)
    docs = tools.get_company_announcements(ctx)
    ctx.timeline = tools.build_event_timeline(ctx, docs)
    ids = []
    for d in docs[-5:]:
        hits = tools.retrieve_document_evidence(ctx, d.title, doc_ids=[d.document_id], top_k=1)
        if not hits:
            continue
        body = (hits[0].parent or hits[0].chunk).text.split("\n")[-1]
        if len(body) > 220:   # real announcements are long; quote the opening and cut at a sentence end
            cut = body[:220]
            body = cut[:cut.rfind("。") + 1] if "。" in cut else cut + "…"
        ev = ctx.evidence_from_hit(hits[0], "news", focus=body.rstrip("…")[:120])
        sec = "bear" if any(k in d.title for k in ("召回", "差错", "关联交易", "立案", "债券", "借款")) else \
            ("bull" if "合同" in d.title else "events")
        cl = ctx.add_claim(sec if sec != "events" else "events", "disclosed_fact",
                           f"{d.published_at.date()}公告《{d.title}》：{body}", [ev], "news")
        ids.append(cl.claim_id)
    ctx.note(f"[news] timeline={ctx.timeline}")
    return {"claim_ids": ids, "scratch": [str(ctx.timeline)]}


# ======================= Investigation (re-planning) =======================
DETAIL_LINE = re.compile(r"[^。\n|]*(合计|主要系|所致|主要原因|变动原因|原因是|计提|新增|调减|调增)[^。\n|]*。")


def explain_anomaly(ctx: RunContext, code: str, query: str, section_types, anomaly_evidence_ids: list[str],
                    agent: str, prefix: str = "追查") -> list[str]:
    """Find the disclosed explanation (MD&A sentence or note table) for an anomaly and bind it to the claim."""
    hits = tools.retrieve_latest_first(ctx, query, section_types, top_k=4)
    key = query.split()[0]
    hits = sorted(hits, key=lambda h: 0 if key in (h.parent or h.chunk).section_path else 1)  # topical match first
    used = {c.evidence_ids[0] for c in ctx.claims.values() if c.produced_by == agent and c.evidence_ids
            and code in c.text}
    for h in hits[:3]:
        c = h.parent or h.chunk
        m = DETAIL_LINE.search(c.text)
        if m:
            finding = focus = m.group(0)[-260:]
        elif c.is_table:
            rows = [r for r in c.text.splitlines() if r.startswith("|") and "---" not in r][1:4]
            if not rows:
                continue
            finding = "；".join(" ".join(x.strip() for x in r.strip("|").split("|") if x.strip()) for r in rows) + "（单位见原表）。"
            focus = rows[0]
        else:
            continue
        ev = ctx.evidence_from_hit(h, agent, focus=focus)
        if ev.evidence_id in used:
            continue
        an_evs = [ctx.evidence[e] for e in anomaly_evidence_ids if e in ctx.evidence]
        cl = ctx.add_claim("bear", "disclosed_fact", f"{prefix}[{code}]：{c.section_path.split(' > ')[-1]}显示，{finding}",
                           [ev, *an_evs], agent)
        return [cl.claim_id]
    return []


def i_investigate(s: SubState) -> dict:
    ctx = _ctx(s)
    t = s["task"]
    an = ctx.anomalies.get(t["anomaly_code"], {})
    ids = explain_anomaly(ctx, t["anomaly_code"], t["query"], t.get("section_types"), an.get("evidence_ids", []),
                          "investigate")
    if not ids:
        ctx.open_questions.append({"topic": f"{t['anomaly_code']} 追查未找到解释性证据", "reason": "检索不充分"})
    return {"claim_ids": ids, "summary": f"investigated {t['anomaly_code']}: {len(ids)} claim(s)"}


def _compile(steps: list[tuple[str, Callable]]):
    g = StateGraph(SubState)
    prev = START
    for name, fn in steps:
        g.add_node(name, fn)
        g.add_edge(prev, name)
        prev = name
    g.add_edge(prev, END)
    return g.compile()


SUBGRAPHS = {
    "fundamental": _compile([("fetch", f_fetch), ("compute", f_compute), ("asks", f_asks), ("anomalies", f_anomalies),
                             ("explain", f_explain), ("summarize", f_summarize)]),
    "market": _compile([("valuation", m_valuation), ("summarize", f_summarize)]),
    "news": _compile([("events", n_events), ("summarize", f_summarize)]),
    "investigate": _compile([("investigate", i_investigate)]),
}


def run_subgraph(run_id: str, task: dict) -> SubTaskSummary:
    ctx = get_context(run_id)
    with ctx.tracer.node(f"subgraph:{task['agent']}:{task['task_id']}") as span:
        out = SUBGRAPHS[task["agent"]].invoke({"run_id": run_id, "task": task, "scratch": [], "evidence_ids": [],
                                               "claim_ids": [], "anomalies": []})
        claim_ids = out.get("claim_ids", [])
        ev_ids = sorted({e for c in claim_ids for e in ctx.claims[c].evidence_ids})
        summ = SubTaskSummary(task_id=task["task_id"], agent=task["agent"], summary=out.get("summary", ""),
                              evidence_ids=ev_ids, metric_ids=sorted({m for c in claim_ids for m in ctx.claims[c].metric_ids}),
                              anomalies=[{"code": a["code"], "severity": a["severity"]} for a in out.get("anomalies", [])],
                              claim_ids=claim_ids)
        span["summary"] = f"{len(claim_ids)} claims, {len(ev_ids)} evidence, scratch={len(out.get('scratch', []))} private notes"
        # private scratch never leaves the subgraph unless isolation is disabled
        span["input_tokens"] += 0
        return summ
