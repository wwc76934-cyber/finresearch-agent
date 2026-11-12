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

from data_platform.calculators import ROUTINE_RPT_RE
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
    if ctx.factor_only:
        return {}
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
    if ctx.factor_only:
        return {}
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


GROUP_CN = ["", "", "二", "三", "四", "五", "六", "七", "八", "九", "十"]


def _group_names(q: int) -> list[str]:
    if q == 3:
        return ["最低组", "中间组", "最高组"]
    return ["最低组"] + [f"第{GROUP_CN[i]}组" for i in range(2, q)] + ["最高组"]


def answer_factor_asks(ctx: RunContext, asks: list[Ask], agent: str) -> list[str]:
    """Scenario D: run the factor test and turn its statistics into MetricRecords the claims can cite."""
    from types import SimpleNamespace

    from data_platform.factors import FACTOR_CN, PanelCoverageError, PITViolationError

    ids: list[str] = []
    for a in asks:
        if a.kind != "factor_test":
            continue
        try:
            rep = tools.evaluate_factor(ctx, a.params)
        except (PITViolationError, PanelCoverageError, ValueError) as exc:
            pit = isinstance(exc, PITViolationError)
            ctx.tracer.tool("evaluate_factor_hypothesis", a.params, error=f"{type(exc).__name__}: {exc}")
            ctx.answers[a.ask_id] = {"ask": a.model_dump(), "metric_id": None, "value": None,
                                     "status": "PIT_VIOLATION" if pit else "INVALID_REQUEST", "error": str(exc)}
            ctx.open_questions.append({"ask_id": a.ask_id, "topic": f"因子检验被拒绝（{type(exc).__name__}）：{exc}",
                                       "reason": "请求会读取截面日之后才公开的数据，已按时点规则拒绝" if pit else "请求参数无效"})
            continue
        ref = SimpleNamespace(fact_id=f"factor_panel:{rep.factor}:{rep.universe}:h{rep.forward_horizon_days}:q{rep.quantiles}",
                              available_at=ctx.as_of)
        suffix = f"_{rep.factor}_{rep.universe}"
        extra = {"factor": rep.factor, "universe": rep.universe, "horizon_days": rep.forward_horizon_days}

        def rec(name, value, unit, sfx="", **more):
            return ctx.calc._record(ctx.ticker, name, rep.as_of_date, value, unit, [ref], extra={"cn": cn_name(name), **extra, **more},
                                    status="OK" if value is not None else "INSUFFICIENT_DATA", suffix=suffix + sfx)

        n_cs, n_st = rec("factor_n_cs", rep.n_cross_sections, "count"), rec("factor_n_stocks", rep.n_stocks_median, "count")
        hz, qn = rec("factor_horizon_days", rep.forward_horizon_days, "count"), rec("factor_quantiles", rep.quantiles, "count")
        name = FACTOR_CN[rep.factor]
        uni = {"all": "全部", "new_energy": "新能源", "consumer": "大消费"}[rep.universe]
        way = "越高越好" if rep.higher_is_better else "越低越好"
        for w in rep.warnings:
            ctx.open_questions.append({"ask_id": a.ask_id, "topic": f"样本局限[{w['code']}]：{w['message']}", "reason": "因子检验方法局限"})
        ctx.factor_reports.append(rep.model_dump())
        if rep.rank_ic_mean is None:
            cl = _claim(ctx, "answer", "derived_metric", [n_cs], lambda: f"{name}的横截面检验只有{fmt(n_cs)}个可用截面，不足以计算显著性。", agent)
            ids.append(cl.claim_id)
            ctx.answers[a.ask_id] = {"ask": a.model_dump(), "metric_id": n_cs.metric_id, "value": n_cs.value, "status": "INSUFFICIENT_DATA",
                                     "claim_id": cl.claim_id}
            continue
        ric, ic, ir = rec("factor_rank_ic_mean", rep.rank_ic_mean, "ic"), rec("factor_ic_mean", rep.ic_mean, "ic"), \
            rec("factor_rank_ic_ir", rep.rank_ic_ir, "ic")
        tt, pv = rec("factor_rank_ic_t", rep.rank_ic_t_stat, "stat"), rec("factor_rank_ic_p", rep.rank_ic_p_value, "stat")
        lsm, lsc, mdd = rec("factor_ls_mean", rep.long_short_mean_return, "ratio"), rec("factor_ls_cum", rep.long_short_cum_return, "ratio"), \
            rec("factor_ls_mdd", rep.long_short_max_drawdown, "ratio")
        names = _group_names(rep.quantiles)
        groups = [rec("factor_group_return", g, "ratio", sfx=f"_g{i + 1}", group=names[i])
                  for i, g in enumerate(rep.group_mean_returns)]
        c1 = _claim(ctx, "answer", "derived_metric", [n_cs, n_st, hz, ric, ic, ir],
                    lambda: (f"{name}（{way}）在{uni}样本池的横截面检验：{fmt(n_cs)}个季度截面，每个截面约{fmt(n_st)}家公司，"
                             f"持有{fmt(hz)}个交易日；RankIC均值{fmt(ric)}，IC均值{fmt(ic)}，RankIC_IR为{fmt(ir)}。"), agent)
        c2 = _claim(ctx, "answer", "derived_metric", [tt, pv],
                    lambda: f"RankIC序列的t统计量为{fmt(tt)}，双侧p值为{fmt(pv)}。", agent)
        c3 = _claim(ctx, "answer", "derived_metric", [qn, *groups, lsm, lsc, mdd],
                    lambda: ("分" + fmt(qn) + "组（因子值从低到高）平均后续收益：" + "，".join(f"{names[i]}{fmt(g)}" for i, g in enumerate(groups))
                             + f"；最高组减最低组每期{fmt(lsm)}，累计{fmt(lsc)}，最大回撤{fmt(mdd)}。"), agent)
        if rep.significant_5pct:
            concl = lambda: (f"推断：RankIC的p值为{fmt(pv)}，达到常规显著性水平；在该样本池和样本期内，因子值与后续收益存在统计上可辨认的"
                             f"{'正' if rep.rank_ic_mean > 0 else '负'}向关系，但样本很小，不能当作稳定规律。")
        else:
            concl = lambda: (f"推断：RankIC的p值为{fmt(pv)}，未达到常规显著性水平；在该样本池和样本期内，没有足够证据说明该因子能预测后续收益，"
                             f"RankIC均值{fmt(ric)}可能只是随机波动。")
        c4 = _claim(ctx, "quant", "model_inference", [ric, pv, n_cs], concl, agent)
        ids += [c1.claim_id, c2.claim_id, c3.claim_id, c4.claim_id]
        ctx.answers[a.ask_id] = {"ask": a.model_dump(), "metric_id": ric.metric_id, "value": ric.value, "status": "OK",
                                 "unit": "ic", "display": fmt(ric), "claim_id": c1.claim_id}
    return ids


def f_asks(s: SubState) -> dict:
    ctx = _ctx(s)
    return {"claim_ids": answer_asks(ctx, ctx.plan.asks, "fundamental", ()) + answer_factor_asks(ctx, ctx.plan.asks, "fundamental")}


def f_anomalies(s: SubState) -> dict:
    ctx = _ctx(s)
    if ctx.factor_only:
        return {}
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


# Accounting topics a question can name -> where the annual report explains them (PRD v1.2 1.1: multi-source tasks).
# Used by the Fundamental subgraph so that "analyse receivables with the annual-report notes" actually reads the notes.
NOTE_TOPICS = [
    (r"应收账款|应收款|回款|账龄", "应收账款 账龄 按欠款方归集 前五名 坏账准备"),
    (r"存货|跌价", "存货 跌价准备 账面余额 原材料 库存商品"),
    (r"借款|融资|负债|债券|偿债", "短期借款 长期借款 应付债券 一年内到期的非流动负债"),
    (r"商誉", "商誉 减值测试 资产组"),
    (r"减值", "资产减值损失 信用减值损失 计提"),
    (r"研发", "研发费用 研发投入 资本化"),
    (r"分红|股利|回购", "利润分配 现金分红 股份回购"),
]


def _named_report_hits(ctx: RunContext, question: str, query: str):
    """When the question names a report ("2025年年报", "2025年半年报", "2025年三季报"), read THAT report's notes."""
    m = re.search(r"(20\d{2})\s*年\s*(年报|年度报告|半年报|中报|三季报|一季报)", question)
    if not m:
        return []
    q = {"年报": 4, "年度报告": 4, "半年报": 2, "中报": 2, "三季报": 3, "一季报": 1}[m.group(2)]
    ids = [d.document_id for d in ctx.svc.documents(ctx.ticker) if d.doc_type != "announcement" and d.period_end
           and d.period_end.year == int(m.group(1)) and (d.period_end.month - 1) // 3 + 1 == q and "_v" not in d.document_id]
    return tools.retrieve_document_evidence(ctx, query, ["NOTES"], top_k=2, doc_ids=ids) if ids else []


def f_topic_notes(s: SubState) -> dict:
    """Read the NOTES for the accounting topic the question names, whether or not an anomaly fired."""
    ctx = _ctx(s)
    if ctx.factor_only or not ctx.config.subgraph_crosscheck or ctx.route.fast:
        return {}
    q = ctx.request.question
    ids = []
    for pat, query in NOTE_TOPICS:
        if not re.search(pat, q):
            continue
        hit_ids = _named_report_hits(ctx, q, query) or tools.retrieve_latest_first(ctx, query, ["NOTES"], top_k=2)
        for h in hit_ids[:1]:
            c = h.parent or h.chunk
            body = c.text.splitlines()
            rows = [r for r in body if r.startswith("|") and "---" not in r][1:4]
            if not rows:
                continue
            finding = "；".join(" ".join(x.strip() for x in r.strip("|").split("|") if x.strip()) for r in rows) + "（单位见原表）。"
            ev = ctx.evidence_from_hit(h, "fundamental", focus=rows[0])
            cl = ctx.add_claim("financials", "disclosed_fact", f"年报附注[{c.section_path.split(' > ')[-1]}]：{finding}", [ev], "fundamental")
            ids.append(cl.claim_id)
        if ids:
            break          # one topic per question keeps the subgraph inside its tool budget
    return {"claim_ids": ids}


# Accounting identities the Fundamental subgraph checks on its own (PRD v1.2 1.1). A break in either one sends the
# subgraph straight to the notes that carry the line items, before anything is returned to the main graph.
# code -> (query, section). Management explains a profit/cash-flow break in MD&A; the receivables ageing and customer
# concentration that explain a receivables build-up are in the NOTES.
CROSSCHECK = {"PROFIT_OCF_DIVERGENCE": ("经营活动现金流量净额 下降 主要系 回款 原因", "MDA"),
              "AR_OUTPACE_REVENUE": ("应收账款 账龄 按欠款方归集 前五名 坏账准备", "NOTES")}


def f_crosscheck(s: SubState) -> dict:
    """Revenue vs receivables vs profit vs operating cash flow. When they disagree, retrieve the NOTES line items
    inside this subgraph and return them as one compact evidence-backed claim per divergence."""
    ctx = _ctx(s)
    if ctx.factor_only or not ctx.config.subgraph_crosscheck or ctx.route.fast:
        return {}
    ids = []
    for code, (query, section) in CROSSCHECK.items():
        an = ctx.anomalies.get(code)
        if not an or an.get("crosschecked"):
            continue
        found = explain_anomaly(ctx, code, query, [section], an.get("evidence_ids", []), "fundamental", prefix="交叉核验")
        an["crosschecked"] = bool(found)
        ids += found
    return {"claim_ids": ids}


def f_probe(s: SubState) -> dict:
    """Semantic anomaly probe: qualitative red flags from notes and announcements (online only)."""
    from . import semantic_probe

    ctx = _ctx(s)
    if ctx.factor_only or not ctx.config.semantic_probe or ctx.route.fast:
        return {}
    ids, out = [], []
    for fl in semantic_probe.probe(ctx):
        ev = ctx.evidence_from_hit(fl["hit"], "fundamental", focus=fl["quote"])
        cl = ctx.add_claim("bear" if fl["severity"] == "HIGH" else "financials", "disclosed_fact",
                           f"语义探针[{fl['code']}]：{fl['quote']}", [ev], "fundamental")
        d = {"code": fl["code"], "severity": fl["severity"], "category": "semantic", "message": fl["quote"],
             "metric_ids": [], "suggested_query": semantic_probe_query(fl["category"]), "section_type": "NOTES",
             "claim_id": cl.claim_id, "evidence_ids": cl.evidence_ids, "evidence_quote": fl["quote"]}
        ctx.anomalies.setdefault(fl["code"], d)
        out.append(d)
        ids.append(cl.claim_id)
    return {"anomalies": out, "claim_ids": ids}


def semantic_probe_query(category: str) -> str:
    from .semantic_probe import PROBES

    return next((q for c, _, q in PROBES if c == category), "")


EXPLAIN_LINE = re.compile(r"[^。\n]*(主要系|所致|原因)[^。\n]*。")
RULE_TEXT = re.compile(r"应(当)?说明|解释性公告|信息披露编报规则")
CHECKBOX = re.compile(r"^.*适用[\s□√]*不适用\s*")   # form heading "……原因说明√适用□不适用" before the real text


def f_explain(s: SubState) -> dict:
    ctx = _ctx(s)
    if ctx.factor_only:
        return {}
    pe = _period(ctx, s)
    q = "营业收入 净利润 毛利率 经营活动现金流量净额 变动 原因 管理层讨论"
    hits = tools.retrieve_latest_first(ctx, q, ["MDA"], top_k=3)
    ids = []
    for h in hits:
        text = (h.parent or h.chunk).text
        # skip disclosure-rule text quoted inside reports ("……应说明原因"): it explains nothing
        m = next((x for x in EXPLAIN_LINE.finditer(text) if not RULE_TEXT.search(x.group(0))), None)
        if m:
            sentence = CHECKBOX.sub("", m.group(0))[-300:]
            if len(sentence) < 15 or not re.search(r"主要系|所致|原因", sentence):
                continue   # the keyword was in the form heading, not in an explanation
            ev = ctx.evidence_from_hit(h, "fundamental", focus=sentence)
            cl = ctx.add_claim("financials", "disclosed_fact", f"管理层讨论与分析披露：{sentence}", [ev], "fundamental")
            ids.append(cl.claim_id)
            break
    return {"claim_ids": ids}


def f_summarize(s: SubState) -> dict:
    ctx = _ctx(s)
    claims = [ctx.claims[c] for c in s.get("claim_ids", [])]
    if ctx.factor_only:   # statistics are already final; nothing for a model to summarise
        return {"summary": "；".join(c.text for c in claims[:2])[:200]}
    scratch = "\n".join(s.get("scratch", []))
    sys_p = "你是基本面分析Agent。用不超过120字总结要点，只能引用给定 claims，不得新增数字。输出JSON {\"summary\":\"...\"}"
    user = scratch + "\n" + "\n".join(f"{c.claim_id}: {c.text}" for c in claims)
    tier = "light"   # a 120-char recap that only feeds downstream pointers: no reason to pay for the strong model (PRD 3.2 model tiering)
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
# the legal statement every A-share announcement opens with; quoting it says nothing about the event
_BOILERPLATE_RE = re.compile(r"^.*?(本公司|公司)?董事会及全体董事保证.*?(承担(个别及连带)?(法律)?责任|连带责任)。")


def _strip_boilerplate(body: str) -> str:
    m = _BOILERPLATE_RE.match(body)
    if m and len(body) - m.end() >= 20:
        body = body[m.end():]
    return re.sub(r"^\s*重要内容提示[：:]\s*", "", body).lstrip("●• ")


def n_events(s: SubState) -> dict:
    ctx = _ctx(s)
    docs = tools.get_company_announcements(ctx)
    ctx.timeline = tools.build_event_timeline(ctx, docs)
    ids = []
    for d in docs[-5:]:
        hits = tools.retrieve_document_evidence(ctx, d.title, doc_ids=[d.document_id], top_k=1)
        if not hits:
            continue
        body = _strip_boilerplate((hits[0].parent or hits[0].chunk).text.split("\n")[-1])
        if len(body) > 220:   # real announcements are long; quote the opening and cut at a sentence end
            cut = body[:220]
            body = cut[:cut.rfind("。") + 1] if "。" in cut else cut + "…"
        ev = ctx.evidence_from_hit(hits[0], "news", focus=body.rstrip("…")[:120])
        sec = "bear" if any(k in d.title for k in ("召回", "差错", "关联交易", "立案", "债券", "借款")) \
            and not ROUTINE_RPT_RE.search(d.title) else ("bull" if "合同" in d.title else "events")
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
                             ("crosscheck", f_crosscheck), ("topic", f_topic_notes), ("probe", f_probe), ("explain", f_explain), ("summarize", f_summarize)]),
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
