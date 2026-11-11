"""B2: one ReAct agent with every tool (PRD v1.2 1.1).

What makes it a *single* agent, as the PRD defines the baseline:
* one shared transcript (thought, action, raw observation) that is re-sent in full at every step;
* a hard context budget (8k tokens). When the transcript outgrows it, the OLDEST observations are replaced by a stub,
  so facts read early can be forgotten on a long task;
* at most 8 tool steps, then the agent must answer;
* tool output goes into the transcript as-is. There is no sub-agent, no per-agent evidence card and no
  distillation: a claim can only cite the observation ids the model wrote down.
It has the same deterministic tools as the multi-agent system (calculators, PIT retrieval, announcements), so
numbers are as trustworthy as everywhere else. It has no Verifier, no debate and no re-planning.
"""
from __future__ import annotations

import json
from datetime import date

from data_platform.catalog import cn_name
from data_platform.schema import MetricRecord, StatementScope
from data_platform.serving import label_to_end, period_label

from . import tools
from .context import RunContext, fmt
from .llm import call_json, estimate_tokens

CONTEXT_BUDGET = 8000       # tokens the transcript may occupy (prompt overhead is counted separately)
MAX_STEPS = 8
OBS_CHARS = 1800            # one observation is cut to this many characters

TOOLS = {
    "get_statements": "读取某报告期的原始报表科目（营收、成本、净利润、经营现金流、应收、存货、资产负债等）。args: {\"period\":\"2025Q3\"}",
    "get_ratios": "读取某报告期的比率与增速（毛利率、净利率、ROE、OCF/净利润、周转天数、负债率、同比、环比）。args: {\"period\":\"2025Q3\"}",
    "get_valuation": "读取估值与风险指标（PE/PB/PS/EV-EBITDA、分位、同行PE中位数、波动率、回撤）。args: {}",
    "detect_anomalies": "运行异常检测，返回财务异常信号。args: {\"period\":\"2025Q3\"}",
    "search_documents": "在财报与公告原文中检索，返回最相关的片段。args: {\"query\":\"应收账款 账龄\",\"section\":\"NOTES|MDA|ANNOUNCEMENT|可省略\"}",
    "list_announcements": "列出研究时点前已公开的公告标题与日期。args: {}",
}
SYSTEM = (
    "你是投研分析师，用工具查数据后回答问题。每一步只输出一个JSON。\n"
    "调用工具：{\"thought\":\"为什么\",\"action\":\"工具名\",\"args\":{...}}\n"
    "给出最终答案：{\"final\":{\"can_answer\":true,\"values\":{\"<ask_id>\":数值},\"claims\":[{\"text\":\"一句话论断\","
    "\"sources\":[\"O1\",\"O3\"],\"type\":\"fact|metric|inference\",\"section\":\"answer|financials|valuation|events|bear|bull\"}]}}\n"
    "规则：1) 观察结果带编号 O1、O2…，论断必须在 sources 里写明依据的编号；2) 数字只能来自观察，不得自己编；"
    "3) values 里金额必须换算成“元”（观察里写“228.38亿元”，values 填 22838000000；“12.5万元”填 125000），"
    "比率/增速用小数（12.3% 填 0.123），倍数用倍（20.8倍填 20.8）；4) 工具里查不到、或问题要求预测/未披露信息时，can_answer 填 false 并说明；"
    f"5) 最多 {MAX_STEPS} 步工具调用，用完必须作答；6) 对话记录有长度上限，较早的观察可能被丢弃，重要数字请及时写进论断。\n"
    "可用工具：\n" + "\n".join(f"- {k}: {v}" for k, v in TOOLS.items()))


class Observation:
    def __init__(self, oid: str, tool: str, text: str, evidence: list):
        self.oid, self.tool, self.text, self.evidence = oid, tool, text, evidence
        self.dropped = False


def _render(records: dict) -> str:
    return "；".join(f"{cn_name(k)}={fmt(v)}" for k, v in records.items() if v is not None and getattr(v, "value", None) is not None)


def _period(ctx: RunContext, args: dict) -> date:
    p = args.get("period")
    return label_to_end(p) if p else tools.latest_period(ctx)


def run_tool(ctx: RunContext, name: str, args: dict) -> tuple[str, list]:
    """Execute one tool. Returns (raw text for the transcript, evidence cards it produced)."""
    if name == "get_statements":
        pe = _period(ctx, args)
        recs = tools.get_financial_statements(ctx, pe)
        return f"{period_label(pe)}报表：" + _render(recs), [ctx.evidence_from_metric(r, "react") for r in recs.values()]
    if name == "get_ratios":
        pe = _period(ctx, args)
        recs = {**tools.calculate_financial_ratios(ctx, pe), **tools.calculate_growth_metrics(ctx, pe)}
        return f"{period_label(pe)}比率与增速：" + _render(recs), [ctx.evidence_from_metric(r, "react") for r in recs.values()
                                                                    if r.value is not None]
    if name == "get_valuation":
        recs = {**tools.calculate_valuation_metrics(ctx), **tools.calculate_risk_metrics(ctx)}
        peer = tools.get_industry_peers(ctx)
        recs["peer_pe_median"] = peer
        return "估值与风险：" + _render(recs), [ctx.evidence_from_metric(r, "react") for r in recs.values() if r.value is not None]
    if name == "detect_anomalies":
        pe = _period(ctx, args)
        flags = tools.detect_financial_anomalies(ctx, pe)
        evs = [ctx.evidence_from_metric(ctx.metric(m), "react") for f in flags for m in f.metric_ids if ctx.metric(m)]
        return (f"{period_label(pe)}异常信号：" + ("；".join(f"[{f.code}]{f.message}" for f in flags) or "未发现")), evs
    if name == "search_documents":
        sec = args.get("section")
        hits = tools.retrieve_document_evidence(ctx, str(args.get("query", ""))[:80], [sec] if sec in ("NOTES", "MDA", "ANNOUNCEMENT") else None, top_k=3)
        evs = [ctx.evidence_from_hit(h, "react") for h in hits]
        return "\n".join(f"[{e.summary}] {e.excerpt[:600]}" for e in evs) or "未检索到相关片段", evs
    if name == "list_announcements":
        docs = tools.get_company_announcements(ctx)
        return "；".join(f"{d.published_at.date()} {d.title[:40]}" for d in docs[-12:]) or "无公告", []
    return f"未知工具 {name}", []


def _transcript(steps: list[dict], obs: dict[str, Observation]) -> str:
    """Re-render the whole history; the oldest observations are dropped until it fits the budget."""
    def build() -> str:
        parts = []
        for s in steps:
            parts.append(f"[步骤{s['n']}] 思考：{s.get('thought', '')}\n动作：{s['action']} {json.dumps(s.get('args', {}), ensure_ascii=False)}")
            o = obs[s["oid"]]
            parts.append(f"观察 {o.oid}：" + ("【已因上下文超限丢弃】" if o.dropped else o.text[:OBS_CHARS]))
        return "\n".join(parts)

    text = build()
    for s in steps:
        if estimate_tokens(text) <= CONTEXT_BUDGET:
            break
        obs[s["oid"]].dropped = True
        text = build()
    return text


def react(ctx: RunContext, asks) -> bool:
    """Run the loop; fills ctx.claims / ctx.answers / ctx.open_questions. Returns False if the model is unavailable."""
    wanted = [{"ask_id": a.ask_id, "metric": a.metric, "name": cn_name(a.metric.split(":")[-1]), "period": a.period}
              for a in asks if a.kind == "metric" and a.metric]
    head = json.dumps({"question": ctx.request.question, "as_of": ctx.as_of.isoformat(), "ticker": ctx.ticker,
                       "wanted": wanted}, ensure_ascii=False)
    steps: list[dict] = []
    obs: dict[str, Observation] = {}
    final = None
    for n in range(1, MAX_STEPS + 2):
        must_answer = n > MAX_STEPS
        user = head + "\n\n" + _transcript(steps, obs) + ("\n\n步骤已用完，现在必须输出 final。" if must_answer else "")
        d = call_json(ctx.llm, ctx.tracer, "react", SYSTEM, user, "strong",
                      validate=lambda d: isinstance(d.get("final"), dict) or isinstance(d.get("action"), str))
        if d is None:
            return bool(steps) and False
        if isinstance(d.get("final"), dict):
            final = d["final"]
            break
        if must_answer:
            break
        name, args = d["action"], d.get("args") if isinstance(d.get("args"), dict) else {}
        oid = f"O{n}"
        try:
            text, evs = run_tool(ctx, name, args)
        except Exception as exc:  # noqa: BLE001 - a tool failure is an observation the agent can react to
            text, evs = f"错误：{type(exc).__name__}: {exc}", []
        obs[oid] = Observation(oid, name, text, evs)
        steps.append({"n": n, "oid": oid, "action": name, "args": args, "thought": str(d.get("thought", ""))[:200]})
    if final is None:
        return True            # no answer after the step budget: report nothing rather than guess
    if final.get("can_answer") is False:
        ctx.open_questions.append({"topic": ctx.request.question[:60], "reason": str(final.get("reason") or "代理判断无法回答")[:120]})
    TYPE = {"fact": "disclosed_fact", "metric": "derived_metric", "inference": "model_inference"}
    SEC = {"answer", "financials", "valuation", "events", "bear", "bull"}
    for c in final.get("claims") or []:
        evs = [e for oid in c.get("sources") or [] if oid in obs for e in obs[oid].evidence]
        uniq = list({e.evidence_id: e for e in evs}.values())
        ctx.add_claim(c.get("section") if c.get("section") in SEC else "financials", TYPE.get(c.get("type"), "disclosed_fact"),
                      str(c.get("text", ""))[:300], uniq, "react")
    allev = [e for o in obs.values() for e in o.evidence]
    for a in asks:
        raw = (final.get("values") or {}).get(a.ask_id)
        if a.kind != "metric" or not a.metric or not isinstance(raw, (int, float)) or final.get("can_answer") is False:
            continue
        unit = ("ratio" if a.metric.endswith(("_yoy", "_qoq", "margin", "ocf_to_net_profit")) or a.metric == "gross_margin"
                else "x" if a.metric in ("pe_ttm", "pb_lf", "ev_ebitda") else "CNY")
        rec = MetricRecord(metric_id=f"m_react_{a.ask_id}", ticker=ctx.ticker, metric=a.metric, value=float(raw), unit=unit,
                           period=a.period or "?", statement_scope=StatementScope.CONSOLIDATED, formula="stated by the agent",
                           input_ids=[e.source_ref for e in allev[:2]], available_at=ctx.as_of, as_of=ctx.as_of)
        ctx.calc.store[rec.metric_id] = rec
        ctx.answers[a.ask_id] = {"ask": a.model_dump(), "metric_id": rec.metric_id, "value": rec.value, "status": "OK",
                                 "unit": unit, "display": fmt(rec)}
        if allev:
            ctx.add_claim("answer", "derived_metric", f"{cn_name(a.metric.split(':')[-1])}为{fmt(rec)}", allev[:1], "react")
    ctx.note(f"[react] steps={[s['action'] for s in steps]} dropped={[o for o, x in obs.items() if x.dropped]}")
    return True
