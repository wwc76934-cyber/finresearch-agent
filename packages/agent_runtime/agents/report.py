"""Report Agent: assembles verified claims, lineage pointers and ECharts options."""
from __future__ import annotations

import json

from ..context import RunContext
from ..llm import call_json

SECTIONS = [("overview", "公司与行业概况"), ("answer", "问题直接回答"), ("quant", "因子检验结论"), ("financials", "财务与经营质量"),
            ("valuation", "估值与市场表现"), ("events", "近期公告与事件"), ("bull", "核心看多论点"),
            ("bear", "核心看空论点"), ("falsification", "关键争议与证伪条件"), ("risk", "六维风险矩阵")]


def charts(ctx: RunContext, risk: dict) -> dict:
    out = {}
    q = ctx.charts.get("quarterly", [])
    if q:
        out["quarterly"] = {
            "title": {"text": "单季营收/归母净利润(亿元)与毛利率"}, "tooltip": {"trigger": "axis"},
            "legend": {"data": ["营收", "净利润", "毛利率"]}, "xAxis": {"type": "category", "data": [x["period"] for x in q]},
            "yAxis": [{"type": "value", "name": "亿元"}, {"type": "value", "name": "%"}],
            "series": [{"name": "营收", "type": "bar", "data": [round(x["revenue"] / 1e8, 2) if x["revenue"] else None for x in q]},
                       {"name": "净利润", "type": "bar", "data": [round(x["net_profit"] / 1e8, 2) if x["net_profit"] is not None else None for x in q]},
                       {"name": "毛利率", "type": "line", "yAxisIndex": 1,
                        "data": [round(x["gross_margin"] * 100, 2) if x["gross_margin"] is not None else None for x in q]}]}
    px = ctx.charts.get("price", [])
    if px:
        out["price"] = {"title": {"text": "近一年收盘价"}, "tooltip": {"trigger": "axis"},
                        "xAxis": {"type": "category", "data": [p["date"] for p in px]}, "yAxis": {"type": "value", "scale": True},
                        "series": [{"type": "line", "showSymbol": False, "data": [p["close"] for p in px]}]}
    band = ctx.charts.get("pe_band")
    peers = ctx.charts.get("peers", {})
    if band and band.get("current"):
        names = ["3年最低", "3年中位", "3年最高", "当前"] + [f"同行{k[:6]}" for k in peers]
        vals = [band["min"], band["median"], band["max"], band["current"]] + list(peers.values())
        out["valuation"] = {"title": {"text": "PE-TTM 历史分位与同行对比"}, "tooltip": {},
                            "xAxis": {"type": "category", "data": names}, "yAxis": {"type": "value", "name": "倍"},
                            "series": [{"type": "bar", "data": [round(v, 2) for v in vals]}]}
    for i, fr in enumerate(ctx.factor_reports):
        cs = fr["cross_sections"]
        if not cs:
            continue
        out[f"factor_rank_ic_{i}"] = {
            "title": {"text": f"{fr['factor']} 横截面 RankIC（{fr['universe']}）"}, "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": [c["date"] for c in cs]}, "yAxis": {"type": "value", "name": "RankIC"},
            "series": [{"type": "bar", "data": [round(c["rank_ic"], 4) for c in cs],
                        "markLine": {"data": [{"type": "average", "name": "均值"}]}}]}
        keys = [k for k in fr["nav"][0] if k != "date"]
        out[f"factor_nav_{i}"] = {
            "title": {"text": "分组与多空累计净值"}, "tooltip": {"trigger": "axis"}, "legend": {"data": keys},
            "xAxis": {"type": "category", "data": [n["date"] for n in fr["nav"]]}, "yAxis": {"type": "value", "scale": True},
            "series": [{"name": k, "type": "line", "data": [round(n[k], 4) for n in fr["nav"]]} for k in keys]}
    if risk.get("matrix"):
        out["risk_radar"] = {"title": {"text": "六维风险雷达(1低-5高)"}, "tooltip": {},
                             "radar": {"indicator": [{"name": m["dimension"], "max": 5} for m in risk["matrix"]]},
                             "series": [{"type": "radar", "data": [{"value": [m["score"] for m in risk["matrix"]], "name": "风险"}]}]}
    return out


def build_report(ctx: RunContext, debate: dict, risk: dict, architecture: str) -> dict:
    verifier_on = ctx.config.tier1_verifier or ctx.config.tier2_verifier
    keep = ("VERIFIED", "PARTIALLY_SUPPORTED") if verifier_on else ("DRAFT", "VERIFIED", "PARTIALLY_SUPPORTED")
    sec = ctx.svc.resolve_security(ctx.ticker)
    claims = [c for c in ctx.claims.values() if c.status in keep]
    sections = []
    overview = (f"{sec.name}（{sec.ticker}），申万行业 {sec.industry_sw_l1}/{sec.industry_sw_l2}，研究时点 {ctx.as_of}。"
                f"{'【演示公司：数据为虚构的离线 Golden Snapshot】' if sec.is_demo else ''}")
    if ctx.factor_only:
        overview = (f"研究型因子检验（场景D），研究时点 {ctx.as_of}。样本池为 benchmarks/factor_panel 的24家A股，"
                    "财务数据按公告次日可见，收益窗口在研究时点前结束。结果只描述样本期内的统计关系。")
    for key, title in SECTIONS:
        items = [c for c in claims if c.section == key]
        if key == "overview" or items:
            sections.append({"key": key, "title": title, "claim_ids": [c.claim_id for c in items]})
    used_ev = sorted({e for c in claims for e in c.evidence_ids})
    used_metrics = sorted({m for e in used_ev for m in ctx.evidence[e].metric_ids if e in ctx.evidence})
    sys_p = "你是研报撰写Agent。根据已核验论断写不超过150字的执行摘要，不得出现任何数字，不得引入新事实。输出JSON {\"summary\":\"...\"}"
    data = None if ctx.route.fast else call_json(ctx.llm, ctx.tracer, "report", sys_p, json.dumps([c.text for c in claims[:30]], ensure_ascii=False),
                     "strong", validate=lambda d: isinstance(d.get("summary"), str) and not any(ch.isdigit() for ch in d["summary"]))
    exec_summary = data["summary"] if data else debate.get("manager_summary", "") if debate else ""
    rep = {
        "run_id": ctx.run_id, "architecture": architecture, "ticker": ctx.ticker, "name": sec.name,
        "as_of": ctx.as_of.isoformat(), "snapshot_id": ctx.snapshot_id, "question": ctx.request.question,
        "overview": overview, "executive_summary": exec_summary, "sections": sections,
        "claims": {c.claim_id: c.model_dump() for c in ctx.claims.values()},
        "evidence": {e: json.loads(ctx.evidence[e].model_dump_json()) for e in used_ev if e in ctx.evidence},
        "metrics": {m: json.loads(ctx.calc.store[m].model_dump_json()) for m in used_metrics if m in ctx.calc.store},
        "answers": ctx.answers, "open_questions": ctx.open_questions,
        "semantic_fault": ctx.fault_state.get("semantic"),
        "uncertainties": [u.model_dump() for u in ctx.uncertainties], "anomalies": list(ctx.anomalies.values()),
        "conflicts": ctx.svc.conflicts, "debate": debate or {}, "risk": risk or {}, "timeline": ctx.timeline,
        "charts": charts(ctx, risk or {}), "verification_log": ctx.verification_log, "judge_notes": ctx.judge_notes,
        "executive_summary_source": "llm_unverified" if data else "template",
        "plan": ctx.plan.model_dump() if ctx.plan else None,
        "scenario": "factor_test" if ctx.factor_only else "company_research",
        "factor_reports": ctx.factor_reports,
    }
    rep["markdown"] = to_markdown(rep)
    return rep


def to_markdown(rep: dict) -> str:
    title = "研究型因子检验报告" if rep.get("scenario") == "factor_test" else f"{rep['name']}（{rep['ticker']}）投资研究报告"
    lines = [f"# {title}", "",
             f"- 研究时点 as_of: `{rep['as_of']}`  快照: `{rep['snapshot_id']}`  架构: `{rep['architecture']}`", "",
             rep["overview"], ""]
    if rep["executive_summary"]:
        note = "（模型生成的概括，未逐句核验；以下各节论断均已核验）" if rep.get("executive_summary_source") == "llm_unverified" else ""
        lines += ["## 执行摘要", rep["executive_summary"] + note, ""]
    for s in rep["sections"]:
        if s["key"] == "overview":
            continue
        lines.append(f"## {s['title']}")
        for cid in s["claim_ids"]:
            c = rep["claims"][cid]
            refs = " ".join(f"[Evidence #{e}]" for e in c["evidence_ids"])
            tag = {"disclosed_fact": "事实", "derived_metric": "算子指标", "model_inference": "模型推断"}[c["claim_type"]]
            lines.append(f"- ({tag}) {c['text']} {refs}")
        lines.append("")
    if rep["risk"].get("matrix"):
        lines += ["### 风险评分", "| 维度 | 评分(1-5) | 依据 |", "| --- | --- | --- |"]
        lines += [f"| {m['dimension']} | {m['score']} | {m['rationale']} |" for m in rep["risk"]["matrix"]]
        lines.append("")
    lines.append("## 数据局限与不确定性声明")
    for q in rep["open_questions"]:
        lines.append(f"- 待解答：{q['topic']}（{q['reason']}）")
    for u in rep["uncertainties"]:
        lines.append(f"- {u['text']}")
    for c in rep["conflicts"]:
        lines.append(f"- 数据源冲突：{c['metric']} {c['period']} 采用 {c['chosen']}，另一来源 {c['other']} 数值不同")
    if not (rep["open_questions"] or rep["uncertainties"] or rep["conflicts"]):
        lines.append("- 无")
    lines += ["", "## 引用", ""]
    for eid, e in rep["evidence"].items():
        lines.append(f"- **{eid}** [{e['kind']}] {e['summary']} — `{e['source_ref']}` available_at={e['available_at']}")
    return "\n".join(lines) + "\n"
