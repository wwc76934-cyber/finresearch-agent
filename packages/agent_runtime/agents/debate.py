"""Evidence-bound Bull/Bear debate with a Review Manager that outputs falsification criteria.

Rules: every argument cites >=1 evidence id; no new evidence -> no further round;
Bear may make at most 2 targeted counter-evidence calls per round.

Online (real LLM): Bull and Bear write their own arguments, Bear chooses its
own counter-evidence queries, Bull rebuts, and the Manager pairs the core
disagreements. Argument text may not contain digits: numbers live only in the
cited MetricRecords, so the LLM can reason about them but never restate them.
Offline: deterministic templates over the same evidence.
"""
from __future__ import annotations

import json
import re

from .. import tools
from ..context import RunContext, fmt
from ..llm import call_json

DETAIL = re.compile(r"[^。\n|]*(合计|风险|集中|主要系|所致)[^。\n|]*。")
DIGIT = re.compile(r"[0-9０-９]|百分之")
SECTIONS = {"NOTES", "MDA", "RISK_FACTORS", "ANNOUNCEMENT", "STATEMENTS"}
MAX_ARGS = 4


def _latest_label(ctx: RunContext) -> str:
    p = tools.latest_period(ctx)
    return f"{p.year}Q{(p.month - 1) // 3 + 1}" if p else ""


# ============================================================ offline templates
def _signals(ctx: RunContext) -> tuple[list, list]:
    """Bull/bear signals from already-computed MetricRecords (no new numbers)."""
    st = ctx.calc.store
    pick = lambda name: next((r for r in st.values() if r.metric == name and r.ticker == ctx.ticker and r.value is not None  # noqa: E731
                              and r.period == _latest_label(ctx)), None)
    bull, bear = [], []
    ry, gm, roe, ocf = pick("revenue_yoy"), pick("gross_margin"), pick("roe_annualized"), pick("ocf_to_net_profit")
    if ry and ry.value > 0.1:
        bull.append((ry, lambda r=ry: f"收入端保持增长：营收同比{fmt(r)}。"))
    if ry and ry.value < 0:
        bear.append((ry, lambda r=ry: f"收入端承压：营收同比{fmt(r)}。"))
    if roe and roe.value > 0.12:
        bull.append((roe, lambda r=roe: f"资本回报较高：年化ROE为{fmt(r)}。"))
    if roe and roe.value < 0:
        bear.append((roe, lambda r=roe: f"公司处于亏损状态：年化ROE为{fmt(r)}。"))
    if gm and gm.value > 0.3:
        bull.append((gm, lambda r=gm: f"盈利能力具备护城河特征：毛利率{fmt(r)}。"))
    if ocf and ocf.value > 1.0:
        bull.append((ocf, lambda r=ocf: f"利润现金含量充足：经营现金流/净利润{fmt(r)}。"))
    return bull, bear


def _template_rounds(ctx: RunContext, log: dict, bull_ids: list, bear_ids: list) -> tuple[list, list]:
    bull_sig, bear_sig = _signals(ctx)
    for side, sig, ids in (("bull", bull_sig, bull_ids), ("bear", bear_sig, bear_ids)):
        for rec, render in sig:
            ev = ctx.evidence_from_metric(rec, side)
            cl = ctx.add_claim(side, "derived_metric", render(), [ev], side)
            ctx.renderers[cl.claim_id] = render
            ids.append(cl.claim_id)
    for rnd in range(1, ctx.config.max_debate_rounds + 1):
        new_ev, new_claims = [], []
        if rnd == 1:
            queries = [("前五名 欠款方 合计 占应收账款 客户集中度", ["NOTES"]), ("风险因素 竞争 原材料 需求", ["RISK_FACTORS"])]
            for q, secs in queries[:2]:  # at most 2 targeted counter-evidence calls
                for h in tools.retrieve_latest_first(ctx, q, secs, top_k=2)[:1]:
                    m = DETAIL.search((h.parent or h.chunk).text)
                    if not m:
                        continue
                    ev = ctx.evidence_from_hit(h, "bear")
                    if ev.evidence_id in {e for c in bear_ids for e in ctx.claims[c].evidence_ids}:
                        continue
                    cl = ctx.add_claim("bear", "disclosed_fact", f"反证：{m.group(0)}", [ev], "bear")
                    bear_ids.append(cl.claim_id)
                    new_claims.append(cl.claim_id)
                    new_ev.append(ev.evidence_id)
        else:
            for e in tools.query_evidence_store(ctx, "合同"):
                if any(e.evidence_id in ctx.claims[c].evidence_ids for c in bull_ids):
                    continue
                cl = ctx.add_claim("bull", "disclosed_fact", f"回应：{e.summary}，订单储备为收入提供支撑。", [e], "bull")
                bull_ids.append(cl.claim_id)
                new_claims.append(cl.claim_id)
        log["rounds"].append({"round": rnd, "bull": list(bull_ids), "bear": list(bear_ids),
                              "new_claims": new_claims, "new_evidence": new_ev})
        if not new_claims:
            break  # 无新证据不辩论
    for b, r in zip(bull_ids, bear_ids):
        log["disagreements"].append({"bull": b, "bear": r, "issue": f"{ctx.claims[b].text[:40]} vs {ctx.claims[r].text[:40]}"})
    return bull_ids, bear_ids


# ============================================================ LLM debate
def _catalog(ctx: RunContext, limit: int = 40) -> list[dict]:
    out = []
    for e in list(ctx.evidence.values())[:limit]:
        if e.available_at > ctx.as_of:
            continue  # never show the debaters anything a PIT check would reject
        item = {"id": e.evidence_id, "kind": e.kind, "summary": e.summary}
        if e.kind == "document":
            item["excerpt"] = re.sub(r"^\[[^\]]*\]\n", "", e.excerpt)[:300]
        out.append(item)
    return out


def _claims_view(ctx: RunContext, ids: list[str]) -> list[dict]:
    return [{"claim_id": c, "text": ctx.claims[c].text, "evidence_ids": ctx.claims[c].evidence_ids} for c in ids]


def _accept_args(ctx: RunContext, items, side: str, prefix: str, log: dict) -> list[str]:
    """Validate LLM arguments: >=1 existing PIT-safe evidence id, no digits. Rejected ones are logged."""
    ids = []
    for a in (items or [])[:MAX_ARGS]:
        if not isinstance(a, dict):
            continue
        text = str(a.get("thesis") or a.get("text") or "").strip()
        evs = [e for e in a.get("evidence_ids", []) if isinstance(e, str) and e in ctx.evidence
               and ctx.evidence[e].available_at <= ctx.as_of]
        if not text or not evs:
            log["filtered"].append({"side": side, "text": text[:80], "reason": "no valid evidence id"})
            continue
        if DIGIT.search(text):
            log["filtered"].append({"side": side, "text": text[:80], "reason": "contains digits"})
            continue
        text = re.sub(r"[（(]见证据[)）]", "", text)  # badges already show the evidence
        assumption = str(a.get("assumption", "")).strip()
        if assumption and not DIGIT.search(assumption):
            text = f"{text}（隐含假设：{assumption}）"
        cl = ctx.add_claim(side, "model_inference", f"{prefix}{text}", [ctx.evidence[e] for e in evs], side)
        if a.get("target") in ctx.claims:
            log.setdefault("targets", {})[cl.claim_id] = a["target"]
        ids.append(cl.claim_id)
    return ids


RULES = ("规则：每条论点必须引用至少一个证据 id（只能用给定列表中的 id）；论点文字中不得出现任何阿拉伯数字或百分比，"
         "需要用到数值时只在 evidence_ids 中引用证据，例如“经营现金流显著弱于利润”；用中文；每条不超过60字；最多4条。")


def _llm_rounds(ctx: RunContext, log: dict, bull_ids: list, bear_ids: list) -> tuple[list, list]:
    sec = ctx.svc.resolve_security(ctx.ticker)
    head = {"company": f"{sec.name}（{sec.ticker}）", "as_of": ctx.as_of.isoformat(), "question": ctx.request.question}
    # ---- round 1: Bull builds the strongest long case
    d = call_json(ctx.llm, ctx.tracer, "bull", "你是看多研究员，基于证据构建最强的看多逻辑，并写出每条逻辑依赖的隐含假设。" + RULES +
                  '输出JSON {"arguments":[{"thesis":"...","assumption":"...","evidence_ids":["E001"]}]}',
                  json.dumps({**head, "evidence": _catalog(ctx), "existing_bull_points": _claims_view(ctx, bull_ids)},
                             ensure_ascii=False), "strong", validate=lambda x: isinstance(x.get("arguments"), list))
    new_bull = _accept_args(ctx, (d or {}).get("arguments"), "bull", "看多：", log)
    bull_ids += new_bull
    # ---- round 1: Bear attacks the bull case and asks for targeted counter-evidence (<= 2 queries)
    d = call_json(ctx.llm, ctx.tracer, "bear", "你是看空研究员。逐条审阅看多论点，找出其隐含假设的漏洞，并提出最多2个定向检索请求以寻找反例"
                  "（例如客户集中度、应收账款账龄、减值计提、关联交易、债务融资）。只输出检索请求，不写论点。"
                  '输出JSON {"queries":[{"query":"中文关键词","section":"NOTES|MDA|RISK_FACTORS|ANNOUNCEMENT","target":"被质疑的看多 claim_id"}]}',
                  json.dumps({**head, "bull_points": _claims_view(ctx, bull_ids)}, ensure_ascii=False), "strong",
                  validate=lambda x: isinstance(x.get("queries"), list))
    new_ev = []
    for q in ((d or {}).get("queries") or [])[:2]:
        if not isinstance(q, dict) or not str(q.get("query", "")).strip():
            continue
        secs = [q["section"]] if q.get("section") in SECTIONS else None
        for h in tools.retrieve_latest_first(ctx, str(q["query"])[:60], secs, top_k=2)[:2]:
            ev = ctx.evidence_from_hit(h, "bear")
            if ev.evidence_id not in new_ev:
                new_ev.append(ev.evidence_id)
    d = call_json(ctx.llm, ctx.tracer, "bear", "你是看空研究员。针对看多论点给出反驳与看空逻辑，优先使用新检索到的反证；"
                  "每条论点用 target 指明反驳的看多 claim_id（独立看空逻辑可留空）。" + RULES +
                  '输出JSON {"arguments":[{"thesis":"...","assumption":"...","target":"C0xx","evidence_ids":["E0xx"]}]}',
                  json.dumps({**head, "bull_points": _claims_view(ctx, bull_ids), "existing_bear_points": _claims_view(ctx, bear_ids),
                              "new_counter_evidence": [x for x in _catalog(ctx, 80) if x["id"] in new_ev],
                              "evidence": _catalog(ctx)}, ensure_ascii=False), "strong",
                  validate=lambda x: isinstance(x.get("arguments"), list))
    new_bear = _accept_args(ctx, (d or {}).get("arguments"), "bear", "看空：", log)
    bear_ids += new_bear
    log["rounds"].append({"round": 1, "bull": list(bull_ids), "bear": list(bear_ids), "new_claims": new_bull + new_bear,
                          "new_evidence": new_ev})
    # ---- round 2: Bull rebuts, only if Bear brought new evidence (无新证据不辩论)
    if ctx.config.max_debate_rounds >= 2 and new_bear and new_ev:
        d = call_json(ctx.llm, ctx.tracer, "bull", "你是看多研究员。逐条回应看空论点：能用证据反驳的就反驳（target 填被反驳的看空 claim_id），"
                      "无法反驳的请放入 concessions 并说明。" + RULES +
                      '输出JSON {"arguments":[{"thesis":"...","target":"C0xx","evidence_ids":["E0xx"]}],"concessions":["..."]}',
                      json.dumps({**head, "bear_points": _claims_view(ctx, new_bear), "evidence": _catalog(ctx)}, ensure_ascii=False),
                      "strong", validate=lambda x: isinstance(x.get("arguments"), list))
        rebut = _accept_args(ctx, (d or {}).get("arguments"), "bull", "回应：", log)
        bull_ids += rebut
        log["concessions"] = [str(c)[:100] for c in (d or {}).get("concessions", []) if not DIGIT.search(str(c))]
        log["rounds"].append({"round": 2, "bull": list(bull_ids), "bear": list(bear_ids), "new_claims": rebut, "new_evidence": []})
    # ---- Manager: pair the core disagreements (no digits)
    d = call_json(ctx.llm, ctx.tracer, "manager", "你是 Review Manager。从多空论点中提炼最多3个核心分歧，每个分歧指明一条看多和一条看空 claim_id，"
                  "并用一句话说明分歧焦点；再用不超过100字概括。全部文字不得出现数字。"
                  '输出JSON {"disagreements":[{"bull":"C0xx","bear":"C0xx","issue":"..."}],"summary":"..."}',
                  json.dumps({"bull": _claims_view(ctx, bull_ids), "bear": _claims_view(ctx, bear_ids)}, ensure_ascii=False),
                  "strong", validate=lambda x: isinstance(x.get("disagreements"), list))
    for x in ((d or {}).get("disagreements") or [])[:3]:
        if isinstance(x, dict) and x.get("bull") in bull_ids and x.get("bear") in bear_ids and not DIGIT.search(str(x.get("issue", ""))):
            log["disagreements"].append({"bull": x["bull"], "bear": x["bear"], "issue": str(x.get("issue", ""))[:120]})
    s = str((d or {}).get("summary", ""))
    if s and not DIGIT.search(s):
        log["manager_summary"] = s
    if not log["disagreements"]:
        for b, r in zip(bull_ids, bear_ids):
            log["disagreements"].append({"bull": b, "bear": r, "issue": "多空观点对立"})
    return bull_ids, bear_ids


# ============================================================ falsification + entry
def _falsification(ctx: RunContext, log: dict) -> None:
    """Thresholds are deterministic MetricRecords, so every criterion is checkable later."""
    st = ctx.calc.store
    label = _latest_label(ctx)
    get = lambda name: next((x for x in st.values() if x.metric == name and x.period == label and x.value is not None), None)  # noqa: E731
    rules = [("gross_margin", -0.02, "若后续报告期毛利率跌破{t}（当前{c}），则“盈利能力稳定”的看多假设证伪。"),
             ("ocf_to_net_profit", None, "若经营现金流/净利润持续低于{t}（当前{c}），则“利润质量可靠”的假设证伪。"),
             ("receivable_days", 1.1, "若应收账款周转天数升至{t}以上（当前{c}），说明回款恶化，看多逻辑需下修。"),
             ("interest_debt_ratio", 0.05, "若有息负债率升至{t}以上（当前{c}），偿债风险假设需上调。")]
    for metric, adj, tmpl in rules:
        cur = get(metric)
        if not cur:
            continue
        tv = 0.8 if metric == "ocf_to_net_profit" else (cur.value * adj if metric == "receivable_days" else cur.value + adj)
        th = ctx.calc._record(ctx.ticker, f"threshold_{metric}", label, tv, cur.unit, [cur], extra={"rule": tmpl.split("，")[0]})
        render = lambda tmpl=tmpl, th=th, cur=cur: tmpl.format(t=fmt(th), c=fmt(cur))
        ev_c, ev_t = ctx.evidence_from_metric(cur, "manager"), ctx.evidence_from_metric(th, "manager")
        cl = ctx.add_claim("falsification", "model_inference", render(), [ev_c, ev_t], "manager")
        ctx.renderers[cl.claim_id] = render
        log["falsification"].append(cl.claim_id)


def run_debate(ctx: RunContext) -> dict:
    log = {"rounds": [], "disagreements": [], "falsification": [], "filtered": [], "manager_summary": "",
           "mode": "llm" if ctx.llm.online else "template"}
    bull_ids = [c.claim_id for c in ctx.claims.values() if c.section == "bull" and c.evidence_ids]
    bear_ids = [c.claim_id for c in ctx.claims.values() if c.section == "bear" and c.evidence_ids]
    if ctx.llm.online:
        bull_ids, bear_ids = _llm_rounds(ctx, log, bull_ids, bear_ids)
    else:
        bull_ids, bear_ids = _template_rounds(ctx, log, bull_ids, bear_ids)
    _falsification(ctx, log)
    if not log["manager_summary"]:
        sys_p = "你是 Review Manager。用不超过100字概括多空核心分歧，不得出现任何数字。输出JSON {\"summary\":\"...\"}"
        user = json.dumps({"bull": [ctx.claims[c].text for c in bull_ids], "bear": [ctx.claims[c].text for c in bear_ids]},
                          ensure_ascii=False)
        data = call_json(ctx.llm, ctx.tracer, "manager", sys_p, user, "strong",
                         validate=lambda d: isinstance(d.get("summary"), str) and not DIGIT.search(d["summary"]))
        log["manager_summary"] = data["summary"] if data else (
            f"多方论据{len(bull_ids)}条、空方论据{len(bear_ids)}条；核心分歧在于盈利增长的质量与可持续性。")
    return log
