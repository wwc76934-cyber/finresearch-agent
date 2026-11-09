"""Evidence-bound Bull/Bear debate with a Review Manager that outputs falsification criteria.

Rules: every argument cites >=1 evidence id; no new evidence -> no further round;
Bear may make at most 2 targeted counter-evidence calls per round.
"""
from __future__ import annotations

import json
import re

from .. import tools
from ..context import RunContext, fmt
from ..llm import call_json

DETAIL = re.compile(r"[^。\n|]*(合计|风险|集中|主要系|所致)[^。\n|]*。")


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


def _latest_label(ctx: RunContext) -> str:
    p = tools.latest_period(ctx)
    return f"{p.year}Q{(p.month - 1) // 3 + 1}" if p else ""


def run_debate(ctx: RunContext) -> dict:
    cfg = ctx.config
    log = {"rounds": [], "disagreements": [], "falsification": [], "filtered": [], "manager_summary": ""}
    bull_ids = [c.claim_id for c in ctx.claims.values() if c.section == "bull"]
    bear_ids = [c.claim_id for c in ctx.claims.values() if c.section == "bear"]
    bull_sig, bear_sig = _signals(ctx)
    for rec, render in bull_sig:
        ev = ctx.evidence_from_metric(rec, "bull")
        cl = ctx.add_claim("bull", "derived_metric", render(), [ev], "bull")
        ctx.renderers[cl.claim_id] = render
        bull_ids.append(cl.claim_id)
    for rec, render in bear_sig:
        ev = ctx.evidence_from_metric(rec, "bear")
        cl = ctx.add_claim("bear", "derived_metric", render(), [ev], "bear")
        ctx.renderers[cl.claim_id] = render
        bear_ids.append(cl.claim_id)
    # filter evidence-free arguments
    for cid in list(bull_ids + bear_ids):
        if not ctx.claims[cid].evidence_ids:
            log["filtered"].append(cid)
            ctx.claims[cid].status = "REJECTED"
    bull_ids = [c for c in bull_ids if c not in log["filtered"]]
    bear_ids = [c for c in bear_ids if c not in log["filtered"]]

    for rnd in range(1, cfg.max_debate_rounds + 1):
        new_ev, new_claims = [], []
        if rnd == 1:
            queries = [("前五名 欠款方 合计 占应收账款 客户集中度", ["NOTES"]), ("风险因素 竞争 原材料 需求", ["RISK_FACTORS"])]
            for q, secs in queries[:2]:  # at most 2 targeted counter-evidence calls
                hits = tools.retrieve_latest_first(ctx, q, secs, top_k=2)
                for h in hits[:1]:
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
            # Bull rebuttal only with evidence already in the store (e.g. signed contracts)
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

    # ---- falsification criteria: thresholds are deterministic MetricRecords ----
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
        if metric == "ocf_to_net_profit":
            tv = 0.8
        elif metric == "receivable_days":
            tv = cur.value * adj
        else:
            tv = cur.value + adj
        th = ctx.calc._record(ctx.ticker, f"threshold_{metric}", label, tv, cur.unit, [cur],
                              extra={"rule": tmpl.split("，")[0]})
        render = lambda tmpl=tmpl, th=th, cur=cur: tmpl.format(t=fmt(th), c=fmt(cur))
        ev_c, ev_t = ctx.evidence_from_metric(cur, "manager"), ctx.evidence_from_metric(th, "manager")
        cl = ctx.add_claim("falsification", "model_inference", render(), [ev_c, ev_t], "manager")
        ctx.renderers[cl.claim_id] = render
        log["falsification"].append(cl.claim_id)

    sys_p = "你是 Review Manager。用不超过100字概括多空核心分歧，不得出现任何数字。输出JSON {\"summary\":\"...\"}"
    user = json.dumps({"bull": [ctx.claims[c].text for c in bull_ids], "bear": [ctx.claims[c].text for c in bear_ids]},
                      ensure_ascii=False)
    data = call_json(ctx.llm, ctx.tracer, "manager", sys_p, user, "strong",
                     validate=lambda d: isinstance(d.get("summary"), str) and not re.search(r"\d", d["summary"]))
    log["manager_summary"] = data["summary"] if data else (
        f"多方论据{len(bull_ids)}条、空方论据{len(bear_ids)}条；核心分歧在于盈利增长的质量与可持续性。")
    return log
