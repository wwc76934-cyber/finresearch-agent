"""Evaluator Registry: plug-in evaluators keyed by name.

Code judges (numbers, citations, PIT, anomalies, refusal, conflicts, recovery,
trajectory) are deterministic. The rubric judge uses an LLM when one is
configured; offline it uses a transparent structural rubric.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Callable, Optional

EVALUATORS: dict[str, Callable[[dict, dict], Optional[dict]]] = {}
KEPT = ("VERIFIED", "PARTIALLY_SUPPORTED")


def evaluator(name: str):
    def deco(fn):
        EVALUATORS[name] = fn
        return fn
    return deco


def kept_claims(rep: dict) -> list[dict]:
    ids = sorted({c for s in rep["sections"] for c in s["claim_ids"]})
    return [rep["claims"][c] for c in ids]


def fmt_value(value: float, metric: str, unit: Optional[str]) -> str:
    if metric in ("pe_ttm", "pb_lf", "ps_ttm"):
        return f"{value:.2f}倍"
    if metric.endswith(("_yoy", "margin", "_ratio", "ocf_to_net_profit")) or metric in ("gross_margin", "debt_to_assets"):
        return f"{value * 100:.2f}%"
    if unit == "万元":
        return f"{value / 1e4:.2f}万元"
    return f"{value / 1e8:.2f}亿元"


def _metric_matches(rec: dict, exp: dict) -> bool:
    m = exp["metric"]
    if m.startswith("single_quarter:"):
        ok = rec["metric"] == "single_quarter" and rec.get("extra", {}).get("base_metric") == m.split(":")[1]
    else:
        ok = rec["metric"] == m
    return ok and rec["period"] == exp["period"] and rec["statement_scope"] == exp["scope"]


def find_reported(rep: dict, exp: dict) -> Optional[dict]:
    """Return {value,status,text} for an expected quantity, or None when not reported."""
    kept_ids = {c["claim_id"] for c in kept_claims(rep)}
    for a in rep.get("answers", {}).values():
        ask = a["ask"]
        if ask.get("metric") == exp["metric"] and (ask.get("period") in (exp["period"], None) or exp["metric"] == "pe_ttm") \
                and ask.get("scope", "consolidated") == exp["scope"]:
            cid = a.get("claim_id")
            if cid and cid not in kept_ids:
                return {"value": None, "status": "DOWNGRADED", "text": ""}
            return {"value": a.get("value"), "status": a.get("status"),
                    "text": rep["claims"][cid]["text"] if cid else "", "metric_id": a.get("metric_id")}
    has_doc = lambda c: any(rep["evidence"].get(e, {}).get("kind") == "document" for e in c["evidence_ids"])  # noqa: E731
    for c in sorted(kept_claims(rep), key=lambda c: (has_doc(c), c["claim_id"])):  # metric sentences first, stable
        for mid in c["metric_ids"]:
            rec = rep["metrics"].get(mid)
            if rec and _metric_matches(rec, exp):
                return {"value": rec["value"], "status": rec["status"], "text": c["text"], "metric_id": mid}
    return None


def _has_number(text: str) -> bool:
    from agent_runtime.agents.verifier import extract_numbers

    return any(u for _, _, u, _ in extract_numbers(text))


def _states_value(text: str, exp: dict) -> bool:
    """True when the sentence contains a number equal to the oracle value at its displayed precision."""
    from agent_runtime.agents.verifier import UNIT_SCALE, extract_numbers

    for raw, v, unit, dec in extract_numbers(text):
        if unit is None:
            continue
        _, scale = UNIT_SCALE[unit]
        truth = exp["value"]
        if abs(v * scale - truth) <= 0.5 * 10 ** (-dec) * scale * 1.0001 + 1e-9:
            return True
    return False


@evaluator("numerical_accuracy")
def numerical_accuracy(task: dict, rep: dict) -> Optional[dict]:
    nums = task["expected"].get("numbers")
    if not nums:
        return None
    details, ok = [], 0
    for exp in nums:
        got = find_reported(rep, exp)
        if got is None:
            details.append({"key": exp["key"], "result": "missing"})
            continue
        if exp["value"] is None:  # expected NA status
            good = got["value"] is None and got["status"] == exp["status"]
        else:
            v = got["value"]
            good = v is not None and abs(v - exp["value"]) <= abs(exp["value"]) * exp["tol_rel"] + 1e-6
            if good:
                # every kept sentence that cites this quantity must state it correctly
                texts = [got["text"]] if got["text"] else []
                if got.get("metric_id"):
                    texts += [c["text"] for c in kept_claims(rep) if got["metric_id"] in c["metric_ids"]
                              and not any(rep["evidence"].get(e, {}).get("kind") == "document" for e in c["evidence_ids"])]
                # a sentence that states no number (e.g. an LLM debate argument citing the metric) cannot misstate it
                texts = [t for t in texts if _has_number(t)]
                good = all(_states_value(t, exp) for t in texts)
        ok += good
        details.append({"key": exp["key"], "result": "ok" if good else "wrong", "got": got.get("value"),
                        "expected": exp["value"], "status": got.get("status")})
    return {"score": ok / len(nums), "details": details}


@evaluator("citation_validity")
def citation_validity(task: dict, rep: dict) -> Optional[dict]:
    claims = kept_claims(rep)
    if not claims:
        return None
    valid = sum(1 for c in claims if c["evidence_ids"] and all(e in rep["evidence"] for e in c["evidence_ids"]))
    return {"score": valid / len(claims), "n_claims": len(claims)}


NUM = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(%|亿元|万元|天|倍)?")


@evaluator("support_rate")
def support_rate(task: dict, rep: dict) -> Optional[dict]:
    """A claim is supported when every number it states equals a cited MetricRecord (as rendered)
    or appears verbatim in a cited source excerpt."""
    from agent_runtime.agents.verifier import extract_numbers, _match_metric
    from data_platform.schema import MetricRecord

    claims = kept_claims(rep)
    if not claims:
        return None
    sup = 0
    for c in claims:
        evs = [rep["evidence"][e] for e in c["evidence_ids"] if e in rep["evidence"]]
        if not evs:
            continue
        recs = [MetricRecord(**rep["metrics"][m]) for e in evs for m in e["metric_ids"] if m in rep["metrics"]]
        doc = " ".join(e["excerpt"] for e in evs if e["kind"] == "document")
        good = all(raw in doc or raw.replace(" ", "") in doc.replace(",", "") or _match_metric(v, u, d, recs)
                   for raw, v, u, d in extract_numbers(c["text"]))
        sup += good
    return {"score": sup / len(claims)}


@evaluator("pit_violation")
def pit_violation(task: dict, rep: dict) -> dict:
    as_of = datetime.fromisoformat(task["as_of"])
    forb = task["expected"].get("forbidden", {})
    hits = []
    for eid, e in rep["evidence"].items():
        if datetime.fromisoformat(e["available_at"]) > as_of:
            hits.append(f"evidence {eid} available_at {e['available_at']}")
        if any(d in e["source_ref"] or d == e.get("document_id") for d in forb.get("doc_ids", [])):
            hits.append(f"evidence {eid} from future doc {e['source_ref']}")
    text = " ".join(c["text"] for c in kept_claims(rep))
    for kw in forb.get("keywords", []):
        if kw in text:
            hits.append(f"future keyword '{kw}' in report")
    lp = task["expected"].get("latest_period")
    if lp:
        for c in kept_claims(rep):
            for mid in c["metric_ids"]:
                rec = rep["metrics"].get(mid, {})
                if re.match(r"\d{4}Q\d", rec.get("period", "")) and rec["period"] > lp:
                    hits.append(f"metric {mid} period after latest visible {lp}")
    return {"score": 1.0 if hits else 0.0, "violations": hits[:5]}


@evaluator("anomaly_recall")
def anomaly_recall(task: dict, rep: dict) -> Optional[dict]:
    exp = task["expected"].get("anomalies")
    if exp is None or task["category"] != "anomaly_attribution":
        return None
    got = {a["code"] for a in rep.get("anomalies", [])}
    # an anomaly counts only if its claim survived into the report
    kept = {c["claim_id"] for c in kept_claims(rep)}
    got = {a["code"] for a in rep.get("anomalies", []) if a.get("claim_id") in kept}
    recall = len(set(exp) & got) / len(exp) if exp else (1.0 if not got else 0.0)
    attributed = sum(1 for c in kept_claims(rep) if c["produced_by"] == "investigate")
    return {"score": recall, "expected": exp, "got": sorted(got), "false_positive": sorted(got - set(exp)),
            "investigations": attributed}


@evaluator("refusal")
def refusal(task: dict, rep: dict) -> Optional[dict]:
    if not task["expected"].get("must_refuse"):
        return None
    answered = [c for c in kept_claims(rep) if c["section"] == "answer"]
    refused = bool(rep.get("open_questions"))
    return {"score": 1.0 if refused and not answered else 0.0, "open_questions": len(rep.get("open_questions", [])),
            "answer_claims": len(answered)}


@evaluator("conflict_handling")
def conflict_handling(task: dict, rep: dict) -> Optional[dict]:
    if not task["expected"].get("conflict"):
        return None
    return {"score": 1.0 if rep.get("conflicts") else 0.0, "conflicts": len(rep.get("conflicts", []))}


@evaluator("recovery")
def recovery(task: dict, rep: dict) -> Optional[dict]:
    fault = task["expected"].get("fault")
    if not fault:
        return None
    pit = pit_violation(task, rep)["score"]
    na = numerical_accuracy(task, rep)
    clean = pit == 0.0 and (na is None or na["score"] == 1.0 or
                            (task["expected"].get("expect_downgrade") and
                             all(d["result"] == "missing" or d.get("status") == "DOWNGRADED" for d in na["details"])))
    return {"score": 1.0 if clean else 0.0, "fault": fault}


RUBRIC_DIMS = ["基本面深度", "归因逻辑", "反证质量", "证伪条件明确性", "风险覆盖"]


@evaluator("rubric")
def rubric(task: dict, rep: dict) -> Optional[dict]:
    if not task["expected"].get("rubric"):
        return None
    from agent_runtime.llm import get_llm, parse_json

    claims = kept_claims(rep)
    llm = get_llm()
    if llm.online:
        res = llm.chat("你是买方研究总监，按5个维度(各0-1分)盲评研报：" + "、".join(RUBRIC_DIMS) +
                       "。只输出JSON {\"scores\":[5个数],\"comment\":\"...\"}", rep["markdown"][:12000], json_mode=True)
        d = parse_json(res.text)
        if d and len(d.get("scores", [])) == 5:
            return {"score": sum(d["scores"]) / 5, "dims": dict(zip(RUBRIC_DIMS, d["scores"])), "judge": "llm",
                    "comment": d.get("comment", "")}
    by = lambda sec: [c for c in claims if c["section"] == sec]  # noqa: E731
    dims = {
        "基本面深度": min(1.0, len(by("financials")) / 5),
        "归因逻辑": 1.0 if any(c["claim_type"] == "disclosed_fact" and any(rep["evidence"].get(e, {}).get("kind") == "document"
                                                                         for e in c["evidence_ids"]) for c in by("financials") + by("bear")) else 0.0,
        "反证质量": min(1.0, sum(1 for c in by("bear") if c["evidence_ids"]) / 3),
        "证伪条件明确性": min(1.0, len(by("falsification")) / 2),
        "风险覆盖": len(rep.get("risk", {}).get("matrix", [])) / 6,
    }
    return {"score": sum(dims.values()) / 5, "dims": dims, "judge": "structural"}


@evaluator("trajectory")
def trajectory(task: dict, rep: dict) -> dict:
    t = rep["totals"]
    plan = rep.get("plan") or {}
    asks = [a for a in plan.get("asks", []) if a["kind"] == "metric"]
    answered = [a for a in asks if a["ask_id"] in rep.get("answers", {}) and rep["answers"][a["ask_id"]].get("metric_id")]
    replans = sum(1 for s in rep.get("trace", []) if s["node"] == "replan")
    return {"redundant_tool_rate": t["redundant_tool_calls"] / max(t["tool_calls"], 1),
            "first_pass_plan_valid": 1.0 if len(answered) == len(asks) else 0.0, "replans": replans,
            "tool_calls": t["tool_calls"]}


def evaluate(task: dict, rep: dict) -> dict:
    out = {}
    for name, fn in EVALUATORS.items():
        try:
            r = fn(task, rep)
        except Exception as exc:  # noqa: BLE001 - one broken evaluator must not hide the others
            r = {"error": f"{type(exc).__name__}: {exc}"}
        if r is not None:
            out[name] = r
    out["cost"] = {"latency_ms": rep["totals"]["latency_ms"],
                   "tokens": rep["totals"]["input_tokens"] + rep["totals"]["output_tokens"],
                   "cost_cny": rep["totals"]["cost_cny"]}
    return out


def task_passed(scores: dict, task: Optional[dict] = None) -> bool:
    checks = []
    keys = ["numerical_accuracy", "anomaly_recall", "refusal", "conflict_handling", "recovery"]
    if task and task["expected"].get("expect_downgrade"):
        keys.remove("numerical_accuracy")  # the right outcome is an uncertainty statement, not a number
    for k in keys:
        if k in scores and "score" in scores[k]:
            checks.append(scores[k]["score"] == 1.0)
    checks.append(scores.get("pit_violation", {}).get("score", 0) == 0.0)
    if "rubric" in scores:
        checks.append(scores["rubric"]["score"] >= 0.8)
    return all(checks)
