"""Error Analysis Engine: map a failed task to one of six root-cause buckets."""
from __future__ import annotations

from typing import Optional

BUCKETS = ["DATA_MISSING", "RETRIEVAL_MISS", "SCOPE_CONFUSION", "TOOL_ERROR", "HALLUCINATION", "PIT_VIOLATION"]


def classify(task: dict, rep: dict, scores: dict) -> Optional[str]:
    from .evaluators import task_passed

    if task_passed(scores, task):
        return None
    if scores.get("pit_violation", {}).get("score"):
        return "PIT_VIOLATION"
    tool_errors = [t for s in rep.get("trace", []) for t in s.get("tool_calls", []) if t.get("error")]
    na = scores.get("numerical_accuracy")
    if na:
        for d in na["details"]:
            if d["result"] == "missing":
                return "TOOL_ERROR" if tool_errors and task["expected"].get("fault") != "tool_error" else "DATA_MISSING"
            if d["result"] == "wrong" and d.get("got") is not None and d.get("expected"):
                r = d["got"] / d["expected"] if d["expected"] else 0
                if any(abs(r - k) < 1e-3 * k for k in (1e4, 1e-4, 1e8, 0.38, 1 / 0.38, 0.30)):
                    return "SCOPE_CONFUSION"
    if "anomaly_recall" in scores and scores["anomaly_recall"]["score"] < 1:
        return "RETRIEVAL_MISS"
    if "rubric" in scores and scores["rubric"]["score"] < 0.8:
        return "RETRIEVAL_MISS" if scores["rubric"]["dims"].get("归因逻辑", 1) < 1 else "HALLUCINATION"
    if tool_errors and "recovery" in scores and scores["recovery"]["score"] < 1:
        return "TOOL_ERROR"
    return "HALLUCINATION"


def funnel(results: list[dict]) -> dict:
    out = {b: 0 for b in BUCKETS}
    for r in results:
        if r.get("error_category"):
            out[r["error_category"]] += 1
    return out
