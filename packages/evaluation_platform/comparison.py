"""Experiment Comparison: per-architecture aggregates, deltas vs B4, paired bootstrap CIs."""
from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean
from typing import Optional

import numpy as np

from .error_analysis import funnel

METRICS = ["numerical_accuracy", "citation_validity", "support_rate", "pit_violation", "anomaly_recall", "refusal",
           "conflict_handling", "recovery", "rubric"]


def _score(r: dict, m: str) -> Optional[float]:
    s = r["scores"].get(m)
    return s.get("score") if isinstance(s, dict) and "score" in s else None


def aggregate(results: list[dict]) -> dict:
    by = defaultdict(list)
    for r in results:
        by[r["architecture"]].append(r)
    out = {}
    for arch, rs in by.items():
        row = {"n": len(rs), "pass_rate": mean(r["passed"] for r in rs)}
        for m in METRICS:
            vals = [v for v in (_score(r, m) for r in rs) if v is not None]
            row[m] = mean(vals) if vals else None
            row[f"{m}_n"] = len(vals)
        tr = [r["scores"].get("trajectory", {}) for r in rs if "trajectory" in r["scores"]]
        row["redundant_tool_rate"] = mean(t["redundant_tool_rate"] for t in tr) if tr else None
        row["first_pass_plan_valid"] = mean(t["first_pass_plan_valid"] for t in tr) if tr else None
        lat = [r["latency_ms"] for r in rs]
        row["latency_p50_ms"] = float(np.percentile(lat, 50))
        row["latency_p95_ms"] = float(np.percentile(lat, 95))
        tok = [r["scores"].get("cost", {}).get("tokens", 0) for r in rs]
        row["tokens_mean"] = mean(tok)
        row["cost_cny_mean"] = mean(r["scores"].get("cost", {}).get("cost_cny", 0) for r in rs)
        row["errors"] = funnel(rs)
        by_cat = defaultdict(list)
        for r in rs:
            by_cat[r.get("category", "?")].append(r["passed"])
        row["pass_by_category"] = {k: mean(v) for k, v in by_cat.items()}
        out[arch] = row
    return out


def paired_bootstrap(results: list[dict], a: str, b: str, n: int = 2000, seed: int = 7) -> dict:
    """CI of pass-rate difference a - b over the same tasks."""
    pa = {r["task_id"]: r["passed"] for r in results if r["architecture"] == a}
    pb = {r["task_id"]: r["passed"] for r in results if r["architecture"] == b}
    ids = sorted(set(pa) & set(pb))
    if not ids:
        return {}
    diffs = [pa[i] - pb[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(mean(rng.choice(diffs) for _ in diffs) for _ in range(n))
    return {"delta": mean(diffs), "ci95": [boots[int(0.025 * n)], boots[int(0.975 * n)]], "n_tasks": len(ids)}


def compare(results: list[dict], reference: str = "B4") -> dict:
    agg = aggregate(results)
    ref = agg.get(reference, {})
    deltas = {}
    for arch, row in agg.items():
        if arch == reference:
            continue
        d = {m: (ref[m] - row[m]) if ref.get(m) is not None and row.get(m) is not None else None for m in METRICS + ["pass_rate"]}
        d["token_ratio"] = ref["tokens_mean"] / row["tokens_mean"] if row["tokens_mean"] else None
        d["latency_ratio_p50"] = ref["latency_p50_ms"] / row["latency_p50_ms"] if row["latency_p50_ms"] else None
        d["bootstrap_pass_rate"] = paired_bootstrap(results, reference, arch)
        deltas[arch] = d
    return {"aggregate": agg, "delta_vs_reference": deltas, "reference": reference}
