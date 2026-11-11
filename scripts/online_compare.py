"""Summarise online runs: pass rate per architecture and per task type, cost, latency, with Wilson 95% intervals.

  python scripts/online_compare.py r1            # reads reports/eval/online_v12_<arch>_r1.partial.jsonl
  python scripts/online_compare.py r1 r2 r3      # several repeats: mean +- std of the pass rate
"""
import collections
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
tags = sys.argv[1:] or ["r1"]
ARCHS = ["B0", "B1", "B2", "B3", "B4"]


def load(arch, tag, prefix="online_v12"):
    p = ROOT / "reports" / "eval" / f"{prefix}_{arch}_{tag}.partial.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else None


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


print(f"{'arch':5s} {'n':>3s} {'pass':>9s} {'95% CI':>13s} {'cost':>8s} {'P50 s':>6s} {'P95 s':>6s}  by category")
for a in ARCHS:
    runs = [r for t in tags if (r := load(a, t))]
    if not runs:
        continue
    rates = [sum(x["passed"] for x in r) / len(r) for r in runs]
    rows = runs[0]
    k, n = sum(x["passed"] for x in rows), len(rows)
    lo, hi = wilson(k, n)
    cat = collections.defaultdict(lambda: [0, 0])
    for x in rows:
        cat[x["category"]][0] += x["passed"]; cat[x["category"]][1] += 1
    cost = sum(x["scores"].get("cost", {}).get("cost_cny", 0) for x in rows)
    lat = sorted(x["latency_ms"] / 1000 for x in rows)
    ms = f"{k}/{n} {k / n:.0%}" + (f" (mean {statistics.mean(rates):.1%} sd {statistics.pstdev(rates):.1%} over {len(runs)} runs)" if len(runs) > 1 else "")
    print(f"{a:5s} {n:3d} {ms:>9s} [{lo:.0%},{hi:.0%}] {cost:7.2f}¥ {statistics.median(lat):6.0f} {lat[int(.95 * len(lat)) - 1]:6.0f}  "
          + " ".join(f"{c.replace('_', '')[:9]}={v[0]}/{v[1]}" for c, v in sorted(cat.items())))
