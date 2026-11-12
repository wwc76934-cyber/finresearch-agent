"""n=3 summary of the v1.2 online comparison: Mean +- Std, Pass@1, Pass^3, per-task stability, cost, latency.

Rescores every stored run with the CURRENT evaluators (no model calls), so all runs share one ruler.
  python scripts/n3_summary.py            writes reports/eval/online_n3_summary.json
Excluded as invalid: B2 r1 (old adapter) and B4 r1 (43 of 75 tasks had failed LLM calls and ran on templates).
"""
import collections
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
sys.stdout.reconfigure(encoding="utf-8")
from data_platform.db import ResearchRunRow, get_session_factory
from evaluation_platform import evaluators
from evaluation_platform.datasets import load_tasks

RUNS = {"B2": ["r2", "r3", "r4"], "B3": ["r1", "r2", "r3"], "B4": ["r2", "r3", "r4"]}   # B2 r1: old adapter; B4 r1: 43/75 tasks ran on template fallback (account out of credit)
tasks = {t["task_id"]: t for t in load_tasks(("golden", "blind", "multidoc"))}
rubric_fn = evaluators.EVALUATORS.pop("rubric")


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


out, per_task = {}, {}
try:
    with get_session_factory()() as s:
        for arch, tags in RUNS.items():
            verdicts = collections.defaultdict(list)
            rates, costs, lats, bycat = [], [], [], collections.defaultdict(list)
            for tag in tags:
                rows = [json.loads(l) for l in (ROOT / "reports" / "eval" / f"online_v12_{arch}_{tag}.partial.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
                ok = 0
                for r in rows:
                    row = s.get(ResearchRunRow, r["run_id"]) if r.get("run_id") else None
                    p = bool(r["passed"])
                    if row is not None and r["task_id"] in tasks:
                        sc = evaluators.evaluate(tasks[r["task_id"]], row.report)
                        if "rubric" in r["scores"]:
                            sc["rubric"] = r["scores"]["rubric"]
                        p = evaluators.task_passed(sc, tasks[r["task_id"]])
                    verdicts[r["task_id"]].append(p)
                    ok += p
                    lats.append(r["latency_ms"] / 1000)
                    bycat[(r["category"], tag)].append(p)
                rates.append(ok / len(rows))
                costs.append(sum(x["scores"].get("cost", {}).get("cost_cny", 0) for x in rows))
            n = len(verdicts)
            allp = sum(all(v) for v in verdicts.values())      # Pass^k: passes in every repeat
            anyp = sum(any(v) for v in verdicts.values())
            flip = sum(1 for v in verdicts.values() if 0 < sum(v) < len(v))
            first = rates[0]
            ks = sum(sum(v) for v in verdicts.values())
            lo, hi = wilson(round(statistics.mean(rates) * n), n)
            cats = sorted({c for c, _ in bycat})
            out[arch] = {"runs": tags, "n_tasks": n, "pass_rate_mean": statistics.mean(rates), "pass_rate_std": statistics.pstdev(rates),
                         "pass_at_1_first_run": first, "pass_all_runs": allp / n, "pass_any_run": anyp / n, "tasks_that_flip": flip,
                         "wilson95": [lo, hi], "cost_cny_per_run": statistics.mean(costs), "latency_p50_s": statistics.median(lats),
                         "latency_p95_s": sorted(lats)[int(.95 * len(lats)) - 1],
                         "by_category_mean": {c: statistics.mean(sum(v) / len(v) for (cc, _), v in bycat.items() if cc == c) for c in cats}}
            per_task[arch] = {t: sum(v) / len(v) for t, v in verdicts.items()}
finally:
    evaluators.EVALUATORS["rubric"] = rubric_fn

print(f"{'arch':4s} {'runs':>4s} {'mean卤std':>13s} {'Wilson95':>11s} {'Pass@1':>7s} {'Pass^k':>7s} {'Pass@any':>8s} {'flip':>4s} {'楼/run':>6s} {'P50 s':>6s} {'P95 s':>6s}")
for a, o in out.items():
    print(f"{a:4s} {len(o['runs']):4d} {o['pass_rate_mean']:6.1%}卤{o['pass_rate_std']:4.1%} [{o['wilson95'][0]:.0%},{o['wilson95'][1]:.0%}] "
          f"{o['pass_at_1_first_run']:7.1%} {o['pass_all_runs']:7.1%} {o['pass_any_run']:8.1%} {o['tasks_that_flip']:4d} {o['cost_cny_per_run']:6.2f} {o['latency_p50_s']:6.0f} {o['latency_p95_s']:6.0f}")
print()
cats = sorted(out["B3"]["by_category_mean"])
print(f"{'category':22s} " + " ".join(f"{a:>6s}" for a in out))
for c in cats:
    print(f"{c:22s} " + " ".join(f"{out[a]['by_category_mean'].get(c, 0):6.0%}" for a in out))
diff = {t: per_task["B3"][t] - per_task["B2"][t] for t in per_task["B3"]}
print("\nB3 - B2 pass-rate gap (mean of runs):", f"{statistics.mean(out['B3']['pass_rate_mean'] - out['B2']['pass_rate_mean'] for _ in [0]):+.1%}")
(ROOT / "reports" / "eval" / "online_n3_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


