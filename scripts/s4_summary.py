"""Step 4 summary: B4 after the ComplexityRouter vs the earlier B4 (r2-r4), same 75 tasks.

  python scripts/s4_summary.py  ->  reports/eval/online_s4_summary.json
Refuses runs that contain failed LLM calls (those ran partly on templates).
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

E = Path(__file__).resolve().parents[1] / "reports" / "eval"
NEW = [f"online_v12s4_B4_r{k}" for k in (1, 2, 3)]
OLD = [f"online_v12_B4_r{k}" for k in (2, 3, 4)]


def load(n):
    p = E / f"{n}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["results"] if "results" in d else d


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(names):
    runs = []
    for n in names:
        rs = load(n)
        if rs is None:
            continue
        bad = sum(1 for r in rs if r.get("llm_calls_failed", 0) > 0)
        if bad:
            print("SKIP", n, "failed-LLM rows:", bad)
            continue
        runs.append((n, rs))
    out = {"runs": [n for n, _ in runs]}
    if not runs:
        return out
    pr = [sum(r["passed"] for r in rs) / len(rs) for _, rs in runs]
    allr = [r for _, rs in runs for r in rs]
    lat = [r["latency_ms"] / 1000 for r in allr]
    out.update({"n_runs": len(runs), "pass_mean": round(st.mean(pr), 4), "pass_std": round(st.pstdev(pr), 4),
                "p50_s": round(st.median(lat), 1), "p90_s": round(pct(lat, 0.9), 1),
                "cost_per_run_cny": round(sum(r["scores"]["cost"]["cost_cny"] for r in allr) / len(runs), 2)})
    s60 = [r["latency_ms"] / 1000 for r in allr if r["dataset"] in ("golden", "blind")]
    out["p50_s_golden_blind_60"] = round(st.median(s60), 1)
    out["cost_per_run_golden_blind_60_cny"] = round(sum(r["scores"]["cost"]["cost_cny"] for r in allr
                                                         if r["dataset"] in ("golden", "blind")) / len(runs), 2)
    by = defaultdict(list)
    for r in allr:
        by[f'{r["dataset"]}/{r["category"]}'].append(r)
    out["by_category"] = {k: {"n": len(v) // len(runs), "pass": round(sum(x["passed"] for x in v) / len(v), 3),
                              "p50_s": round(st.median(x["latency_ms"] / 1000 for x in v), 1),
                              "cost_per_run": round(sum(x["scores"]["cost"]["cost_cny"] for x in v) / len(runs), 3)}
                      for k, v in sorted(by.items())}
    return out


if __name__ == "__main__":
    res = {"after": summarize(NEW), "before": summarize(OLD)}
    (E / "online_s4_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
