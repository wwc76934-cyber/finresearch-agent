"""Run the 80-task benchmark with the real LLM on one architecture (default B4).

  python scripts/run_online_eval.py --arch B4 --workers 2
Results: reports/eval/online_<arch>_<time>.json and ..._summary.json (does not touch the offline latest.json).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from agent_runtime.llm import get_llm  # noqa: E402
from evaluation_platform.comparison import aggregate  # noqa: E402
from evaluation_platform.runner import OUT, run_experiment  # noqa: E402

if __name__ == "__main__":
    import faulthandler

    faulthandler.dump_traceback_later(300, repeat=True)  # dump all thread stacks to stderr every 5 min (hang diagnosis)
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="B4")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N tasks (0 = all)")
    ap.add_argument("--name", default="", help="reuse a name to resume an interrupted run")
    a = ap.parse_args()
    if not get_llm().online:
        sys.exit("LLM not configured: fill LLM_API_KEY in .env")
    name = a.name or f"online_{a.arch}_{datetime.now():%Y%m%d_%H%M%S}"
    seen = []
    flt = (lambda t: len(seen) < a.limit and not seen.append(1)) if a.limit else None
    p = run_experiment([a.arch], workers=a.workers, persist_archs=(a.arch,), name=name, task_filter=flt,
                       write_latest=False, resume_file=OUT / f"{name}.partial.jsonl")
    agg = aggregate(p["results"])[a.arch]
    total_cost = sum(r["scores"].get("cost", {}).get("cost_cny", 0) for r in p["results"])
    (OUT / f"{name}_summary.json").write_text(json.dumps({"aggregate": agg, "total_cost_cny": total_cost},
                                                         ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"pass_rate": agg["pass_rate"], "by_category": agg["pass_by_category"], "errors": agg["errors"],
                      "p50_ms": agg["latency_p50_ms"], "total_cost_cny": round(total_cost, 3)}, ensure_ascii=False))
    for r in p["results"]:
        if not r["passed"]:
            print("FAIL", r["task_id"], r.get("category"), r["error_category"],
                  json.dumps({k: v for k, v in r["scores"].items() if k in ("numerical_accuracy", "refusal", "anomaly_recall",
                                                                            "pit_violation", "recovery", "crash")},
                             ensure_ascii=False, default=str)[:400])
