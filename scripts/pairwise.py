"""Pairwise-compare two online eval runs on their deep-research reports with an independent judge.

  python scripts/pairwise.py online_B4_v1.partial online_B4_debate_v2_golden.partial
Writes reports/eval/pairwise_<x>_vs_<y>.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from data_platform.db import ResearchRunRow, get_session_factory  # noqa: E402
from evaluation_platform.datasets import load_tasks  # noqa: E402
from evaluation_platform.judges import DIMS, judge_llm, pairwise  # noqa: E402

EVAL = ROOT / "reports" / "eval"


def runs(name: str) -> dict:
    rows = [json.loads(l) for l in (EVAL / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["task_id"]: r["run_id"] for r in rows if r.get("run_id")}


if __name__ == "__main__":
    nx, ny = sys.argv[1], sys.argv[2]
    cats = sys.argv[3].split(",") if len(sys.argv) > 3 else ["deep_research"]
    llm = judge_llm()
    if llm is None:
        sys.exit("judge LLM not configured")
    tasks = {t["task_id"]: t for t in load_tasks() if t["category"] in cats}
    rx, ry = runs(nx), runs(ny)
    results = []
    with get_session_factory()() as s:
        for tid in sorted(set(rx) & set(ry) & set(tasks)):
            a, b = s.get(ResearchRunRow, rx[tid]), s.get(ResearchRunRow, ry[tid])
            if not a or not b:
                continue
            r = pairwise(tasks[tid]["question"], a.report, b.report, llm)
            if r:
                results.append({"task_id": tid, **r})
                print(tid, r["overall"], {d: r[d] for d in DIMS}, flush=True)
    tally = {d: dict(Counter(r[d] for r in results)) for d in DIMS + ["overall"]}
    out = {"x": nx, "y": ny, "judge": llm.models["strong"], "n": len(results), "tally": tally, "results": results}
    (EVAL / f"pairwise_{nx.split('.')[0]}_vs_{ny.split('.')[0]}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"judge": out["judge"], "n": out["n"], "tally": tally}, ensure_ascii=False))
