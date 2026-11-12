"""Re-score stored online runs with the CURRENT evaluators, no model calls.

  python scripts/rescore_all.py online_v12_B2_r2 online_v12_B3_r1 ...
Prints old vs new pass counts and every task whose verdict flipped.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
sys.stdout.reconfigure(encoding="utf-8")
from data_platform.db import ResearchRunRow, get_session_factory
from evaluation_platform import evaluators
from evaluation_platform.datasets import load_tasks

tasks = {t["task_id"]: t for t in load_tasks(("golden", "blind", "chaos", "lab", "quant", "multidoc"))}
rubric_fn = evaluators.EVALUATORS.pop("rubric")      # the stored LLM rubric is kept; re-judging costs money
try:
    with get_session_factory()() as s:
        for name in sys.argv[1:]:
            rows = [json.loads(l) for l in (ROOT / "reports" / "eval" / f"{name}.partial.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            old = new = 0
            flips = []
            for r in rows:
                row = s.get(ResearchRunRow, r["run_id"]) if r.get("run_id") else None
                if row is None or r["task_id"] not in tasks:
                    old += r["passed"]; new += r["passed"]
                    continue
                sc = evaluators.evaluate(tasks[r["task_id"]], row.report)
                if "rubric" in r["scores"]:
                    sc["rubric"] = r["scores"]["rubric"]
                p = evaluators.task_passed(sc, tasks[r["task_id"]])
                old += r["passed"]; new += p
                if bool(r["passed"]) != p:
                    flips.append((r["task_id"], r["passed"], p))
            print(f"{name:22s} old {old}/{len(rows)}  new {new}/{len(rows)}  flips {flips[:14]}")
finally:
    evaluators.EVALUATORS["rubric"] = rubric_fn
