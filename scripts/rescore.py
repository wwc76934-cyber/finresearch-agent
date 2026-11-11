"""Re-score a finished eval from the persisted reports after an evaluator fix, without re-running agents.

The LLM rubric score is kept as stored (re-judging would cost money and change with sampling).
  python scripts/rescore.py online_B4_debate_golden
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from data_platform.db import ResearchRunRow, get_session_factory  # noqa: E402
from evaluation_platform import evaluators  # noqa: E402
from evaluation_platform.comparison import aggregate  # noqa: E402
from evaluation_platform.datasets import load_tasks  # noqa: E402
from evaluation_platform.error_analysis import classify  # noqa: E402

if __name__ == "__main__":
    name = sys.argv[1]
    part = ROOT / "reports" / "eval" / f"{name}.partial.jsonl"
    tasks = {t["task_id"]: t for t in load_tasks(("golden", "blind", "chaos", "lab", "quant"))}
    out = []
    rubric_fn = evaluators.EVALUATORS.pop("rubric")
    try:
        with get_session_factory()() as s:
            for line in part.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                row = s.get(ResearchRunRow, r["run_id"]) if r.get("run_id") else None
                if row is None:
                    out.append(r)
                    continue
                task, rep = tasks[r["task_id"]], row.report
                scores = evaluators.evaluate(task, rep)
                if "rubric" in r["scores"]:
                    scores["rubric"] = r["scores"]["rubric"]
                r.update(scores=scores, passed=evaluators.task_passed(scores, task),
                         error_category=classify(task, rep, scores))
                out.append(r)
    finally:
        evaluators.EVALUATORS["rubric"] = rubric_fn
    (part.parent / f"{name}.rescored.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in out),
                                                        encoding="utf-8")
    agg = aggregate(out)
    for arch, a in agg.items():
        print(arch, f"pass={a['pass_rate']:.3f} rubric={a['rubric']} num={a['numerical_accuracy']}", a["pass_by_category"])
    dims: dict = {}
    for r in out:
        for k, v in r["scores"].get("rubric", {}).get("dims", {}).items():
            dims.setdefault(k, []).append(v)
    print("rubric dims:", {k: round(sum(v) / len(v), 3) for k, v in dims.items()})
    for r in out:
        if not r["passed"]:
            print("FAIL", r["task_id"], r["error_category"], {k: v.get("score") for k, v in r["scores"].items()
                                                              if isinstance(v, dict) and "score" in v})
