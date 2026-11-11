"""Offline pass-rate check for a few architectures without rewriting EVALUATION.md.

  python scripts/quick_offline.py B4,B3 golden,blind,chaos,lab
"""
import os
import sys
from collections import Counter
from pathlib import Path

os.environ["LLM_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from evaluation_platform.runner import run_experiment  # noqa: E402

if __name__ == "__main__":
    archs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["B4"]
    ds = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("golden", "blind", "chaos", "lab", "quant")
    p = run_experiment(archs, datasets=ds, workers=2, persist_archs=(), name="quick_offline", write_latest=False)
    for a in archs:
        rows = [r for r in p["results"] if r["architecture"] == a]
        tot, ok = Counter(r["dataset"] for r in rows), Counter(r["dataset"] for r in rows if r["passed"])
        print(a, {d: f"{ok[d]}/{tot[d]}" for d in ds}, f"{sum(ok.values())}/{len(rows)}",
              "fail:", [r["task_id"] for r in rows if not r["passed"]])
