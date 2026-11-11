"""Why does B1 (naive RAG) score so low? Per-category pass/fail and the PIT leak rate, offline."""
import collections
import os
import sys

os.environ["LLM_OFFLINE"] = "1"
sys.path.insert(0, "packages")
sys.stdout.reconfigure(encoding="utf-8")
from evaluation_platform.runner import run_experiment

p = run_experiment(["B1"], datasets=("golden", "blind", "chaos", "lab", "quant"), workers=2, persist_archs=(),
                   name="b1_diag", write_latest=False)
by = collections.defaultdict(lambda: [0, 0])
why = collections.Counter()
for r in p["results"]:
    by[r["category"]][0] += r["passed"]; by[r["category"]][1] += 1
    if not r["passed"]:
        s = r["scores"]
        fails = [k for k in ("numerical_accuracy", "refusal", "pit_violation", "anomaly_recall", "attribution", "citation_validity")
                 if k in s and ((k == "pit_violation" and s[k]["score"] > 0) or (k != "pit_violation" and s[k].get("score", 1) < 1))]
        why[tuple(fails)] += 1
print({k: f"{v[0]}/{v[1]}" for k, v in sorted(by.items())})
print("total", sum(v[0] for v in by.values()), "/", sum(v[1] for v in by.values()))
for k, n in why.most_common(8):
    print(n, k)
