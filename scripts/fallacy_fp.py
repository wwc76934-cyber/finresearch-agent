"""False-positive check of the Tier-2 fallacy detectors on reports that are known to be fine.

Runs B4 (offline) on real-company tasks, then applies every detector to every kept claim. A kept claim in a passing
report is a correct claim, so every hit is a false positive. Stored online reports (no new model calls) can be added.
  python scripts/fallacy_fp.py [online_run_name ...]
"""
import collections
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["LLM_OFFLINE"] = "1"; os.environ["CHECKPOINTER"] = "memory"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages")); sys.stdout.reconfigure(encoding="utf-8")
from agent_runtime import ResearchRequest
from agent_runtime import graph
from agent_runtime.agents import semantic_check as S
from agent_runtime.state import ARCHITECTURES, Claim
from evaluation_platform.datasets import load_tasks

hits = collections.Counter(); n_claims = 0; examples = []
for t in load_tasks(("golden", "blind", "multidoc")):
    ctx = graph.prepare(ResearchRequest(ticker=t["ticker"], as_of=datetime.fromisoformat(t["as_of"]), question=t["question"], mode="deep"),
                        ARCHITECTURES["B4"])
    ctx.started, ctx.t0, ctx.persist, ctx.arch_name = datetime.now(), 0, False, "B4"
    import time
    ctx.t0 = time.perf_counter()
    init = {"run_id": ctx.run_id, "request": json.loads(ctx.request.model_dump_json()), "snapshot_id": ctx.snapshot_id,
            "summaries": [], "handled_anomalies": [], "claim_ids": [], "shared_history": []}
    rep = graph._drive(ctx, init)
    kept = {c for s in rep["sections"] for c in s["claim_ids"]}
    for cid in kept:
        n_claims += 1
        c = ctx.claims[cid]
        for name, why in S.fallacies(ctx, c):
            hits[name] += 1
            if len(examples) < 8:
                examples.append((t["task_id"], name, c.text[:70], why[:90]))
print(f"kept claims scanned: {n_claims} | false positives by class: {dict(hits)} | total {sum(hits.values())}")
for e in examples:
    print("  ", e)
