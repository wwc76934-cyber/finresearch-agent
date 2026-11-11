"""Detector-only comparison of anomaly modes (rules vs statistical) against each set's answer key.

  python scripts/anomaly_eval.py [dataset ...]          default: blind golden lab
No agent, no model: just calculators.detect_anomalies at the task's (ticker, as_of). blind2 is refused here on
purpose: it is sealed and only gets one comparison run at the end (scripts/seal_blind.py).
"""
import os
import sys
from datetime import datetime

os.environ["LLM_OFFLINE"] = "1"
sys.path.insert(0, "packages")
sys.stdout.reconfigure(encoding="utf-8")
from agent_runtime import ResearchRequest
from agent_runtime.graph import prepare
from agent_runtime.state import ARCHITECTURES
from evaluation_platform.datasets import load_tasks
from data_platform.serving import label_to_end, period_label

sets = sys.argv[1:] or ["blind", "golden", "lab"]
if "blind2" in sets:
    sys.exit("blind2 is sealed; run it only through the final comparison")
import json
from pathlib import Path

tasks = [t for t in load_tasks(tuple(x for x in sets if x != "dev")) if t["category"] == "anomaly_attribution"] if \
    [x for x in sets if x != "dev"] else []
if "dev" in sets:       # independent analyst labels, benchmarks/dev/anomaly_dev.jsonl
    for l in Path("benchmarks/dev/anomaly_dev.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            d = json.loads(l)
            tasks.append({"task_id": d["id"], "dataset": "dev", "ticker": d["ticker"], "as_of": d["as_of"],
                          "expected": {"anomalies": d["expect_anomalies"]}, "category": "anomaly_attribution"})
CODES = ["PROFIT_OCF_DIVERGENCE", "AR_OUTPACE_REVENUE", "INVENTORY_DAYS_DETERIORATION", "GROSS_MARGIN_SHIFT",
         "LARGE_IMPAIRMENT", "INTEREST_DEBT_JUMP", "RELATED_PARTY_OR_POLICY_CHANGE"]
verbose = os.environ.get("VERBOSE")
res = {}
for mode in ("rules", "statistical"):
    tp = fp = fn = 0
    per = {}
    for t in tasks:
        exp = set(t["expected"].get("anomalies", []))
        req = ResearchRequest(ticker=t["ticker"], as_of=datetime.fromisoformat(t["as_of"]))
        ctx = prepare(req, ARCHITECTURES["B4"])
        pe = ctx.svc.latest_period(t["ticker"])
        if pe is None:
            continue
        anns = [(d.document_id, d.title) for d in ctx.svc.documents(t["ticker"], "announcement")]
        got = {f.code for f in ctx.calc.detect_anomalies(t["ticker"], pe, anns, mode=mode)}
        tp += len(exp & got); fp += len(got - exp); fn += len(exp - got)
        per.setdefault(t["dataset"], [0, 0, 0])
        per[t["dataset"]][0] += len(exp & got); per[t["dataset"]][1] += len(got - exp); per[t["dataset"]][2] += len(exp - got)
        if verbose:
            print(f"  [{mode}] {t['task_id']} {t['ticker']} {period_label(pe)} exp={sorted(exp)} got={sorted(got)}")
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    res[mode] = (tp, fp, fn, p, r)
    print(f"{mode:12s} TP={tp:3d} FP={fp:3d} FN={fn:3d}  precision={p:.2f} recall={r:.2f}   per-set (tp,fp,fn): {per}")
