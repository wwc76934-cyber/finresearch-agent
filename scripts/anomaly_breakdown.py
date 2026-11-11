"""Per-code precision/recall of the statistical detector on the analyst-labelled dev set, plus parameter sweeps.

  python scripts/anomaly_breakdown.py
The dev set (benchmarks/dev/anomaly_dev.jsonl) is development data. blind2 is never read here.
"""
import collections
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["LLM_OFFLINE"] = "1"
sys.path.insert(0, "packages")
sys.stdout.reconfigure(encoding="utf-8")
from agent_runtime import ResearchRequest
from agent_runtime.graph import prepare
from agent_runtime.state import ARCHITECTURES
from data_platform import anomaly_stats as A

rows = [json.loads(l) for l in Path("benchmarks/dev/anomaly_dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
ctxs = []
for d in rows:
    ctx = prepare(ResearchRequest(ticker=d["ticker"], as_of=datetime.fromisoformat(d["as_of"])), ARCHITECTURES["B4"])
    pe = ctx.svc.latest_period(d["ticker"])
    if pe:
        ctxs.append((d, ctx, pe))


def run(label, **over):
    saved = {k: getattr(A, k) for k in ("Z_TRIGGER", "Z_HIGH", "MIN_PEERS")}
    sig_saved = dict(A.SIGNALS)
    for k, v in over.items():
        if k == "floors":
            for code, f in v.items():
                A.SIGNALS[code] = type(A.SIGNALS[code])(**{**A.SIGNALS[code].__dict__, "floor": f})
        elif k == "no_peer":
            A.MIN_PEERS = 10 ** 6
        else:
            setattr(A, k, v)
    per = collections.defaultdict(lambda: [0, 0, 0])
    for d, ctx, pe in ctxs:
        exp = set(d["expect_anomalies"]) - {"RELATED_PARTY_OR_POLICY_CHANGE"}
        got = {f.code for f in A.detect(ctx.calc, d["ticker"], pe)}
        for c in A.SIGNALS:
            per[c][0] += c in exp and c in got
            per[c][1] += c in got and c not in exp
            per[c][2] += c in exp and c not in got
    for k, v in saved.items():
        setattr(A, k, v)
    A.SIGNALS.clear(); A.SIGNALS.update(sig_saved)
    tp, fp, fn = (sum(v[i] for v in per.values()) for i in range(3))
    print(f"{label:28s} TP={tp:2d} FP={fp:2d} FN={fn:2d} P={tp / max(tp + fp, 1):.2f} R={tp / max(tp + fn, 1):.2f}",
          {c[:8]: tuple(v) for c, v in per.items()} if label == "default" else "")


run("default")
run("no peer z", no_peer=True)
for z in (1.5, 2.0, 2.5):
    run(f"Z_TRIGGER={z}", Z_TRIGGER=z)
