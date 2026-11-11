"""Seal a blind set: record the SHA-256 of the question file and of the built tasks, with the git HEAD that existed
BEFORE any rule was tuned against it. Commit the sealed file and this manifest together, then do not tune on it.

  python scripts/seal_blind.py blind_v2 blind2
"""
import collections
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src_name, dataset = sys.argv[1], sys.argv[2]
src = ROOT / "benchmarks" / "blind" / f"{src_name}.jsonl"
built = ROOT / "benchmarks" / "tasks" / f"{dataset}_v1.jsonl"
rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
manifest = {
    "dataset": dataset, "sealed_at": datetime.now().isoformat(timespec="seconds"), "git_head_before_tuning": head,
    "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "tasks_sha256": hashlib.sha256(built.read_bytes()).hexdigest(),
    "n": len(rows), "by_type": dict(collections.Counter(r["type"] for r in rows)),
    "by_ticker": dict(collections.Counter(r["ticker"] for r in rows)), "authors": sorted({r["author"] for r in rows}),
    "rule": "First-run score on this set is the blind score. Any detector/threshold/prompt change made after seeing "
            "results on it makes later runs a development score, not a blind score.",
}
out = ROOT / "benchmarks" / "blind" / f"SEALED_{src_name}.json"
out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=1))
