"""Load the snapshots into the database and build the chunk index.

benchmarks/real_snapshot   8 real A-share companies (built by scripts/build_real_snapshot.py)
benchmarks/golden_snapshot 6 fictitious "chaos lab" companies with designed traps (restatements, vendor conflicts)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from data_platform.ingestion import IngestionPipeline  # noqa: E402
from rag_engine.index import persist_chunks  # noqa: E402

if __name__ == "__main__":
    pipe = IngestionPipeline()
    out = {}
    for name in ("golden_snapshot", "real_snapshot"):
        path = ROOT / "benchmarks" / name
        if not (path / "financial_facts.csv").exists():
            continue
        stats = pipe.ingest_golden(path)
        issues = stats.pop("quality_issues")
        stats["quality_issue_counts"] = {}
        for i in issues:
            stats["quality_issue_counts"][i["rule"]] = stats["quality_issue_counts"].get(i["rule"], 0) + 1
        out[name] = stats
    out["chunks"] = persist_chunks()
    print(json.dumps(out, ensure_ascii=False, indent=2))
