"""Load the Golden Snapshot into the database and build the chunk index."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from data_platform.ingestion import IngestionPipeline  # noqa: E402
from rag_engine.index import persist_chunks  # noqa: E402

if __name__ == "__main__":
    stats = IngestionPipeline().ingest_golden()
    issues = stats.pop("quality_issues")
    stats["chunks"] = persist_chunks()
    stats["quality_issue_counts"] = {}
    for i in issues:
        stats["quality_issue_counts"][i["rule"]] = stats["quality_issue_counts"].get(i["rule"], 0) + 1
    print(json.dumps(stats, ensure_ascii=False, indent=2))
