import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
os.environ.setdefault("LLM_OFFLINE", "1")  # tests are deterministic


@pytest.fixture(scope="session", autouse=True)
def bootstrapped():
    """A fresh checkout (CI) has no database: load both snapshots, like scripts/bootstrap.py."""
    from sqlalchemy import func, select

    from data_platform.db import DocumentChunkRow, FinancialFactRow, SecurityRow, get_session_factory

    real = ROOT / "benchmarks" / "real_snapshot"
    with get_session_factory()() as s:
        n = s.scalar(select(func.count()).select_from(FinancialFactRow))
        c = s.scalar(select(func.count()).select_from(DocumentChunkRow))
        has_real = s.scalar(select(func.count()).select_from(SecurityRow).where(SecurityRow.ticker == "300750.SZ"))
    if not n or not c or ((real / "financial_facts.csv").exists() and not has_real):
        from data_platform.ingestion import IngestionPipeline
        from rag_engine.index import persist_chunks

        pipe = IngestionPipeline()
        for name in ("golden_snapshot", "real_snapshot"):
            if (ROOT / "benchmarks" / name / "financial_facts.csv").exists():
                pipe.ingest_golden(ROOT / "benchmarks" / name)
        persist_chunks()
    yield
