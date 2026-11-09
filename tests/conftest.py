import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
os.environ.setdefault("LLM_OFFLINE", "1")  # tests are deterministic


@pytest.fixture(scope="session", autouse=True)
def bootstrapped():
    from sqlalchemy import func, select

    from data_platform.db import DocumentChunkRow, FinancialFactRow, get_session_factory

    with get_session_factory()() as s:
        n = s.scalar(select(func.count()).select_from(FinancialFactRow))
        c = s.scalar(select(func.count()).select_from(DocumentChunkRow))
    if not n or not c:
        from data_platform.ingestion import IngestionPipeline
        from rag_engine.index import persist_chunks

        IngestionPipeline().ingest_golden()
        persist_chunks()
    yield
