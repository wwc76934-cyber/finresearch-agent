"""Build, persist and load the chunk index for all source documents."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from sqlalchemy import delete, select

from data_platform.db import ROOT, DocumentChunkRow, SourceDocumentRow, get_session_factory
from data_platform.serving import period_label

from .chunking import Chunk, chunk_document, naive_chunks
from .retrieval import Retriever

GOLDEN = ROOT / "benchmarks" / "golden_snapshot"


def _docs(db_url: Optional[str]):
    with get_session_factory(db_url)() as s:
        return list(s.scalars(select(SourceDocumentRow)))


def build_chunks(db_url: Optional[str] = None, anchors: bool = True, naive: bool = False) -> list[Chunk]:
    out = []
    for d in _docs(db_url):
        text = (GOLDEN / d.path).read_text(encoding="utf-8")
        if naive:
            out += naive_chunks(d.document_id, d.ticker, text, d.available_at)
        else:
            period = period_label(d.period_end) if d.period_end else d.published_at.date().isoformat()
            out += chunk_document(d.document_id, d.ticker, period, d.doc_type, text, d.available_at, anchors=anchors)
    return out


def persist_chunks(db_url: Optional[str] = None) -> int:
    chunks = build_chunks(db_url)
    with get_session_factory(db_url)() as s:
        s.execute(delete(DocumentChunkRow))
        s.add_all(DocumentChunkRow(**c.__dict__) for c in chunks)
        s.commit()
    return len(chunks)


@lru_cache(maxsize=6)
def get_retriever(db_url: Optional[str] = None, anchors: bool = True, naive: bool = False) -> Retriever:
    return Retriever(build_chunks(db_url, anchors=anchors, naive=naive))


def chunk_text(db_url: Optional[str], chunk_id: str) -> Optional[DocumentChunkRow]:
    with get_session_factory(db_url)() as s:
        return s.get(DocumentChunkRow, chunk_id)
