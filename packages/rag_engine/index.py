"""Build, persist and load the chunk index for all source documents."""
from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional

from sqlalchemy import delete, select

from data_platform.db import ROOT, DocumentChunkRow, SourceDocumentRow, get_session_factory
from data_platform.serving import period_label

from .chunking import Chunk, chunk_document, naive_chunks
from .retrieval import HashingDenseEncoder, Retriever
from .vector_store import pg_store_if_enabled

GOLDEN = ROOT / "benchmarks" / "golden_snapshot"
DENSE_DIM = HashingDenseEncoder().dim


def _docs(db_url: Optional[str]):
    with get_session_factory(db_url)() as s:
        return list(s.scalars(select(SourceDocumentRow)))


def doc_path(path: str):
    """Real-snapshot paths are relative to benchmarks/ ('real_snapshot/documents/..'); demo ones to golden_snapshot/."""
    p = ROOT / "benchmarks" / path
    return p if p.exists() else GOLDEN / path


def build_chunks(db_url: Optional[str] = None, anchors: bool = True, naive: bool = False) -> list[Chunk]:
    out = []
    for d in _docs(db_url):
        p = doc_path(d.path)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        if naive:
            out += naive_chunks(d.document_id, d.ticker, text, d.available_at)
        else:
            period = period_label(d.period_end) if d.period_end else d.published_at.date().isoformat()
            out += chunk_document(d.document_id, d.ticker, period, d.doc_type, text, d.available_at, anchors=anchors)
    return out


def persist_chunks(db_url: Optional[str] = None) -> int:
    """Write the chunks; with VECTOR_BACKEND=pgvector also write one embedding per child chunk."""
    chunks = build_chunks(db_url)
    with get_session_factory(db_url)() as s:
        s.execute(delete(DocumentChunkRow))
        s.add_all(DocumentChunkRow(**c.__dict__) for c in chunks)
        s.commit()
    store = pg_store_if_enabled(db_url, DENSE_DIM)
    if store is not None:
        r = Retriever(chunks)           # in-memory: fits the encoder on this exact corpus
        store.upsert([c.chunk_id for c in r.children], r.embeddings())
        get_retriever.cache_clear()
    return len(chunks)


@lru_cache(maxsize=6)
def get_retriever(db_url: Optional[str] = None, anchors: bool = True, naive: bool = False) -> Retriever:
    chunks = build_chunks(db_url, anchors=anchors, naive=naive)
    store = None
    # only the default configuration has stored embeddings; the ablation indexes (no anchors, naive
    # chunking) have different texts and stay in memory
    if anchors and not naive:
        store = pg_store_if_enabled(db_url, DENSE_DIM)
        n_children = sum(1 for c in chunks if c.level == "child")
        if store is not None and store.embedded_count() != n_children:
            warnings.warn("pgvector index is stale or empty; run scripts/bootstrap.py. Using the in-memory index.")
            store = None
    return Retriever(chunks, vector_store=store)


def chunk_text(db_url: Optional[str], chunk_id: str) -> Optional[DocumentChunkRow]:
    with get_session_factory(db_url)() as s:
        return s.get(DocumentChunkRow, chunk_id)
