"""pgvector-backed dense search (PRD v1.2 section 3.1).

VECTOR_BACKEND=pgvector stores one halfvec(2048) per child chunk in document_chunks.embedding, with an HNSW
index, and runs the metadata pre-filter (ticker, available_at, statement_scope, section_type, document) in SQL
next to the vector search. BM25 stays in memory; Retriever fuses both with RRF.
VECTOR_BACKEND=memory (default) keeps the numpy matrix, so unit tests need no services.

halfvec rather than vector: pgvector's HNSW index caps `vector` at 2000 dimensions, `halfvec` at 4000, and the
hashed encoder uses 2048.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import text

from data_platform.db import get_engine


def vector_backend() -> str:
    return os.environ.get("VECTOR_BACKEND", "memory").lower()


def _literal(row: np.ndarray) -> str:
    """Sparse-friendly text literal '[0,0,0.12,...]' (the hashed vectors are >85% zeros)."""
    out = ["0"] * len(row)
    for i in np.flatnonzero(row):
        out[i] = f"{row[i]:.5g}"
    return "[" + ",".join(out) + "]"


class PgVectorStore:
    def __init__(self, db_url: Optional[str], dim: int):
        self.db_url, self.dim = db_url, dim
        self.engine = get_engine(db_url)
        if self.engine.dialect.name != "postgresql":
            raise RuntimeError("VECTOR_BACKEND=pgvector needs a PostgreSQL DATABASE_URL (STORAGE_BACKEND=postgres)")
        self._iterative = False

    def ensure_schema(self) -> None:
        with self.engine.begin() as c:
            c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            c.execute(text(f"ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding halfvec({self.dim})"))
            c.execute(text("CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw ON document_chunks "
                           "USING hnsw (embedding halfvec_cosine_ops)"))
            ver = c.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'")).scalar() or "0"
        major_minor = tuple(int(x) for x in ver.split(".")[:2])
        self._iterative = major_minor >= (0, 8)   # hnsw.iterative_scan arrived in pgvector 0.8

    def upsert(self, chunk_ids: list[str], matrix: np.ndarray, batch: int = 200) -> int:
        sql = text("UPDATE document_chunks SET embedding = CAST(:v AS halfvec) WHERE chunk_id = :id")
        n = 0
        for i in range(0, len(chunk_ids), batch):
            rows = [{"id": cid, "v": _literal(matrix[i + j])} for j, cid in enumerate(chunk_ids[i:i + batch])]
            with self.engine.begin() as c:
                c.execute(sql, rows)
            n += len(rows)
        return n

    def embedded_count(self) -> int:
        with self.engine.connect() as c:
            try:
                return c.execute(text("SELECT count(*) FROM document_chunks WHERE level='child' "
                                      "AND embedding IS NOT NULL")).scalar() or 0
            except Exception:  # noqa: BLE001 - column or extension missing
                return 0

    def search(self, qvec: np.ndarray, limit: int, ticker: Optional[str] = None, as_of: Optional[datetime] = None,
               scope: Optional[str] = None, section_types: Optional[list[str]] = None,
               doc_ids: Optional[list[str]] = None) -> list[tuple[str, float]]:
        where, args = ["level = 'child'", "embedding IS NOT NULL"], {"q": _literal(qvec), "n": limit}
        if ticker:
            where.append("ticker = :ticker"); args["ticker"] = ticker
        if as_of:
            where.append("available_at <= :as_of"); args["as_of"] = as_of
        if scope:
            where.append("(statement_scope = :scope OR statement_scope IS NULL)"); args["scope"] = scope
        if section_types:
            where.append("section_type = ANY(:sts)"); args["sts"] = list(section_types)
        if doc_ids:
            where.append("document_id = ANY(:docs)"); args["docs"] = list(doc_ids)
        sql = text("SELECT chunk_id, 1 - (embedding <=> CAST(:q AS halfvec)) AS sim FROM document_chunks "
                   f"WHERE {' AND '.join(where)} ORDER BY embedding <=> CAST(:q AS halfvec) LIMIT :n")
        with self.engine.begin() as c:
            # HNSW applies the filter after the index scan; a wide candidate list (and iterative scan on 0.8+)
            # keeps selective filters like ticker + as_of from coming back short
            c.execute(text("SET LOCAL hnsw.ef_search = 400"))
            if self._iterative:
                c.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            return [(r[0], float(r[1])) for r in c.execute(sql, args)]


def pg_store_if_enabled(db_url: Optional[str], dim: int) -> Optional[PgVectorStore]:
    if vector_backend() != "pgvector" or get_engine(db_url).dialect.name != "postgresql":
        return None
    store = PgVectorStore(db_url, dim)
    store.ensure_schema()
    return store
