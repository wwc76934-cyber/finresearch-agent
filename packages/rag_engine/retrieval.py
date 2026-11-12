"""Hybrid retrieval: BM25 + dense + RRF fusion + reranker, with metadata pre-filtering.

The dense encoder here is a hashed character n-gram TF-IDF model, so the
project runs offline with no model download. `DenseEncoder` is the plug-in
point for bge-m3 / bge-large-zh; `Reranker` is the plug-in point for
bge-reranker. Both are swapped by passing a different object to Retriever.
"""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol

import numpy as np

from .chunking import Chunk

WORD_RE = re.compile(r"[A-Za-z0-9.]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Chinese: unigrams + bigrams; ASCII: words."""
    units = WORD_RE.findall(text)
    toks = []
    for i, u in enumerate(units):
        toks.append(u.lower())
        if i + 1 < len(units) and len(u) == 1 and len(units[i + 1]) == 1 and "\u4e00" <= u <= "\u9fff":
            toks.append(u + units[i + 1])
    return toks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, q: list[str], idx: list[int]) -> np.ndarray:
        out = np.zeros(len(idx))
        for j, i in enumerate(idx):
            tf, L = self.tf[i], self.len[i]
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out[j] = s
        return out


class DenseEncoder(Protocol):
    def fit(self, texts: list[str]) -> None: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingDenseEncoder:
    """Offline stand-in for an embedding model: L2-normalised TF-IDF over hashed char 1-3 grams."""

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self.idf = np.ones(dim, dtype=np.float32)

    def _grams(self, t: str) -> list[int]:
        t = re.sub(r"\s+", "", t)
        # zlib.crc32 is stable across processes (built-in hash() is salted)
        return [zlib.crc32(t[i:i + n].encode()) % self.dim for n in (1, 2, 3) for i in range(len(t) - n + 1)]

    def fit(self, texts: list[str]) -> None:
        df = np.zeros(self.dim)
        for t in texts:
            df[list(set(self._grams(t)))] += 1
        self.idf = (np.log((1 + len(texts)) / (1 + df)) + 1).astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)  # float32: ~10x more chunks with real filings
        for i, t in enumerate(texts):
            for g in self._grams(t):
                m[i, g] += 1
        m *= self.idf
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(n == 0, 1, n)


class ApiDenseEncoder:
    """Real embedding model over an OpenAI-compatible /embeddings endpoint (default BAAI/bge-m3, 1024-d).

    Vectors are cached on disk by text hash, so a corpus is embedded once and offline runs reuse it.
    Selected with EMBEDDING_BACKEND=api; the default stays the offline hashing encoder."""

    def __init__(self, model: Optional[str] = None, dim: int = 1024, batch: int = 32):
        import os
        import sqlite3
        from pathlib import Path

        self.model = model or os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.dim, self.batch = dim, batch
        root = Path(os.environ.get("EMBEDDING_CACHE_DIR", Path(__file__).resolve().parents[2] / ".cache"))
        root.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(root / f"emb_{self.model.replace('/', '_')}.sqlite"), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS e (h TEXT PRIMARY KEY, v BLOB)")
        self._lock = __import__("threading").Lock()

    def fit(self, texts: list[str]) -> None:  # nothing to fit
        return None

    @staticmethod
    def _key(t: str) -> str:
        import hashlib
        return hashlib.sha1(t.encode("utf-8")).hexdigest()

    def _fetch(self, texts: list[str]) -> list[np.ndarray]:
        import os
        import time

        import httpx

        from agent_runtime.llm import load_dotenv
        load_dotenv()
        url, key = os.environ["LLM_BASE_URL"].rstrip("/") + "/embeddings", os.environ["LLM_API_KEY"]
        for attempt in range(5):
            try:
                r = httpx.post(url, headers={"Authorization": f"Bearer {key}"}, timeout=60,
                               json={"model": self.model, "input": [t[:6000] for t in texts]})
                if r.status_code == 402:
                    raise RuntimeError("embedding API: insufficient balance (HTTP 402)")
                r.raise_for_status()
                data = sorted(r.json()["data"], key=lambda d: d["index"])
                return [np.asarray(d["embedding"], dtype=np.float32) for d in data]
            except RuntimeError:
                raise
            except Exception:  # noqa: BLE001 - transient network / rate limit: back off and retry
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError

    def encode(self, texts: list[str]) -> np.ndarray:
        from concurrent.futures import ThreadPoolExecutor

        keys = [self._key(t) for t in texts]
        with self._lock:
            have = {h: np.frombuffer(v, dtype=np.float32) for h, v in
                    self._db.execute(f"SELECT h, v FROM e WHERE h IN ({','.join('?' * len(keys))})", keys)} if keys and len(keys) < 900 else \
                {h: np.frombuffer(v, dtype=np.float32) for h, v in self._db.execute("SELECT h, v FROM e")}
        todo = list(dict.fromkeys(i for i, k in enumerate(keys) if k not in have))
        todo_first = {keys[i]: i for i in reversed(todo)}
        miss = [texts[i] for i in sorted(todo_first.values())]
        mk = [keys[i] for i in sorted(todo_first.values())]
        batches = [(mk[i:i + self.batch], miss[i:i + self.batch]) for i in range(0, len(miss), self.batch)]
        if batches:
            with ThreadPoolExecutor(max_workers=4) as ex:
                results = list(ex.map(lambda b: self._fetch(b[1]), batches))
            with self._lock:
                for (ks, _), vs in zip(batches, results):
                    for k, v in zip(ks, vs):
                        v = v / (np.linalg.norm(v) or 1.0)
                        have[k] = v.astype(np.float32)
                        self._db.execute("INSERT OR REPLACE INTO e VALUES (?, ?)", (k, have[k].tobytes()))
                self._db.commit()
        return np.vstack([have[k] for k in keys]) if keys else np.zeros((0, self.dim), dtype=np.float32)


def default_encoder() -> "DenseEncoder":
    import os
    if os.environ.get("EMBEDDING_BACKEND", "hashing") == "api":
        return ApiDenseEncoder()
    return HashingDenseEncoder()


class Reranker(Protocol):
    def score(self, query: str, chunks: list[Chunk]) -> list[float]: ...


class LexicalCrossReranker:
    """Offline stand-in for a cross-encoder: query-term coverage on the full chunk text,
    with a bonus when query words appear in the section path/table name."""

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        q = [t for t in tokenize(query) if len(t) > 1] or tokenize(query)
        out = []
        for c in chunks:
            body = set(tokenize(c.text))
            head = set(tokenize(c.section_path + (c.table_name or "")))
            cov = sum(1 for t in q if t in body) / len(q)
            hcov = sum(1 for t in q if t in head) / len(q)
            out.append(cov + 0.5 * hcov)
        return out


@dataclass
class Hit:
    chunk: Chunk
    score: float
    parent: Optional[Chunk] = None


class Retriever:
    def __init__(self, chunks: list[Chunk], encoder: Optional[DenseEncoder] = None,
                 reranker: Optional[Reranker] = None, vector_store=None):
        self.children = [c for c in chunks if c.level == "child"]
        self.parents = {c.chunk_id: c for c in chunks if c.level == "parent"}
        self.tokens = [tokenize(c.text) for c in self.children]
        self.bm25 = BM25(self.tokens)
        self.encoder = encoder or default_encoder()
        self.encoder.fit([c.text for c in self.children])
        self.vector_store = vector_store          # PgVectorStore: dense search runs in the database
        self._pos = {c.chunk_id: i for i, c in enumerate(self.children)}
        # with a vector store the matrix is not needed in memory
        self.emb = (np.zeros((0, 1)) if vector_store is not None or not self.children
                    else self.encoder.encode([c.text for c in self.children]))
        self.reranker = reranker or LexicalCrossReranker()

    def embeddings(self) -> np.ndarray:
        """The in-memory matrix (also used to fill the vector store)."""
        return self.encoder.encode([c.text for c in self.children])

    def _dense_top(self, query: str, idx: list[int], pool: int, flt: dict) -> list[tuple[int, float]]:
        """Top `pool` (child index, cosine) by dense similarity among the pre-filtered candidates."""
        qv = self.encoder.encode([query])[0]
        if self.vector_store is not None:
            rows = self.vector_store.search(qv, pool, **flt)
            return [(self._pos[cid], s) for cid, s in rows if cid in self._pos]
        sd = self.emb[idx] @ qv
        return [(idx[j], float(sd[j])) for j in np.argsort(-sd)[:pool]]

    def _filter(self, ticker: Optional[str], as_of: Optional[datetime], scope: Optional[str],
                section_types: Optional[list[str]], doc_ids: Optional[list[str]]) -> list[int]:
        idx = []
        for i, c in enumerate(self.children):
            if ticker and c.ticker != ticker:
                continue
            if as_of and c.available_at > as_of:
                continue
            if scope and c.statement_scope not in (scope, None):
                continue
            if section_types and c.section_type not in section_types:
                continue
            if doc_ids and c.document_id not in doc_ids:
                continue
            idx.append(i)
        return idx

    def search(self, query: str, ticker: Optional[str] = None, as_of: Optional[datetime] = None,
               mode: str = "hybrid", top_k: int = 5, scope: Optional[str] = None,
               section_types: Optional[list[str]] = None, rerank: bool = True,
               doc_ids: Optional[list[str]] = None, return_parents: bool = True) -> list[Hit]:
        idx = self._filter(ticker, as_of, scope, section_types, doc_ids)
        if not idx:
            return []
        qt = tokenize(query)
        pool = max(top_k * 4, 20)
        flt = {"ticker": ticker, "as_of": as_of, "scope": scope, "section_types": section_types, "doc_ids": doc_ids}
        if mode in ("bm25", "dense"):
            if mode == "bm25":
                s = self.bm25.scores(qt, idx)
                top = np.argsort(-s)[:pool]
                ranked = [(idx[j], float(s[j])) for j in top]
            else:
                ranked = self._dense_top(query, idx, pool, flt)
            order = [i for i, _ in ranked]
            fused = dict(ranked)
        else:  # hybrid RRF
            sb = self.bm25.scores(qt, idx)
            bm25_rank = [idx[j] for j in np.argsort(-sb)[:pool]]
            dense_rank = [i for i, _ in self._dense_top(query, idx, pool, flt)]
            fused = {}
            for ranking in (bm25_rank, dense_rank):
                for rank, i in enumerate(ranking):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank + 1)
            order = sorted(fused, key=lambda i: -fused[i])[:pool]
        cands = [self.children[i] for i in order]
        if rerank and mode == "hybrid":
            rs = self.reranker.score(query, cands)
            ranked = sorted(zip(cands, rs, [fused[i] for i in order]), key=lambda x: (-x[1], -x[2]))
            hits = [Hit(c, s) for c, s, _ in ranked]
        else:
            hits = [Hit(self.children[i], fused[i]) for i in order]
        out, seen_parent = [], set()
        for h in hits:
            if return_parents and h.chunk.parent_id:
                if h.chunk.parent_id in seen_parent:
                    continue
                seen_parent.add(h.chunk.parent_id)
                h.parent = self.parents.get(h.chunk.parent_id)
            out.append(h)
            if len(out) >= top_k:
                break
        return out
