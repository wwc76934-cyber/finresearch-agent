"""Compare dense/hybrid retrieval through pgvector with the in-memory index on the real-corpus queries.

  # needs docker compose up -d db, then bootstrap with VECTOR_BACKEND=pgvector
  DATABASE_URL=postgresql+psycopg://... python scripts/check_pg_parity.py
Writes reports/eval/pg_parity.json.
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from evaluation_platform.rag_eval import build_real_queries  # noqa: E402
from rag_engine.index import build_chunks, get_retriever  # noqa: E402
from rag_engine.retrieval import Retriever  # noqa: E402

if __name__ == "__main__":
    os.environ["VECTOR_BACKEND"] = "pgvector"
    get_retriever.cache_clear()
    pg = get_retriever(None, anchors=True)
    assert pg.vector_store is not None, "pgvector index is not active (not postgres, or embeddings missing)"
    mem = Retriever(build_chunks(None))
    rows = build_real_queries()
    out = {"n_queries": len(rows), "modes": {}}
    for mode, rerank in (("dense", False), ("hybrid", False), ("hybrid", True)):
        same_top5, same_set, rec = 0, 0, {"pg": 0, "mem": 0}
        t_pg = t_mem = 0.0
        for q in rows:
            kw = dict(ticker=q["ticker"], as_of=datetime.fromisoformat(q["as_of"]), mode=mode, top_k=5, rerank=rerank)
            t0 = time.perf_counter(); a = pg.search(q["query"], **kw); t_pg += time.perf_counter() - t0
            t0 = time.perf_counter(); b = mem.search(q["query"], **kw); t_mem += time.perf_counter() - t0
            ia, ib = [h.chunk.chunk_id for h in a], [h.chunk.chunk_id for h in b]
            same_top5 += ia == ib
            same_set += set(ia) == set(ib)
            for key, hits in (("pg", a), ("mem", b)):
                rec[key] += any(h.chunk.document_id == q["target_document_id"]
                                and q["target_section_contains"] in h.chunk.section_path for h in hits)
        n = len(rows)
        out["modes"][f"{mode}{'+rerank' if rerank else ''}"] = {
            "identical_top5_order": same_top5 / n, "identical_top5_set": same_set / n,
            "recall@5_pgvector": rec["pg"] / n, "recall@5_memory": rec["mem"] / n,
            "ms_per_query_pgvector": round(t_pg / n * 1000, 1), "ms_per_query_memory": round(t_mem / n * 1000, 1)}
    (ROOT / "reports" / "eval" / "pg_parity.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
