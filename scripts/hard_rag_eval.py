"""Recall@5 / MRR of each retrieval config on the hard (paraphrased) query set. Offline, no LLM.

  python scripts/hard_rag_eval.py   ->  reports/eval/rag_hard_<EMBEDDING_BACKEND>.json
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["LLM_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from evaluation_platform.rag_eval import CONFIGS  # noqa: E402
from rag_engine.index import get_retriever  # noqa: E402

if __name__ == "__main__":
    rows = [json.loads(l) for l in (ROOT / "benchmarks/real_snapshot/rag_eval_hard_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for name, cfg in CONFIGS.items():
        if not cfg["prefilter"]:
            continue
        r = get_retriever(None, anchors=cfg["anchors"])
        pid = {c.chunk_id: c.parent_id for c in r.children}
        rec = mrr = 0
        by_type: dict = {}
        for q in rows:
            hits = r.search(q["query"], ticker=q["ticker"], as_of=datetime.fromisoformat(q["as_of"]), mode=cfg["mode"], top_k=5,
                            rerank=cfg["rerank"])
            rank = next((i + 1 for i, h in enumerate(hits) if h.chunk.chunk_id == q["source_chunk"]), None)
            rec += rank is not None
            mrr += 1 / rank if rank else 0
            b = by_type.setdefault(q["section_type"], [0, 0])
            b[0] += rank is not None
            b[1] += 1
        out[name] = {"recall@5": round(rec / len(rows), 3), "mrr": round(mrr / len(rows), 3), "n": len(rows),
                     "recall_by_section": {k: f"{a}/{b}" for k, (a, b) in by_type.items()}}
        print(name, out[name], flush=True)
    (ROOT / f"reports/eval/rag_hard_{os.environ.get('EMBEDDING_BACKEND', 'hashing')}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
