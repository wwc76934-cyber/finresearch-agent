"""Retrieval evaluation: Recall@5 / MRR / table integrity across retrieval configurations."""
from __future__ import annotations

import json
from datetime import datetime

from data_platform.db import ROOT
from rag_engine.index import get_retriever

EVAL = ROOT / "benchmarks" / "golden_snapshot" / "rag_eval_v1.jsonl"

CONFIGS = {
    "bm25": dict(mode="bm25", anchors=True, prefilter=True, rerank=False),
    "dense": dict(mode="dense", anchors=True, prefilter=True, rerank=False),
    "hybrid_rrf": dict(mode="hybrid", anchors=True, prefilter=True, rerank=False),
    "hybrid_rrf_rerank": dict(mode="hybrid", anchors=True, prefilter=True, rerank=True),
    "hybrid_rerank_no_anchor": dict(mode="hybrid", anchors=False, prefilter=True, rerank=True),
    "hybrid_rerank_no_anchor_no_prefilter": dict(mode="hybrid", anchors=False, prefilter=False, rerank=True),
}


def table_integrity(retriever) -> float:
    """Share of 合并利润表 parent chunks that still contain the row printed after the page break."""
    tabs = [c for c in retriever.parents.values() if c.table_name and "合并利润表" in c.table_name]
    ok = sum(1 for c in tabs if "归属于母公司股东的净利润" in c.text and "营业收入" in c.text)
    return ok / len(tabs) if tabs else 0.0


def run() -> dict:
    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for name, cfg in CONFIGS.items():
        r = get_retriever(None, anchors=cfg["anchors"])
        rec, mrr = 0, 0.0
        for q in rows:
            hits = r.search(q["query"], ticker=q["ticker"] if cfg["prefilter"] else None,
                            as_of=datetime.fromisoformat(q["as_of"]) if cfg["prefilter"] else None,
                            mode=cfg["mode"], top_k=5, rerank=cfg["rerank"])
            rank = next((i + 1 for i, h in enumerate(hits) if h.chunk.document_id == q["target_document_id"]
                         and q["target_section_contains"] in h.chunk.section_path), None)
            rec += rank is not None
            mrr += 1 / rank if rank else 0
        out[name] = {"recall@5": rec / len(rows), "mrr": mrr / len(rows), "n": len(rows)}
    out["table_integrity"] = {"structure_aware": table_integrity(get_retriever(None, anchors=True))}
    naive = get_retriever(None, naive=True)
    naive_ok = sum(1 for c in naive.children if "营业收入" in c.text and "归属于母公司股东的净利润" in c.text and "合并利润表" in c.text)
    total = sum(1 for c in naive.children if "合并利润表" in c.text)
    out["table_integrity"]["naive_fixed_size"] = naive_ok / total if total else 0.0
    return out
