"""Retrieval evaluation: Recall@5 / MRR / table integrity across retrieval configurations.

Two query sets:
* lab:  72 hand-written queries over the 6 fictitious companies (benchmarks/golden_snapshot/rag_eval_v1.jsonl)
* real: queries generated over the 8 real companies' filings. For every periodic report the target is the
  section that answers the query (合并利润表 / 应收账款 note / 存货 note / 管理层讨论与分析), searched across
  all 157 real documents (~14k chunks) with the PIT filter at the report's own availability time.
"""
from __future__ import annotations

import json
from datetime import datetime

from data_platform.db import ROOT
from rag_engine.index import get_retriever

EVAL = ROOT / "benchmarks" / "golden_snapshot" / "rag_eval_v1.jsonl"
REAL_EVAL = ROOT / "benchmarks" / "real_snapshot" / "rag_eval_real_v1.jsonl"

CONFIGS = {
    "bm25": dict(mode="bm25", anchors=True, prefilter=True, rerank=False),
    "dense": dict(mode="dense", anchors=True, prefilter=True, rerank=False),
    "hybrid_rrf": dict(mode="hybrid", anchors=True, prefilter=True, rerank=False),
    "hybrid_rrf_rerank": dict(mode="hybrid", anchors=True, prefilter=True, rerank=True),
    "hybrid_rerank_no_anchor": dict(mode="hybrid", anchors=False, prefilter=True, rerank=True),
    "hybrid_rerank_no_anchor_no_prefilter": dict(mode="hybrid", anchors=False, prefilter=False, rerank=True),
}
PERIOD_CN = {"Q1": "第一季度报告", "Q2": "半年度报告", "Q3": "第三季度报告", "Q4": "年度报告"}
INTENTS = [  # (query template, section keyword the target chunk path must contain, doc types)
    ("{p} 合并利润表 营业收入 净利润", "合并利润表", None),
    ("{p} 应收账款 账龄 坏账准备", "应收账款", ("annual_report", "semi_annual_report")),
    ("{p} 存货 跌价准备 账面余额", "存货", ("annual_report", "semi_annual_report")),
    ("{p} 管理层讨论与分析 主营业务 经营情况", "管理层讨论与分析", ("annual_report", "semi_annual_report")),
]


def build_real_queries() -> list[dict]:
    """Generate (and cache) real-corpus queries: one per intent per periodic report whose target section exists."""
    if REAL_EVAL.exists():
        return [json.loads(l) for l in REAL_EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    r = get_retriever(None, anchors=True)
    by_doc: dict[str, set[str]] = {}
    meta = {}
    for c in r.children:
        if c.ticker[:2] == "99":
            continue
        by_doc.setdefault(c.document_id, set()).add(c.section_path)
        meta[c.document_id] = c
    rows = []
    for did, paths in sorted(by_doc.items()):
        if not did.startswith("doc_") or "_v" in did[-3:]:
            continue
        c = meta[did]
        q = did.split("_")[-1]
        period = f"{q[:4]}年{PERIOD_CN.get(q[4:], '')}"
        dtype = {"Q1": "q1_report", "Q2": "semi_annual_report", "Q3": "q3_report", "Q4": "annual_report"}[q[4:]]
        for tmpl, sec, types in INTENTS:
            if types and dtype not in types:
                continue
            if not any(sec in p.split(" > ")[-1] or (sec == "管理层讨论与分析" and sec in p) for p in paths):
                continue
            rows.append({"query": tmpl.format(p=period), "ticker": c.ticker, "as_of": c.available_at.isoformat(),
                         "target_document_id": did, "target_section_contains": sec, "period": q})
    REAL_EVAL.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    return rows


def _score(rows: list[dict]) -> dict:
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
    return out


def table_integrity(retriever, real: bool = False) -> float:
    """Share of 合并利润表 parent chunks that contain both the first (revenue) and a late (net profit) row,
    i.e. the statement was not cut where the PDF page broke."""
    tabs = [c for c in retriever.parents.values() if c.table_name and "合并利润表" in c.table_name
            and (c.ticker[:2] != "99") == real]
    ok = sum(1 for c in tabs if "归属于母公司" in c.text and "营业收入" in c.text)
    return ok / len(tabs) if tabs else 0.0


def run() -> dict:
    lab_rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = _score(lab_rows)
    r = get_retriever(None, anchors=True)
    out["table_integrity"] = {"structure_aware": table_integrity(r)}
    naive = get_retriever(None, naive=True)
    naive_ok = sum(1 for c in naive.children if "营业收入" in c.text and "归属于母公司股东的净利润" in c.text and "合并利润表" in c.text)
    total = sum(1 for c in naive.children if "合并利润表" in c.text and c.ticker[:2] == "99")
    out["table_integrity"]["naive_fixed_size"] = naive_ok / total if total else 0.0
    real_rows = build_real_queries()
    if real_rows:
        out["real"] = _score(real_rows)
        cross = [c for c in r.parents.values() if c.table_name and "合并利润表" in c.table_name
                 and c.ticker[:2] != "99" and c.page_end > c.page_start]
        out["real"]["table_integrity"] = {"structure_aware": table_integrity(r, real=True),
                                          "statements_spanning_pages": len(cross)}
    return out
