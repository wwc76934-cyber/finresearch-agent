"""Agentic RAG: numeric-vs-explanatory routing and iterative self-grading retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

from .retrieval import Hit, Retriever

NUMERIC_PAT = re.compile(r"(是多少|多少|数值|金额|比率|率是|几)")
EXPLAIN_PAT = re.compile(r"(为什么|原因|主要系|驱动|归因|解释|如何|why)", re.I)
SYNONYMS = {"现金流": ["回款", "经营活动产生的现金流量净额"], "应收": ["信用期", "欠款方"],
            "毛利率": ["售价", "成本"], "存货": ["跌价准备", "可变现净值"], "负债": ["借款", "债券"]}


def route(query: str) -> Literal["structured", "hybrid"]:
    """Numbers come from the data platform; explanations come from documents."""
    if EXPLAIN_PAT.search(query):
        return "hybrid"
    if NUMERIC_PAT.search(query):
        return "structured"
    return "hybrid"


def grade(query: str, hits: list[Hit]) -> Literal["SUFFICIENT", "INSUFFICIENT_EXPLANATION", "EMPTY"]:
    if not hits:
        return "EMPTY"
    if EXPLAIN_PAT.search(query) or "原因" in query:
        if all(h.chunk.is_table for h in hits):
            return "INSUFFICIENT_EXPLANATION"
        if not any(k in (h.parent.text if h.parent else h.chunk.text) for h in hits for k in ("主要系", "原因", "所致")):
            return "INSUFFICIENT_EXPLANATION"
    return "SUFFICIENT"


def rewrite(query: str, attempt: int) -> tuple[str, Optional[list[str]]]:
    extra = [w for k, ws in SYNONYMS.items() if k in query for w in ws]
    q = query + " " + " ".join(extra) + " 主要系 原因"
    return q, (["MDA"] if attempt == 0 else ["MDA", "ANNOUNCEMENT", "NOTES"])


@dataclass
class AgenticResult:
    hits: list[Hit]
    grade: str
    attempts: list[dict] = field(default_factory=list)


def agentic_search(retriever: Retriever, query: str, ticker: str, as_of: datetime, top_k: int = 5,
                   section_types: Optional[list[str]] = None, scope: Optional[str] = None,
                   mode: str = "hybrid", max_rewrites: int = 2, self_rag: bool = True) -> AgenticResult:
    hits = retriever.search(query, ticker=ticker, as_of=as_of, mode=mode, top_k=top_k,
                            section_types=section_types, scope=scope)
    g = grade(query, hits)
    attempts = [{"query": query, "sections": section_types, "grade": g, "n": len(hits)}]
    i = 0
    while self_rag and g != "SUFFICIENT" and i < max_rewrites:
        q2, secs = rewrite(query, i)
        more = retriever.search(q2, ticker=ticker, as_of=as_of, mode=mode, top_k=top_k, section_types=secs)
        seen = {h.chunk.chunk_id for h in hits}
        hits = hits + [h for h in more if h.chunk.chunk_id not in seen]
        g = grade(query, hits)
        attempts.append({"query": q2, "sections": secs, "grade": g, "n": len(more)})
        i += 1
    return AgenticResult(hits[: top_k + 3], g, attempts)
