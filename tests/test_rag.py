from datetime import datetime

from rag_engine.agentic import agentic_search, route
from rag_engine.chunking import chunk_document
from rag_engine.index import get_retriever

DOC = """# 演示公司 2025年第三季度报告

<!-- page 3 -->
## 第四节 财务报表
### 合并利润表（单位：万元）
| 项目 | 本期金额 |
| --- | --- |
| 营业收入 | 100.00 |
<!-- page 4 -->
| 归属于母公司股东的净利润 | 10.00 |
"""


def test_cross_page_table_is_stitched_with_anchor():
    chunks = chunk_document("d1", "990001.SZ", "2025Q3", "q3_report", DOC, datetime(2025, 10, 29, 9, 30))
    parent = next(c for c in chunks if c.level == "parent" and c.is_table)
    assert "营业收入" in parent.text and "归属于母公司股东的净利润" in parent.text
    assert (parent.page_start, parent.page_end) == (3, 4)
    assert parent.text.startswith("[990001.SZ | 2025Q3 |") and "合并" in parent.text.splitlines()[0] and "万元" in parent.text.splitlines()[0]
    assert parent.statement_scope == "consolidated"


def test_retrieval_never_returns_future_documents():
    r = get_retriever()
    as_of = datetime(2025, 12, 31, 15)
    hits = r.search("立案告知书 证监会", ticker="990001.SZ", as_of=as_of, top_k=10)
    assert all(h.chunk.available_at <= as_of for h in hits)
    assert not any("20260120" in h.chunk.document_id for h in hits)


def test_router_sends_numbers_to_structured_store():
    assert route("2024年研发费用率是多少") == "structured"
    assert route("为什么研发费用率同比上升") == "hybrid"


def test_self_rag_rewrites_when_only_tables_found():
    r = get_retriever()
    res = agentic_search(r, "经营现金流 下降 原因", "990001.SZ", datetime(2025, 12, 31, 15), section_types=["STATEMENTS"])
    assert len(res.attempts) >= 2
    assert res.grade == "SUFFICIENT"
