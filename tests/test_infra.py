"""Storage / vector / tracing backends (PRD v1.2 Epic 1 of infrastructure).

SQLite + in-memory index + JSON tracing run everywhere. The PostgreSQL tests need
`docker compose up -d db`, a bootstrapped database and FINRESEARCH_TEST_PG_URL; otherwise they are skipped.
"""
import os
from datetime import datetime

import pytest

from agent_runtime import ResearchRequest, run_research
from data_platform import db as dbmod
from evaluation_platform import tracing

A = datetime(2025, 12, 31, 15)
PG_URL = os.environ.get("FINRESEARCH_TEST_PG_URL")


def test_default_backend_is_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)
    assert dbmod.storage_backend() == "sqlite"
    assert dbmod.database_url().startswith("sqlite:///")


def test_postgres_backend_switch(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("STORAGE_BACKEND", "postgres")
    assert dbmod.database_url() == dbmod.DEFAULT_PG_URL
    monkeypatch.setenv("DATABASE_URL", "sqlite:///x.db")   # an explicit URL wins
    assert dbmod.storage_backend() == "sqlite"


def test_pgvector_requires_postgres(monkeypatch):
    from rag_engine.vector_store import pg_store_if_enabled

    monkeypatch.setenv("VECTOR_BACKEND", "pgvector")
    assert pg_store_if_enabled(None, 2048) is None      # sqlite database: falls back to the in-memory index


def test_core_tables_and_pit_index_exist():
    names = set(dbmod.Base.metadata.tables)
    assert {"securities", "market_prices", "financial_facts", "data_snapshots", "data_lineage_edges", "source_documents",
            "document_chunks", "evidence_records", "research_runs", "agent_executions", "research_claims",
            "verification_results", "benchmark_tasks", "eval_experiments", "eval_task_results"} <= names
    idx = {i.name for i in dbmod.Base.metadata.tables["financial_facts"].indexes}
    assert "idx_facts_ticker_pit" in idx


@pytest.fixture(scope="module")
def otel_run():
    os.environ["TRACING_BACKEND"] = "otel"
    tracing.clear_otel_spans()
    try:
        rep = run_research(ResearchRequest(ticker="990001.SZ", as_of=A), "B4", persist=False)
        tracing.flush_otel()
        yield rep, tracing.finished_otel_spans()
    finally:
        os.environ.pop("TRACING_BACKEND", None)


def test_otel_span_hierarchy_and_attributes(otel_run):
    rep, spans = otel_run
    root = [s for s in spans if s.name == "research_run"]
    assert len(root) == 1
    r = root[0]
    for k in ("run_id", "ticker", "as_of_date", "snapshot_id", "architecture_tier", "pit_compliant", "latency_ms"):
        assert k in r.attributes, k
    assert r.attributes["ticker"] == "990001.SZ" and r.attributes["pit_compliant"] is True
    ids = {s.context.span_id for s in spans}
    for s in spans:
        if s is not r:
            assert s.parent is not None and s.parent.span_id in ids, f"orphan span {s.name}"
    kinds = {s.name.split(".")[0] + "." + s.name.split(".")[1] if "." in s.name else s.name for s in spans}
    assert {"langgraph.node", "tool.call", "verifier.check"} <= kinds
    assert any(s.name.startswith("llm.completion.") for s in spans) or not rep["totals"]["llm_online"]


def test_otel_matches_json_trace(otel_run):
    rep, spans = otel_run
    nodes = [s for s in spans if s.name.startswith("langgraph.node.")]
    assert len(nodes) == len(rep["trace"])
    assert sorted(s.name.split("langgraph.node.")[1] for s in nodes) == sorted(t["node"] for t in rep["trace"])
    tools = [s for s in spans if s.name.startswith("tool.call.")]
    assert len(tools) == rep["totals"]["tool_calls"]


def test_json_backend_creates_no_spans(monkeypatch):
    monkeypatch.setenv("TRACING_BACKEND", "json")
    tracing.clear_otel_spans()
    run_research(ResearchRequest(ticker="990001.SZ", as_of=A), "B3", persist=False)
    assert tracing.finished_otel_spans() == []


# ---------------------------------------------------------------- PostgreSQL + pgvector
needs_pg = pytest.mark.skipif(not PG_URL, reason="set FINRESEARCH_TEST_PG_URL (docker compose up -d db; bootstrap)")


@needs_pg
def test_pgvector_index_is_complete_and_hnsw():
    from sqlalchemy import text

    e = dbmod.get_engine(PG_URL)
    with e.connect() as c:
        n_child = c.execute(text("select count(*) from document_chunks where level='child'")).scalar()
        n_emb = c.execute(text("select count(*) from document_chunks where level='child' and embedding is not null")).scalar()
        idx = {r[0] for r in c.execute(text("select indexname from pg_indexes"))}
        ext = c.execute(text("select extversion from pg_extension where extname='vector'")).scalar()
    assert n_child > 10000 and n_emb == n_child
    assert {"ix_chunks_embedding_hnsw", "idx_facts_ticker_pit"} <= idx
    assert ext


@needs_pg
def test_pgvector_prefilter_blocks_future_documents():
    from rag_engine.index import get_retriever

    os.environ["VECTOR_BACKEND"] = "pgvector"
    get_retriever.cache_clear()
    r = get_retriever(PG_URL, anchors=True)
    assert r.vector_store is not None
    as_of = datetime(2025, 3, 1)
    for q in ("营业收入 净利润", "应收账款 账龄", "管理层讨论与分析 经营情况"):
        for mode in ("dense", "hybrid"):
            hits = r.search(q, ticker="600519.SH", as_of=as_of, mode=mode, top_k=5)
            assert hits and all(h.chunk.ticker == "600519.SH" and h.chunk.available_at <= as_of for h in hits)
    assert r.search("营业收入", ticker="600519.SH", as_of=datetime(2000, 1, 1), mode="dense") == []
    get_retriever.cache_clear()
    os.environ.pop("VECTOR_BACKEND", None)


@needs_pg
def test_pgvector_matches_memory_recall():
    from rag_engine.index import build_chunks
    from rag_engine.retrieval import Retriever
    from rag_engine.index import get_retriever

    os.environ["VECTOR_BACKEND"] = "pgvector"
    get_retriever.cache_clear()
    pg = get_retriever(PG_URL, anchors=True)
    mem = Retriever(build_chunks(PG_URL))
    for q, t in (("2025年年度报告 合并资产负债表 应收账款", "000333.SZ"), ("2024年半年度 存货 跌价准备", "300750.SZ")):
        a = [h.chunk.chunk_id for h in pg.search(q, ticker=t, as_of=datetime(2026, 5, 1), mode="hybrid", top_k=5)]
        b = [h.chunk.chunk_id for h in mem.search(q, ticker=t, as_of=datetime(2026, 5, 1), mode="hybrid", top_k=5)]
        assert len(set(a) & set(b)) >= 4          # fp16 storage and HNSW may swap the 5th hit
    get_retriever.cache_clear()
    os.environ.pop("VECTOR_BACKEND", None)
