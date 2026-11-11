"""SQLAlchemy 2.0 storage: the 16 core tables of PRD section 7.3.

DATABASE_URL defaults to a local SQLite file so the project runs with no
services; docker-compose points it at PostgreSQL 16.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, Index, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

ROOT = Path(__file__).resolve().parents[2]


class Base(DeclarativeBase):
    pass


# ---------------- data platform (6) ----------------
class DataSourceRow(Base):
    __tablename__ = "data_sources"
    source_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))
    priority: Mapped[int] = mapped_column(Integer)
    rate_limit_per_min: Mapped[int] = mapped_column(Integer, default=60)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    health: Mapped[str] = mapped_column(String(16), default="unknown")
    config: Mapped[dict] = mapped_column(JSON, default=dict)


class SecurityRow(Base):
    __tablename__ = "securities"
    ticker: Mapped[str] = mapped_column(String(12), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    list_date: Mapped[date] = mapped_column(Date)
    delisted: Mapped[bool] = mapped_column(Boolean, default=False)
    industry_sw_l1: Mapped[str] = mapped_column(String(32))
    industry_sw_l2: Mapped[str] = mapped_column(String(32))
    template: Mapped[str] = mapped_column(String(16))
    is_demo: Mapped[bool] = mapped_column(Boolean, default=True)


class MarketPriceRow(Base):
    __tablename__ = "market_prices"
    ticker: Mapped[str] = mapped_column(String(12), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    adjust: Mapped[str] = mapped_column(String(4), primary_key=True, default="none")
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    turnover_rate: Mapped[float] = mapped_column(Float)
    total_mv: Mapped[float] = mapped_column(Float)
    float_mv: Mapped[float] = mapped_column(Float)
    available_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class FinancialFactRow(Base):
    __tablename__ = "financial_facts"
    fact_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(12))
    metric: Mapped[str] = mapped_column(String(64))
    statement_scope: Mapped[str] = mapped_column(String(16))
    period_type: Mapped[str] = mapped_column(String(24))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(12))
    period_end: Mapped[date] = mapped_column(Date)
    published_at: Mapped[datetime] = mapped_column(DateTime)
    available_at: Mapped[datetime] = mapped_column(DateTime)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime)
    source: Mapped[str] = mapped_column(String(48))
    source_document_id: Mapped[Optional[str]] = mapped_column(String(96), nullable=True)
    revision_id: Mapped[int] = mapped_column(Integer, default=1)
    is_restated: Mapped[bool] = mapped_column(Boolean, default=False)
    pit_unsafe: Mapped[bool] = mapped_column(Boolean, default=False)
    __table_args__ = (Index("ix_facts_pit", "ticker", "metric", "period_end", "available_at"),
                      # PRD v1.2 3.1: the point-in-time lookup "latest revision visible at t"
                      Index("idx_facts_ticker_pit", "ticker", "metric", "available_at", "revision_id"))


class DataSnapshotRow(Base):
    __tablename__ = "data_snapshots"
    snapshot_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    as_of: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    manifest: Mapped[dict] = mapped_column(JSON)


class LineageEdgeRow(Base):
    __tablename__ = "data_lineage_edges"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parent_entity_id: Mapped[str] = mapped_column(String(160), index=True)
    child_entity_id: Mapped[str] = mapped_column(String(160), index=True)
    parent_type: Mapped[str] = mapped_column(String(24))
    child_type: Mapped[str] = mapped_column(String(24))
    transform_tool: Mapped[str] = mapped_column(String(64))
    formula_version: Mapped[str] = mapped_column(String(16))
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)


# ---------------- RAG + evidence (3) ----------------
class SourceDocumentRow(Base):
    __tablename__ = "source_documents"
    document_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(12), index=True)
    title: Mapped[str] = mapped_column(String(200))
    doc_type: Mapped[str] = mapped_column(String(24))
    period_end: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    published_at: Mapped[datetime] = mapped_column(DateTime)
    available_at: Mapped[datetime] = mapped_column(DateTime)
    content_hash: Mapped[str] = mapped_column(String(64))
    path: Mapped[str] = mapped_column(String(300))
    version: Mapped[int] = mapped_column(Integer, default=1)


class DocumentChunkRow(Base):
    __tablename__ = "document_chunks"
    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(96), index=True)
    ticker: Mapped[str] = mapped_column(String(12), index=True)
    parent_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    level: Mapped[str] = mapped_column(String(8))
    section_path: Mapped[str] = mapped_column(String(200))
    section_type: Mapped[str] = mapped_column(String(16))
    is_table: Mapped[bool] = mapped_column(Boolean)
    table_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    statement_scope: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    unit: Mapped[Optional[str]] = mapped_column(String(12), nullable=True)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    available_at: Mapped[datetime] = mapped_column(DateTime)


class EvidenceRecordRow(Base):
    __tablename__ = "evidence_records"
    evidence_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(16))
    source_ref: Mapped[str] = mapped_column(String(160))
    summary: Mapped[str] = mapped_column(Text)
    excerpt: Mapped[str] = mapped_column(Text)
    available_at: Mapped[datetime] = mapped_column(DateTime)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)


# ---------------- runs + delivery (4) ----------------
class ResearchRunRow(Base):
    __tablename__ = "research_runs"
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String(80))
    ticker: Mapped[str] = mapped_column(String(12), index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime)
    request: Mapped[dict] = mapped_column(JSON)
    architecture: Mapped[str] = mapped_column(String(48))
    status: Mapped[str] = mapped_column(String(16))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_cny: Mapped[float] = mapped_column(Float, default=0.0)
    report: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class AgentExecutionRow(Base):
    __tablename__ = "agent_executions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    node: Mapped[str] = mapped_column(String(48))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    latency_ms: Mapped[float] = mapped_column(Float)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    tool_calls: Mapped[list] = mapped_column(JSON, default=list)
    anomaly_flags: Mapped[list] = mapped_column(JSON, default=list)
    summary: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="ok")


class ResearchClaimRow(Base):
    __tablename__ = "research_claims"
    claim_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    section: Mapped[str] = mapped_column(String(32))
    claim_type: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text)
    evidence_ids: Mapped[list] = mapped_column(JSON, default=list)
    metric_ids: Mapped[list] = mapped_column(JSON, default=list)
    verification_status: Mapped[str] = mapped_column(String(24))


class VerificationResultRow(Base):
    __tablename__ = "verification_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    claim_id: Mapped[str] = mapped_column(String(64))
    tier: Mapped[int] = mapped_column(Integer)
    passed: Mapped[bool] = mapped_column(Boolean)
    error_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    detail: Mapped[str] = mapped_column(Text, default="")
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    circuit_broken: Mapped[bool] = mapped_column(Boolean, default=False)


# ---------------- evaluation (3) ----------------
class BenchmarkTaskRow(Base):
    __tablename__ = "benchmark_tasks"
    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset: Mapped[str] = mapped_column(String(32))
    version: Mapped[str] = mapped_column(String(16))
    snapshot_id: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    trap_type: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict] = mapped_column(JSON)


class EvalExperimentRow(Base):
    __tablename__ = "eval_experiments"
    experiment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    architecture: Mapped[str] = mapped_column(String(48))
    config: Mapped[dict] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)


class EvalTaskResultRow(Base):
    __tablename__ = "eval_task_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(64), index=True)
    task_id: Mapped[str] = mapped_column(String(64))
    run_id: Mapped[str] = mapped_column(String(64))
    scores: Mapped[dict] = mapped_column(JSON)
    judge_notes: Mapped[str] = mapped_column(Text, default="")
    error_category: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)


DEFAULT_PG_URL = "postgresql+psycopg://finresearch:finresearch@localhost:5432/finresearch"


def storage_backend() -> str:
    """STORAGE_BACKEND=postgres|sqlite. An explicit DATABASE_URL wins; otherwise sqlite (no services needed)."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return "sqlite" if url.startswith("sqlite") else "postgres"
    return os.environ.get("STORAGE_BACKEND", "sqlite").lower()


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    if storage_backend() == "postgres":
        return DEFAULT_PG_URL
    path = ROOT / "data" / "finresearch.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


@lru_cache(maxsize=4)
def get_engine(url: Optional[str] = None):
    url = url or database_url()
    kwargs: dict[str, Any] = {"future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs.update(pool_pre_ping=True, pool_size=8, max_overflow=8)
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(url: Optional[str] = None):
    return sessionmaker(bind=get_engine(url), expire_on_commit=False, future=True)
