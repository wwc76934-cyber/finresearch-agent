"""Unified Financial Schema: frozen Pydantic v2 contracts for the data platform.

Every record that crosses a module boundary is one of these models. The three
timestamps (period_end / published_at / available_at) are mandatory on facts.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StatementScope(str, Enum):
    CONSOLIDATED = "consolidated"
    PARENT = "parent"


class PeriodType(str, Enum):
    CUMULATIVE_Q1 = "cumulative_q1"
    CUMULATIVE_H1 = "cumulative_h1"
    CUMULATIVE_Q3 = "cumulative_q3"
    ANNUAL = "annual"
    SINGLE_QUARTER = "single_quarter"
    INSTANT = "instant"  # balance-sheet point in time


class Unit(str, Enum):
    CNY = "CNY"
    CNY_10K = "CNY_10K"      # 万元
    CNY_100M = "CNY_100M"    # 亿元
    SHARES = "SHARES"
    RATIO = "RATIO"


UNIT_MULTIPLIER = {Unit.CNY: 1.0, Unit.CNY_10K: 1e4, Unit.CNY_100M: 1e8, Unit.SHARES: 1.0, Unit.RATIO: 1.0}


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Security(Contract):
    ticker: str = Field(pattern=r"^[0-9A-Z]{6}\.(SZ|SH|BJ)$")
    name: str
    list_date: date
    delisted: bool = False
    industry_sw_l1: str
    industry_sw_l2: str
    template: Literal["manufacturing", "consumer", "bank", "insurance", "broker"] = "manufacturing"
    is_demo: bool = True


class FinancialFact(Contract):
    fact_id: str
    ticker: str
    metric: str
    statement_scope: StatementScope
    period_type: PeriodType
    value: float
    unit: Unit = Unit.CNY
    period_end: date
    published_at: datetime
    available_at: datetime
    retrieved_at: datetime
    source: str
    source_document_id: Optional[str] = None
    revision_id: int = 1
    is_restated: bool = False
    pit_unsafe: bool = False

    @model_validator(mode="after")
    def _check_time_order(self) -> "FinancialFact":
        if self.published_at.date() < self.period_end:
            raise ValueError(f"{self.fact_id}: published_at before period_end")
        if self.available_at < self.published_at:
            raise ValueError(f"{self.fact_id}: available_at before published_at")
        return self

    def value_in_cny(self) -> float:
        return self.value * UNIT_MULTIPLIER[self.unit]


class PriceBar(Contract):
    ticker: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover_rate: float
    total_mv: float
    float_mv: float
    adjust: Literal["none", "qfq", "hfq"] = "none"
    available_at: datetime


class SourceDocument(Contract):
    document_id: str
    ticker: str
    title: str
    doc_type: Literal["annual_report", "semi_annual_report", "q1_report", "q3_report", "announcement"]
    period_end: Optional[date] = None
    published_at: datetime
    available_at: datetime
    content_hash: str
    path: str
    version: int = 1


class DocumentChunk(Contract):
    chunk_id: str
    document_id: str
    ticker: str
    parent_id: Optional[str]
    level: Literal["parent", "child"]
    section_path: str
    section_type: Literal["MDA", "NOTES", "RISK_FACTORS", "STATEMENTS", "ANNOUNCEMENT", "OTHER"]
    is_table: bool
    table_name: Optional[str] = None
    statement_scope: Optional[StatementScope] = None
    unit: Optional[str] = None
    page_start: int
    page_end: int
    text: str
    available_at: datetime


class MetricRecord(Contract):
    """A deterministic calculator output. LLMs may cite it but never compute it."""
    metric_id: str
    ticker: str
    metric: str
    value: Optional[float]
    unit: str
    status: Literal["OK", "NEGATIVE_EARNINGS_NA", "ZERO_DENOMINATOR_NA", "NEGATIVE_EQUITY_NA",
                    "INSUFFICIENT_DATA", "RESTATEMENT_BREAKS_SPLIT"] = "OK"
    period: str
    statement_scope: StatementScope = StatementScope.CONSOLIDATED
    formula: str
    formula_version: str = "v1"
    input_ids: list[str]
    available_at: datetime
    as_of: datetime
    extra: dict[str, Any] = Field(default_factory=dict)


class SnapshotManifest(Contract):
    snapshot_id: str
    as_of: datetime
    created_at: datetime
    tickers: list[str]
    watermarks: dict[str, str]          # table -> max(available_at) visible
    row_counts: dict[str, int]
    file_hashes: dict[str, str]         # source file -> sha256
    source_versions: dict[str, str]
    manifest_hash: str


class LineageEdge(Contract):
    parent_entity_id: str
    child_entity_id: str
    parent_type: str
    child_type: str
    transform_tool: str
    formula_version: str = "v1"
    run_id: Optional[str] = None


class QualityIssue(Contract):
    rule: str
    severity: Literal["ERROR", "WARN"]
    entity_id: str
    message: str
