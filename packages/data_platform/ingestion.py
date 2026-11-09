"""Ingestion Pipeline: provider adapters with retry + idempotent upsert into storage."""
from __future__ import annotations

import csv
import logging
import random
import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterator, Optional, TypeVar

from sqlalchemy import delete

from . import quality
from .db import (ROOT, DataSourceRow, FinancialFactRow, MarketPriceRow, SecurityRow, SourceDocumentRow,
                 get_session_factory)
from .registry import DataSourceRegistry
from .schema import FinancialFact, PeriodType, PriceBar, Security, SourceDocument, StatementScope, Unit

log = logging.getLogger(__name__)
T = TypeVar("T")


def with_retry(fn: Callable[[], T], attempts: int = 4, base_delay: float = 0.5) -> T:
    """Exponential backoff with jitter."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - adapters raise anything
            if i == attempts - 1:
                raise
            delay = base_delay * 2 ** i * (1 + random.random() * 0.25)
            log.warning("retry %s/%s after %s: %.2fs", i + 1, attempts, exc, delay)
            time.sleep(delay)
    raise RuntimeError("unreachable")


class GoldenFileProvider:
    """Reads the offline Golden Snapshot CSV + markdown files."""

    def __init__(self, path: Path):
        self.path = path

    def _rows(self, name: str) -> Iterator[dict]:
        with open(self.path / name, encoding="utf-8") as f:
            yield from csv.DictReader(f)

    def securities(self) -> list[Security]:
        return [Security(ticker=r["ticker"], name=r["name"], list_date=date.fromisoformat(r["list_date"]),
                         industry_sw_l1=r["industry_sw_l1"], industry_sw_l2=r["industry_sw_l2"],
                         template=r["template"], is_demo=r["is_demo"] == "true") for r in self._rows("securities.csv")]

    def facts(self, sources: Optional[set[str]] = None) -> tuple[list[FinancialFact], dict[str, bool]]:
        out, reliable = [], {}
        for r in self._rows("financial_facts.csv"):
            if sources and r["source"] not in sources:
                continue
            f = FinancialFact(fact_id=r["fact_id"], ticker=r["ticker"], metric=r["metric"],
                              statement_scope=StatementScope(r["statement_scope"]),
                              period_type=PeriodType(r["period_type"]), value=float(r["value"]), unit=Unit(r["unit"]),
                              period_end=date.fromisoformat(r["period_end"]),
                              published_at=datetime.fromisoformat(r["published_at"]),
                              available_at=datetime.fromisoformat(r["available_at"]),
                              retrieved_at=datetime.fromisoformat(r["retrieved_at"]), source=r["source"],
                              source_document_id=r["source_document_id"] or None, revision_id=int(r["revision_id"]),
                              is_restated=r["is_restated"] == "true")
            out.append(f)
            reliable[f.fact_id] = r["published_at_reliable"] == "true"
        return out, reliable

    def prices(self) -> list[PriceBar]:
        return [PriceBar(ticker=r["ticker"], trade_date=date.fromisoformat(r["trade_date"]), open=float(r["open"]),
                         high=float(r["high"]), low=float(r["low"]), close=float(r["close"]), volume=float(r["volume"]),
                         turnover_rate=float(r["turnover_rate"]), total_mv=float(r["total_mv"]),
                         float_mv=float(r["float_mv"]), available_at=datetime.fromisoformat(r["available_at"]))
                for r in self._rows("market_prices.csv")]

    def documents(self) -> list[SourceDocument]:
        return [SourceDocument(document_id=r["document_id"], ticker=r["ticker"], title=r["title"],
                               doc_type=r["doc_type"],
                               period_end=date.fromisoformat(r["period_end"]) if r["period_end"] else None,
                               published_at=datetime.fromisoformat(r["published_at"]),
                               available_at=datetime.fromisoformat(r["available_at"]),
                               content_hash=r["content_hash"], path=r["path"], version=int(r["version"]))
                for r in self._rows("documents.csv")]


class IngestionPipeline:
    def __init__(self, db_url: Optional[str] = None, registry: Optional[DataSourceRegistry] = None):
        self.Session = get_session_factory(db_url)
        self.registry = registry or DataSourceRegistry()

    def register_sources(self) -> None:
        with self.Session() as s:
            for cfg in self.registry.sources.values():
                s.merge(DataSourceRow(source_id=cfg.source_id, kind=cfg.kind, priority=cfg.priority,
                                      rate_limit_per_min=cfg.rate_limit_per_min, enabled=cfg.enabled,
                                      health=self.registry.health[cfg.source_id], config=cfg.model_dump()))
            s.commit()

    def ingest_golden(self, path: Optional[Path] = None) -> dict:
        path = path or ROOT / "benchmarks" / "golden_snapshot"
        provider = GoldenFileProvider(path)
        self.register_sources()
        secs = with_retry(provider.securities)
        facts, reliable = with_retry(provider.facts)
        prices = with_retry(provider.prices)
        docs = with_retry(provider.documents)
        issues = quality.run_all(facts, reliable)
        rejected = {i.entity_id for i in issues if i.severity == "ERROR"}
        unsafe = {i.entity_id for i in issues if i.rule == "pit_reliability"}
        with self.Session() as s:
            for sec in secs:
                s.merge(SecurityRow(**sec.model_dump()))
            tickers = [sec.ticker for sec in secs]
            # idempotent: replace this provider's slice
            s.execute(delete(MarketPriceRow).where(MarketPriceRow.ticker.in_(tickers)))
            s.execute(delete(FinancialFactRow).where(FinancialFactRow.ticker.in_(tickers)))
            s.flush()
            s.add_all(MarketPriceRow(**p.model_dump()) for p in prices)
            for f in facts:
                if f.fact_id in rejected:
                    continue
                d = f.model_dump()
                d["statement_scope"] = f.statement_scope.value
                d["period_type"] = f.period_type.value
                d["unit"] = f.unit.value
                d["pit_unsafe"] = f.fact_id in unsafe
                s.add(FinancialFactRow(**d))
            for doc in docs:
                s.merge(SourceDocumentRow(**doc.model_dump()))
            s.commit()
        for sid in ("demo_filing", "demo_vendor_b"):
            self.registry.mark(sid, "healthy")
        self.register_sources()
        return {"securities": len(secs), "facts": len(facts) - len(rejected), "prices": len(prices),
                "documents": len(docs), "quality_issues": [i.model_dump() for i in issues], "rejected": len(rejected)}


class AkshareProvider:
    """Live adapter (needs network + `pip install akshare`). Maps AKShare daily bars
    into PriceBar. Statement ingestion from AKShare lacks reliable published_at,
    so those facts are written with pit_unsafe=True until joined with 巨潮 filing dates."""

    def __init__(self, registry: DataSourceRegistry):
        self.registry = registry

    def prices(self, ticker: str, start: str, end: str, shares: float) -> list[PriceBar]:
        import akshare as ak  # optional dependency

        self.registry.acquire("akshare")
        df = with_retry(lambda: ak.stock_zh_a_hist(symbol=ticker[:6], period="daily", start_date=start,
                                                   end_date=end, adjust=""))
        bars = []
        for _, r in df.iterrows():
            d = date.fromisoformat(str(r["日期"])[:10])
            bars.append(PriceBar(ticker=ticker, trade_date=d, open=r["开盘"], high=r["最高"], low=r["最低"],
                                 close=r["收盘"], volume=r["成交量"], turnover_rate=r["换手率"],
                                 total_mv=r["收盘"] * shares, float_mv=r["收盘"] * shares,
                                 available_at=datetime(d.year, d.month, d.day, 15, 0)))
        return bars
