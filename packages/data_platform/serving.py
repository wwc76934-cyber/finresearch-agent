"""Data Serving: PIT-safe query interface. Every query is bound to (as_of, snapshot_id)
and the time predicate is injected in SQL, so a caller cannot forget it."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from typing import Optional

from sqlalchemy import select

from .db import FinancialFactRow, MarketPriceRow, SecurityRow, SourceDocumentRow, get_session_factory
from .registry import DataSourceRegistry


class PITViolation(Exception):
    pass


def period_label(d: date) -> str:
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def label_to_end(label: str) -> date:
    y, q = int(label[:4]), int(label[-1])
    return {1: date(y, 3, 31), 2: date(y, 6, 30), 3: date(y, 9, 30), 4: date(y, 12, 31)}[q]


class PITDataService:
    def __init__(self, as_of: datetime, snapshot_id: str, db_url: Optional[str] = None,
                 registry: Optional[DataSourceRegistry] = None, allow_pit_unsafe: bool = False):
        self.as_of = as_of
        self.snapshot_id = snapshot_id
        self.Session = get_session_factory(db_url)
        self.registry = registry or DataSourceRegistry()
        self.allow_pit_unsafe = allow_pit_unsafe
        self._facts: dict[str, list[FinancialFactRow]] = {}
        self._prices: dict[str, list[MarketPriceRow]] = {}
        self.conflicts: list[dict] = []
        self.query_log: list[dict] = []

    # ---------- guard ----------
    def _guard(self, available_at: datetime, entity: str) -> None:
        if available_at > self.as_of:
            raise PITViolation(f"{entity} available_at={available_at} > as_of={self.as_of}")

    # ---------- securities ----------
    def securities(self) -> list[SecurityRow]:
        with self.Session() as s:
            return list(s.scalars(select(SecurityRow)))

    def resolve_security(self, query: str) -> Optional[SecurityRow]:
        q = query.strip().upper()
        for sec in self.securities():
            if q in (sec.ticker, sec.ticker[:6]) or query.strip() in sec.name or sec.name in query:
                return sec
        return None

    def industry_peers(self, ticker: str) -> list[SecurityRow]:
        secs = self.securities()
        me = next(s for s in secs if s.ticker == ticker)
        same_l2 = [s for s in secs if s.industry_sw_l2 == me.industry_sw_l2 and s.ticker != ticker]
        return same_l2 or [s for s in secs if s.industry_sw_l1 == me.industry_sw_l1 and s.ticker != ticker]

    # ---------- facts ----------
    def _load_facts(self, ticker: str) -> list[FinancialFactRow]:
        if ticker not in self._facts:
            with self.Session() as s:
                stmt = select(FinancialFactRow).where(FinancialFactRow.ticker == ticker,
                                                      FinancialFactRow.available_at <= self.as_of)
                if not self.allow_pit_unsafe:
                    stmt = stmt.where(FinancialFactRow.pit_unsafe.is_(False))
                self._facts[ticker] = list(s.scalars(stmt))
        return self._facts[ticker]

    def fact(self, ticker: str, metric: str, period_end: date, scope: str = "consolidated",
             at: Optional[datetime] = None) -> Optional[FinancialFactRow]:
        """Latest visible revision from the highest-priority source. Records cross-source conflicts."""
        at = min(at or self.as_of, self.as_of)
        cands = [f for f in self._load_facts(ticker)
                 if f.metric == metric and f.period_end == period_end and f.statement_scope == scope
                 and f.available_at <= at]
        if not cands:
            return None
        by_source: dict[str, FinancialFactRow] = {}
        for f in cands:
            cur = by_source.get(f.source)
            if cur is None or f.revision_id > cur.revision_id:
                by_source[f.source] = f
        ranked = sorted(by_source.values(), key=lambda f: self.registry.priority(f.source))
        best = ranked[0]
        for other in ranked[1:]:
            if best.value and abs(other.value / best.value - 1) > 0.005:
                c = {"ticker": ticker, "metric": metric, "period": period_label(period_end),
                     "chosen": best.fact_id, "chosen_value": best.value, "other": other.fact_id,
                     "other_value": other.value, "rule": "registry_priority"}
                if c not in self.conflicts:
                    self.conflicts.append(c)
        self._guard(best.available_at, best.fact_id)
        self.query_log.append({"fact_id": best.fact_id, "at": at.isoformat()})
        return best

    def revisions(self, ticker: str, metric: str, period_end: date) -> list[FinancialFactRow]:
        return sorted([f for f in self._load_facts(ticker) if f.metric == metric and f.period_end == period_end
                       and f.statement_scope == "consolidated"], key=lambda f: (f.source, f.revision_id))

    def visible_periods(self, ticker: str, at: Optional[datetime] = None) -> list[date]:
        at = min(at or self.as_of, self.as_of)
        return sorted({f.period_end for f in self._load_facts(ticker) if f.available_at <= at})

    def latest_period(self, ticker: str, at: Optional[datetime] = None) -> Optional[date]:
        p = self.visible_periods(ticker, at)
        return p[-1] if p else None

    def statements(self, ticker: str, period_end: date, scope: str = "consolidated") -> dict[str, FinancialFactRow]:
        metrics = {f.metric for f in self._load_facts(ticker) if f.period_end == period_end and f.statement_scope == scope}
        out = {}
        for m in sorted(metrics):
            f = self.fact(ticker, m, period_end, scope)
            if f:
                out[m] = f
        return out

    # ---------- prices ----------
    def prices(self, ticker: str) -> list[MarketPriceRow]:
        if ticker not in self._prices:
            with self.Session() as s:
                stmt = select(MarketPriceRow).where(MarketPriceRow.ticker == ticker,
                                                    MarketPriceRow.available_at <= self.as_of).order_by(MarketPriceRow.trade_date)
                self._prices[ticker] = list(s.scalars(stmt))
        return self._prices[ticker]

    def price_on_or_before(self, ticker: str, d: date) -> Optional[MarketPriceRow]:
        best = None
        for p in self.prices(ticker):
            if p.trade_date <= d:
                best = p
            else:
                break
        return best

    # ---------- documents ----------
    def documents(self, ticker: str, doc_type: Optional[str] = None) -> list[SourceDocumentRow]:
        with self.Session() as s:
            stmt = select(SourceDocumentRow).where(SourceDocumentRow.ticker == ticker,
                                                   SourceDocumentRow.available_at <= self.as_of)
            if doc_type:
                stmt = stmt.where(SourceDocumentRow.doc_type == doc_type)
            rows = list(s.scalars(stmt.order_by(SourceDocumentRow.published_at)))
        for r in rows:
            self._guard(r.available_at, r.document_id)
        return rows

    def coverage(self, ticker: str) -> dict:
        periods = self.visible_periods(ticker)
        return {"ticker": ticker, "periods": [period_label(p) for p in periods],
                "latest_period": period_label(periods[-1]) if periods else None,
                "price_days": len(self.prices(ticker)), "documents": len(self.documents(ticker)),
                "pit_unsafe_excluded": not self.allow_pit_unsafe}


def group_by(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out
