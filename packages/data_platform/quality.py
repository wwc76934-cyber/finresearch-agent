"""Data Quality Engine: deterministic rules run at ingestion and at query time."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterable

from .schema import FinancialFact, QualityIssue

NON_NEGATIVE = {"operating_revenue", "operating_cost", "total_assets", "accounts_receivable", "inventory",
                "goodwill", "cash", "shares_outstanding", "total_liabilities"}


def check_uniqueness(facts: Iterable[FinancialFact]) -> list[QualityIssue]:
    seen: dict[tuple, str] = {}
    issues = []
    for f in facts:
        key = (f.ticker, f.metric, f.statement_scope, f.period_type, f.period_end, f.source, f.revision_id)
        if key in seen:
            issues.append(QualityIssue(rule="uniqueness", severity="ERROR", entity_id=f.fact_id,
                                       message=f"duplicate of {seen[key]}"))
        seen[key] = f.fact_id
    return issues


def check_non_negative(facts: Iterable[FinancialFact]) -> list[QualityIssue]:
    return [QualityIssue(rule="non_negative", severity="ERROR", entity_id=f.fact_id,
                         message=f"{f.metric} must be >= 0, got {f.value}")
            for f in facts if f.metric in NON_NEGATIVE and f.value < 0]


def check_balance_identity(facts: Iterable[FinancialFact], tol: float = 1e-3) -> list[QualityIssue]:
    """资产 = 负债 + 所有者权益, per (ticker, period_end, source, revision)."""
    groups: dict[tuple, dict[str, FinancialFact]] = defaultdict(dict)
    for f in facts:
        if f.metric in ("total_assets", "total_liabilities", "total_equity") and f.statement_scope.value == "consolidated":
            groups[(f.ticker, f.period_end, f.source, f.revision_id)][f.metric] = f
    issues = []
    for key, g in groups.items():
        if len(g) < 3:
            continue
        a, l, e = g["total_assets"].value, g["total_liabilities"].value, g["total_equity"].value
        if abs(a - (l + e)) > tol * abs(a):
            issues.append(QualityIssue(rule="balance_identity", severity="ERROR", entity_id=g["total_assets"].fact_id,
                                       message=f"assets {a:.0f} != liabilities+equity {l + e:.0f}"))
    return issues


def check_unit_jump(facts: Iterable[FinancialFact], ratio: float = 50.0) -> list[QualityIssue]:
    """Flags a value that moves by >50x between consecutive periods: usually a 元/万元 mix-up."""
    series: dict[tuple, list[FinancialFact]] = defaultdict(list)
    for f in facts:
        series[(f.ticker, f.metric, f.statement_scope, f.period_type, f.source)].append(f)
    issues = []
    for s in series.values():
        s.sort(key=lambda x: (x.period_end, x.revision_id))
        for prev, cur in zip(s, s[1:]):
            if prev.value and cur.value and prev.value * cur.value > 0:
                r = abs(cur.value / prev.value)
                if r > ratio or r < 1 / ratio:
                    issues.append(QualityIssue(rule="unit_jump", severity="WARN", entity_id=cur.fact_id,
                                               message=f"{cur.metric} changed {r:.1f}x vs {prev.fact_id}"))
    return issues


def check_pit_reliability(facts: Iterable[FinancialFact], reliable: dict[str, bool]) -> list[QualityIssue]:
    return [QualityIssue(rule="pit_reliability", severity="WARN", entity_id=f.fact_id,
                         message="published_at not reliable -> pit_unsafe=true")
            for f in facts if not reliable.get(f.fact_id, True)]


def run_all(facts: list[FinancialFact], reliable: dict[str, bool] | None = None) -> list[QualityIssue]:
    issues = []
    issues += check_uniqueness(facts)
    issues += check_non_negative(facts)
    issues += check_balance_identity(facts)
    issues += check_unit_jump(facts)
    issues += check_pit_reliability(facts, reliable or {})
    return issues


def assert_pit(records_available_at: Iterable[tuple[str, datetime]], as_of: datetime) -> list[str]:
    """Query-time guard: returns ids that are visible only after as_of."""
    return [rid for rid, av in records_available_at if av > as_of]
