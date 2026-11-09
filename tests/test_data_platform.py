from datetime import date, datetime

import pytest

from data_platform import FinancialCalculator, PITDataService, PITViolation, create_snapshot
from data_platform.quality import check_balance_identity, check_uniqueness, check_unit_jump
from data_platform.schema import FinancialFact, PeriodType, StatementScope
from evaluation_platform.datasets import Oracle

A = datetime(2025, 12, 31, 15)


def fact(**kw):
    base = dict(fact_id="f1", ticker="990001.SZ", metric="total_assets", statement_scope=StatementScope.CONSOLIDATED,
                period_type=PeriodType.INSTANT, value=100.0, period_end=date(2025, 9, 30),
                published_at=datetime(2025, 10, 28, 20), available_at=datetime(2025, 10, 29, 9, 30),
                retrieved_at=datetime(2026, 1, 1), source="t")
    base.update(kw)
    return FinancialFact(**base)


def test_contract_rejects_period_end_as_publication_date():
    with pytest.raises(ValueError):
        fact(published_at=datetime(2025, 9, 1), available_at=datetime(2025, 9, 2))
    with pytest.raises(ValueError):
        fact(available_at=datetime(2025, 10, 28, 19))


def test_quality_rules_catch_bad_rows():
    a = fact(fact_id="a", metric="total_assets", value=100)
    l = fact(fact_id="l", metric="total_liabilities", value=60)
    e = fact(fact_id="e", metric="total_equity", value=30)
    assert check_balance_identity([a, l, e])[0].rule == "balance_identity"
    assert check_uniqueness([a, fact(fact_id="a2", metric="total_assets")])
    jump = fact(fact_id="j", period_end=date(2025, 12, 31), published_at=datetime(2026, 3, 1), available_at=datetime(2026, 3, 2), value=100 * 1e4)
    assert check_unit_jump([a, jump])[0].rule == "unit_jump"


def test_pit_hides_report_published_after_close_until_next_open():
    svc = PITDataService(datetime(2025, 10, 28, 21, 0), "t")
    assert svc.latest_period("990001.SZ") == date(2025, 6, 30)
    svc2 = PITDataService(datetime(2025, 10, 29, 9, 30), "t")
    assert svc2.latest_period("990001.SZ") == date(2025, 9, 30)


def test_restatement_visibility_follows_as_of():
    q3 = date(2025, 9, 30)
    before = PITDataService(datetime(2025, 11, 19, 15), "t").fact("990003.SH", "net_profit", q3)
    after = PITDataService(datetime(2025, 11, 25, 15), "t").fact("990003.SH", "net_profit", q3)
    assert before.revision_id == 1 and after.revision_id == 2
    assert after.value == pytest.approx(before.value - 0.9e9)


def test_pit_unsafe_facts_are_excluded():
    svc = PITDataService(A, "t")
    assert svc.fact("990004.SH", "operating_revenue", date(2022, 3, 31)) is None
    assert PITDataService(A, "t", allow_pit_unsafe=True).fact("990004.SH", "operating_revenue", date(2022, 3, 31))


def test_guard_raises_on_future_record():
    svc = PITDataService(A, "t")
    with pytest.raises(PITViolation):
        svc._guard(datetime(2026, 1, 1), "x")


def test_source_priority_and_conflict_recorded():
    svc = PITDataService(datetime(2025, 6, 30, 15), "t")
    f = svc.fact("990002.SZ", "operating_revenue", date(2025, 3, 31))
    assert f.source == "demo_filing"
    assert svc.conflicts and svc.conflicts[0]["other"].endswith("ndor_b_v1")


def test_snapshot_is_reproducible():
    a = create_snapshot(A, ["990001.SZ"])
    b = create_snapshot(A, ["990001.SZ"])
    assert a.snapshot_id == b.snapshot_id and a.manifest_hash == b.manifest_hash
    assert a.watermarks["financial_facts"] <= A.isoformat()


def test_calculators_match_independent_oracle():
    o = Oracle()
    calc = FinancialCalculator(PITDataService(A, "t"))
    q3 = date(2025, 9, 30)
    for t in ["990001.SZ", "990002.SZ", "990004.SH"]:
        sq = calc.single_quarter(t, "operating_revenue", q3)
        assert sq.value == pytest.approx(o.single_q(t, "operating_revenue", q3, A)[0])
        pe = calc.valuation(t)["pe_ttm"]
        assert pe.value == pytest.approx(o.pe_ttm(t, A)[0], rel=1e-9)


def test_restatement_breaks_quarter_split():
    calc = FinancialCalculator(PITDataService(datetime(2025, 6, 30, 15), "t"))
    r = calc.single_quarter("990002.SZ", "operating_revenue", date(2024, 9, 30))
    assert r.status == "RESTATEMENT_BREAKS_SPLIT" and r.value is None


def test_loss_maker_pe_is_not_reported_as_a_number():
    pe = FinancialCalculator(PITDataService(A, "t")).valuation("990005.SZ")["pe_ttm"]
    assert pe.status == "NEGATIVE_EARNINGS_NA" and pe.value is None


@pytest.mark.parametrize("ticker,codes", [("990001.SZ", {"PROFIT_OCF_DIVERGENCE", "AR_OUTPACE_REVENUE"}),
                                          ("990002.SZ", {"GROSS_MARGIN_SHIFT"}),
                                          ("990003.SH", {"LARGE_IMPAIRMENT", "INVENTORY_DAYS_DETERIORATION"}),
                                          ("990005.SZ", {"INTEREST_DEBT_JUMP"})])
def test_designed_anomalies_detected(ticker, codes):
    calc = FinancialCalculator(PITDataService(A, "t"))
    got = {f.code for f in calc.detect_anomalies(ticker, date(2025, 9, 30))}
    assert codes <= got


def test_healthy_company_has_no_financial_anomaly():
    calc = FinancialCalculator(PITDataService(A, "t"))
    assert not calc.detect_anomalies("990004.SH", date(2025, 9, 30))
