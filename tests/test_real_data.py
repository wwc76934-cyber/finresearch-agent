"""Tests on the real A-share snapshot (skipped when benchmarks/real_snapshot is not built)."""
from datetime import date, datetime
from pathlib import Path

import pytest

from agent_runtime import ResearchRequest, run_research
from data_platform import FinancialCalculator, PITDataService
from rag_engine.pdf_parser import _fix_number, row_name

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "benchmarks" / "real_snapshot" / "financial_facts.csv").exists(),
                                reason="real snapshot not built")
A = datetime(2025, 12, 31, 15)


def test_report_published_only_from_disclosure_day():
    # 宁德时代 2025 Q3 report: disclosed 2025-10-21
    assert PITDataService(datetime(2025, 10, 20, 15), "t").latest_period("300750.SZ") == date(2025, 6, 30)
    assert PITDataService(datetime(2025, 10, 21, 10), "t").latest_period("300750.SZ") == date(2025, 9, 30)


def test_restated_operating_cost_is_versioned():
    # 2024 H1 operating cost was restated in 2025 (warranty costs moved into cost of sales)
    pe = date(2024, 6, 30)
    before = PITDataService(datetime(2025, 1, 31, 15), "t").fact("300750.SZ", "operating_cost", pe)
    after = PITDataService(datetime(2025, 9, 30, 15), "t").fact("300750.SZ", "operating_cost", pe)
    assert before.revision_id == 1 and after.revision_id == 2
    assert after.value > before.value * 1.02


def test_ev_ebitda_and_percentiles():
    v = FinancialCalculator(PITDataService(A, "t")).valuation("300750.SZ")
    assert v["ev_ebitda"].status == "OK" and 5 < v["ev_ebitda"].value < 60
    assert v["pe_percentile_3y"].status == "OK"
    assert v["pe_percentile_5y"].status == "INSUFFICIENT_DATA"   # prices start 2021-12: < 48 months


def test_qoq_and_loss_maker_pe():
    calc = FinancialCalculator(PITDataService(A, "t"))
    g = calc.growth("300750.SZ", date(2025, 9, 30))
    assert g["revenue_qoq"].status == "OK"
    assert calc.valuation("601012.SH")["pe_ttm"].status == "NEGATIVE_EARNINGS_NA"   # 隆基 TTM loss


def test_dcf_requires_explicit_assumptions_and_returns_sensitivity():
    rec = FinancialCalculator(PITDataService(A, "t")).dcf("600519.SH", fcf0=6e10, growth=0.05, wacc=0.08,
                                                          terminal_growth=0.02)
    m = rec.extra["sensitivity"]["value_per_share"]
    assert rec.status == "OK" and len(m) == 5 and len(m[0]) == 5
    assert m[0][4] > m[4][0]   # lower WACC + higher growth -> higher value
    assert rec.extra["assumptions"]["wacc"] == 0.08


def test_question_about_undisclosed_period_is_refused():
    rep = run_research(ResearchRequest(ticker="000858.SZ", as_of=A, mode="fast",
                                       question="五粮液2025全年归母净利润是多少亿元？"), "B4", persist=False)
    assert not rep["answers"] and any("尚未披露" in q["topic"] for q in rep["open_questions"])


def test_pdf_number_repair():
    assert _fix_number("524334467795-,,,.") == "-5243344677.95"
    assert _fix_number("13090388963488,,,.") == "130903889634.88"
    assert row_name("1.归属于母公司股东的净利润（净亏损以“－”号填列）") == "归属于母公司股东的净利润"
