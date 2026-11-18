"""v2.0 Epic 4: a deep run leaves a thesis with machine-readable falsification conditions; a later run on the same
company re-tests them against newer disclosures. Memory is switched on explicitly here (it is off in offline_golden)."""
from datetime import date, datetime

import pytest
from sqlalchemy import delete, select

from agent_runtime import ResearchRequest, run_research
from agent_runtime.memory import STATUS_CN, ThesisMemoryStore, judge, memory_enabled
from data_platform import FinancialCalculator, LineageRecorder, PITDataService
from data_platform.db import ResearchThesisRow, get_session_factory

T = "300750.SZ"
FIRST, SECOND = datetime(2025, 6, 30, 15), datetime(2025, 12, 31, 15)


@pytest.fixture()
def clean_memory():
    """Theses written by a test are removed afterwards; any that existed before stay."""
    Session = get_session_factory()
    with Session() as s:
        before = set(s.scalars(select(ResearchThesisRow.thesis_id)))
    yield
    with Session() as s:
        s.execute(delete(ResearchThesisRow).where(ResearchThesisRow.thesis_id.not_in(before) if before else True))
        s.commit()


def run(as_of, **kw):
    return run_research(ResearchRequest(ticker=T, as_of=as_of, data_mode="offline_golden", use_memory=True, **kw), "B4", persist=False)


def saved(thesis_id):
    with get_session_factory()() as s:
        return s.get(ResearchThesisRow, thesis_id)


# ------------------------------------------------------------------ the rule
@pytest.mark.parametrize("comparator,base,thr,now,want", [
    ("lt", 0.30, 0.28, 0.31, "CONFIRMED"), ("lt", 0.30, 0.28, 0.29, "WEAKENED"), ("lt", 0.30, 0.28, 0.27, "FALSIFIED"),
    ("lt", 0.30, 0.28, 0.28, "WEAKENED"),                 # exactly on the threshold is not below it; but all the way there
    ("gt", 60.0, 66.0, 59.0, "CONFIRMED"), ("gt", 60.0, 66.0, 64.0, "WEAKENED"), ("gt", 60.0, 66.0, 67.0, "FALSIFIED"),
    ("lt", 0.30, 0.28, None, "UNKNOWN")])
def test_judge_applies_the_stored_condition(comparator, base, thr, now, want):
    assert judge({"comparator": comparator, "threshold": thr}, base, now) == want


def test_memory_is_off_in_the_benchmark_mode_unless_asked():
    base = dict(ticker=T, as_of=FIRST)
    assert memory_enabled(ResearchRequest(**base, data_mode="offline_golden")) is False
    assert memory_enabled(ResearchRequest(**base, data_mode="hybrid_live")) is True
    assert memory_enabled(ResearchRequest(**base, data_mode="offline_golden", use_memory=True)) is True
    assert memory_enabled(ResearchRequest(**base, data_mode="hybrid_live", use_memory=False)) is False


# ------------------------------------------------------------------ write
def test_a_deep_run_saves_its_thesis_and_conditions(clean_memory):
    rep = run(FIRST)
    tid = rep["thesis_id"]
    row = saved(tid)
    assert row.ticker == T and row.as_of_date == "2025-06-30" and row.stance in ("bullish", "neutral", "bearish")
    assert row.core_thesis_summary and row.locked_metrics_snapshot["period"] == "2025Q1"
    assert len(row.falsification_criteria) >= 2
    for c in row.falsification_criteria:
        assert set(c) >= {"metric", "comparator", "threshold", "baseline", "label", "claim_id"}
        assert c["claim_id"] in rep["claims"] and rep["claims"][c["claim_id"]]["status"] in ("VERIFIED", "PARTIALLY_SUPPORTED")
    assert rep["thesis_review"]["items"] == []                      # nothing earlier to review


def test_benchmark_mode_leaves_no_trace(clean_memory):
    before = len(ThesisMemoryStore().listing(T, 100))
    rep = run_research(ResearchRequest(ticker=T, as_of=FIRST, data_mode="offline_golden"), "B4", persist=False)
    assert rep.get("thesis_id") is None and rep["thesis_review"] is None
    assert len(ThesisMemoryStore().listing(T, 100)) == before


def test_a_lookup_or_a_run_without_a_debate_saves_nothing(clean_memory):
    fast = run(FIRST, mode="fast", question="宁德时代最新一期累计营业收入是多少亿元？")
    assert fast.get("thesis_id") is None
    nomem = run_research(ResearchRequest(ticker=T, as_of=FIRST, data_mode="offline_golden", use_memory=True), "B4-no_memory", persist=False)
    assert nomem.get("thesis_id") is None


# ------------------------------------------------------------------ review
def test_the_second_run_reviews_the_first_runs_conditions(clean_memory):
    first = run(FIRST)
    second = run(SECOND)
    rv = second["thesis_review"]
    assert rv["theses"][0]["thesis_id"] == first["thesis_id"] and rv["theses"][0]["from_period"] == "2025Q1"
    assert rv["theses"][0]["to_period"] == "2025Q3"
    spec_by_metric = {c["metric"]: c for c in saved(first["thesis_id"]).falsification_criteria}
    assert {i["metric"] for i in rv["items"]} == set(spec_by_metric)

    # independent check: recompute the metrics with the calculators and apply the rules by hand
    calc = FinancialCalculator(PITDataService(SECOND, "x"), LineageRecorder(None, "t"))
    old, new = calc.ratios(T, date(2025, 3, 31)), calc.ratios(T, date(2025, 9, 30))
    for item in rv["items"]:
        spec = spec_by_metric[item["metric"]]
        o, n, thr = old[item["metric"]].value, new[item["metric"]].value, spec["threshold"]
        lt = spec["comparator"] == "lt"
        if (lt and n < thr) or (not lt and n > thr):
            want = "FALSIFIED"
        else:
            progress = (o - n) if lt else (n - o)
            want = "WEAKENED" if progress / abs(o - thr) >= 0.5 else "CONFIRMED"
        assert item["status"] == want, item
        claim = second["claims"][item["claim_id"]]
        assert claim["section"] == "review" and claim["status"] == "VERIFIED"          # Tier-1 checked every number in it
        assert f"[{want}]" in claim["text"] and STATUS_CN[want] in claim["text"]
    assert "## 历史投研观点复盘与演进" in second["markdown"]


def test_a_breached_condition_is_reported_as_falsified(clean_memory):
    with get_session_factory()() as s:
        s.add(ResearchThesisRow(
            thesis_id="thesis_manual_1", ticker=T, as_of_date="2025-06-30", run_id="manual", stance="bullish",
            core_thesis_summary="毛利率保持高位", created_at="2025-06-30T16:00:00",
            locked_metrics_snapshot={"period": "2025Q1"},
            falsification_criteria=[{"claim_id": "C000", "metric": "gross_margin", "comparator": "lt", "threshold": 0.99,
                                     "baseline": 0.2531, "period": "2025Q1", "label": "毛利率", "text": ""}]))
        s.commit()
    rv = run(SECOND)["thesis_review"]
    items = [i for i in rv["items"] if i["thesis_id"] == "thesis_manual_1"]
    assert items and items[0]["status"] == "FALSIFIED"
    assert [t for t in rv["theses"] if t["thesis_id"] == "thesis_manual_1"][0]["overall"] == "FALSIFIED"


def test_no_new_filing_since_the_last_run_means_no_review(clean_memory):
    first = run(FIRST)
    again = run(datetime(2025, 7, 15, 15))              # Q2 is not out until August: same latest period as the thesis
    assert first["thesis_id"] and again["thesis_review"]["items"] == []
    assert any("没有新披露的财报" in q["reason"] for q in again["open_questions"] if q["topic"] == "历史观点复盘")


def test_only_earlier_runs_are_reviewed(clean_memory):
    run(SECOND)
    assert run(FIRST)["thesis_review"]["items"] == []    # a run from the past never reviews a later thesis


def test_theses_are_listed_by_the_api(clean_memory):
    from fastapi.testclient import TestClient

    from apps.api_server.main import app

    tid = run(FIRST)["thesis_id"]
    rows = TestClient(app).get("/api/theses", params={"ticker": T}).json()
    assert any(r["thesis_id"] == tid and r["criteria"] >= 2 for r in rows)
