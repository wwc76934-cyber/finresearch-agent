from datetime import datetime

import pytest

from agent_runtime import ResearchRequest, run_research
from agent_runtime.agents.verifier import extract_numbers

A = datetime(2025, 12, 31, 15)


@pytest.fixture(scope="module")
def b4_report():
    return run_research(ResearchRequest(ticker="990001.SZ", as_of=A), "B4", persist=False)


def kept(rep):
    return [rep["claims"][c] for s in rep["sections"] for c in s["claim_ids"]]


def test_b4_report_structure(b4_report):
    keys = {s["key"] for s in b4_report["sections"]}
    assert {"financials", "valuation", "bull", "bear", "falsification", "risk"} <= keys
    assert len(b4_report["risk"]["matrix"]) == 6
    assert all(c["evidence_ids"] for c in kept(b4_report))
    assert all(datetime.fromisoformat(e["available_at"]) <= A for e in b4_report["evidence"].values())


def test_anomaly_triggers_replanning_and_investigation(b4_report):
    nodes = [s["node"] for s in b4_report["trace"]]
    assert "replan" in nodes and any(n.startswith("subgraph:investigate") for n in nodes)
    assert any(c["produced_by"] == "investigate" for c in kept(b4_report))


def test_future_event_not_in_report(b4_report):
    assert "立案" not in b4_report["markdown"]


def test_parent_state_holds_pointers_not_raw_tables(b4_report):
    # private subgraph scratchpads never appear in the parent-level summaries
    for s in b4_report["trace"]:
        if s["node"].startswith("subgraph:"):
            assert "private notes" in s["summary"]


def test_verifier_catches_fake_number_and_rolls_back():
    rep = run_research(ResearchRequest(ticker="990002.SZ", as_of=A, question="2025年第三季度毛利率是多少？", mode="fast",
                                       fault_injection={"fake_number": {"sticky": 0}}), "B4", persist=False)
    fails = [v for v in rep["verification_log"] if not v["passed"]]
    assert any(v["error_type"] in ("NUMBER_MISMATCH", "WILD_NUMBER") for v in fails)
    ans = next(iter(rep["answers"].values()))
    assert rep["claims"][ans["claim_id"]]["status"] == "VERIFIED"
    assert any(s["node"] == "rollback_rewrite" for s in rep["trace"])


def test_two_strike_circuit_breaker_downgrades():
    rep = run_research(ResearchRequest(ticker="990006.SZ", as_of=A, question="2025年第三季度毛利率是多少？", mode="fast",
                                       fault_injection={"fake_number": {"sticky": 5}}), "B4", persist=False)
    ans = next(iter(rep["answers"].values()))
    assert rep["claims"][ans["claim_id"]]["status"] == "DOWNGRADED"
    assert rep["uncertainties"] and any(v.get("circuit_broken") for v in rep["verification_log"])


def test_verifier_blocks_leaked_future_document():
    rep = run_research(ResearchRequest(ticker="990001.SZ", as_of=A, question="评估公司治理与合规风险",
                                       fault_injection={"future_evidence": {"document_id": "ann_990001_20260120"}}),
                       "B4", persist=False)
    assert any(v.get("error_type") == "PIT_VIOLATION" for v in rep["verification_log"])
    assert "立案告知书" not in " ".join(c["text"] for c in kept(rep))


def test_b3_without_verifier_leaks_the_same_fault():
    rep = run_research(ResearchRequest(ticker="990001.SZ", as_of=A, question="评估公司治理与合规风险",
                                       fault_injection={"future_evidence": {"document_id": "ann_990001_20260120"}}),
                       "B3", persist=False)
    assert "立案告知书" in " ".join(c["text"] for c in kept(rep))


def test_refusal_on_undisclosed_data():
    rep = run_research(ResearchRequest(ticker="990004.SH", as_of=A, question="公司的目标价是多少？", mode="fast"),
                       "B4", persist=False)
    assert rep["open_questions"] and not rep["answers"]


def test_tool_error_self_heals():
    rep = run_research(ResearchRequest(ticker="990002.SZ", as_of=datetime(2025, 6, 30, 15), mode="fast",
                                       question="2025年第一季度营业收入是多少亿元？",
                                       fault_injection={"tool_error": {"tool": "get_financial_statements", "remaining": 2}}),
                       "B4", persist=False)
    errs = [t for s in rep["trace"] for t in s["tool_calls"] if t.get("error")]
    assert errs and next(iter(rep["answers"].values()))["value"]


def test_number_extraction_ignores_dates_and_ids():
    nums = extract_numbers("2025Q3 截至2025-12-31，E001 风险3/5，营收同比18.36%，PE 19.10倍")
    assert [n[0] for n in nums] == ["18.36%", "19.10倍"]
