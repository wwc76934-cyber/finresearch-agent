from datetime import datetime

import pytest

from agent_runtime import ResearchRequest, run_research
from agent_runtime.graph import checkpoint_history, resume_research
from agent_runtime.pdf_export import markdown_to_pdf

A = datetime(2025, 12, 31, 15)


def _paused(ticker="990001.SZ", **kw):
    r = run_research(ResearchRequest(ticker=ticker, as_of=A, hitl=True, token_budget=100, **kw), "B4", persist=False)
    return r


def test_over_budget_pauses_for_approval_then_approve_runs_investigation():
    r = _paused()
    assert r["status"] == "awaiting_approval"
    assert {p["code"] for p in r["approval"]["proposed"]} == {"PROFIT_OCF_DIVERGENCE", "AR_OUTPACE_REVENUE"}
    rep = resume_research(r["run_id"], {"action": "approve", "extra_tokens": 5000})
    nodes = [s["node"] for s in rep["trace"]]
    assert rep["status"] == "done" and "approval" in nodes and any(n.startswith("subgraph:investigate") for n in nodes)


def test_skip_records_open_question_and_no_investigation():
    r = _paused()
    rep = resume_research(r["run_id"], {"action": "skip"})
    assert not any(s["node"].startswith("subgraph:investigate") for s in rep["trace"])
    assert any("人工跳过" in q["topic"] for q in rep["open_questions"])


def test_modify_uses_human_query():
    r = _paused()
    rep = resume_research(r["run_id"], {"action": "modify", "query": "应收账款 前五名 欠款方", "extra_tokens": 5000})
    calls = [t for s in rep["trace"] if s["node"].startswith("subgraph:investigate") for t in s["tool_calls"]]
    assert any("应收账款 前五名 欠款方" in str(t["args"]) for t in calls)


def test_without_hitl_over_budget_skips_silently():
    rep = run_research(ResearchRequest(ticker="990001.SZ", as_of=A, token_budget=100), "B4", persist=False)
    assert rep["status"] == "done" and "replan" not in [s["node"] for s in rep["trace"]]


def test_paused_state_is_checkpointed():
    r = _paused()
    hist = checkpoint_history(r["run_id"])
    assert hist and hist[0]["next"] == ["approval"]
    resume_research(r["run_id"], {"action": "skip"})


def test_pdf_export_renders_chinese_report():
    rep = run_research(ResearchRequest(ticker="990004.SH", as_of=A, mode="fast"), "B4", persist=False)
    pdf = markdown_to_pdf(rep["markdown"])
    assert pdf.startswith(b"%PDF") and len(pdf) > 5000


def test_unknown_run_cannot_be_resumed():
    with pytest.raises(KeyError):
        resume_research("run_does_not_exist", {"action": "approve"})
