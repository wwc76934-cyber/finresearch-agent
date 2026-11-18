"""v2.0 Epic 3: POST /api/research/stream and the catalog search behind the web UI's company picker."""
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from apps.api_server.main import app  # noqa: E402
from apps.api_server.stream import PROTOCOL_TYPES, to_stream_events  # noqa: E402
from data_platform.ashare_catalog import AShareCatalog, set_catalog  # noqa: E402

client = TestClient(app)
BODY = {"ticker": "990001.SZ", "as_of": "2025-12-31T15:00:00", "architecture": "B4", "data_mode": "offline_golden", "hitl": False}


def read_stream(body, limit=400):
    events, first_at, t0 = [], None, time.perf_counter()
    with client.stream("POST", "/api/research/stream", json=body) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            if first_at is None:
                first_at = time.perf_counter() - t0
            ev = json.loads(line[6:])
            events.append(ev)
            if ev["type"] in ("done", "error") or len(events) >= limit:
                break
    return events, first_at


def test_the_first_event_arrives_before_any_work_and_the_protocol_is_complete():
    events, first_at = read_stream(BODY)
    types = [e["type"] for e in events]
    assert first_at is not None and first_at < 0.8, f"first byte after {first_at:.2f}s"
    assert types[0] == "run_started" and types[-1] == "done"
    first = events[0]
    assert first["run_id"].startswith("run_") and first["tickers"] == ["990001.SZ"] and first["data_mode"] == "offline_golden"
    got = set(types)
    # offline_golden never fetches anything, so live_ingestion_status is the one protocol type that stays silent here
    assert got >= set(PROTOCOL_TYPES) - {"live_ingestion_status"}, set(PROTOCOL_TYPES) - got
    assert types.index("plan_generated") < types.index("verifier_audit") < types.index("report_completed")
    done = events[-1]
    assert done["run_id"] == first["run_id"]


def test_event_payloads_carry_what_the_ui_needs():
    events, _ = read_stream(BODY)
    by = lambda t: [e for e in events if e["type"] == t]  # noqa: E731
    plan = by("plan_generated")[0]
    assert plan["route"] in ("fast", "standard", "full") and plan["subtasks"] and plan["tickers"] == ["990001.SZ"]
    sub = by("subgraph_progress")
    assert {e["phase"] for e in sub} >= {"start", "end"} and {e["agent"] for e in sub} >= {"fundamental"}
    an = by("anomaly_detected")
    assert an and all({"code", "severity", "action"} <= set(e) for e in an)           # the lab company is built with anomalies
    deb = by("debate_round")
    assert any(isinstance(e["round"], int) for e in deb) and any(e["round"] == "summary" and "fact_lock" in e for e in deb)
    aud = by("verifier_audit")
    assert aud[0]["pass"] == 1 and "failures_by_type" in aud[0] and aud[-1]["action"] == "pass"
    rep = by("report_completed")[0]
    assert rep["run_id"] == events[0]["run_id"] and rep["report"]["claims"] and rep["report"]["charts"] is not None
    assert rep["bill"]["tokens"] > 0 and "cost_cny" in rep["bill"]


def test_a_lookup_streams_the_lean_route():
    events, _ = read_stream({**BODY, "ticker": "300750.SZ", "mode": "fast", "question": "宁德时代最新一期累计营业收入是多少亿元？"})
    plan = next(e for e in events if e["type"] == "plan_generated")
    assert plan["route"] == "fast"
    assert not [e for e in events if e["type"] == "debate_round"]


def test_a_bad_architecture_is_rejected_before_streaming():
    assert client.post("/api/research/stream", json={**BODY, "architecture": "nope"}).status_code == 400


def test_the_translator_maps_runtime_events_and_drops_unknown_ones():
    assert to_stream_events({"type": "node_start", "node": "planner"}) == [{"type": "node_start", "node": "planner"}]
    out = to_stream_events({"type": "node_end", "node": "subgraph:news:T3", "latency_ms": 5, "tool_calls": 2, "summary": "s"})
    assert [e["type"] for e in out] == ["node_end", "subgraph_progress"] and out[1]["agent"] == "news"
    tool = to_stream_events({"type": "tool", "node": "subgraph:fundamental:T1", "tool": "explore_hypothesis", "args": {"sub_id": "H1"}, "result": "C018"})
    assert tool[0]["phase"] == "tool" and tool[0]["tool"] == "explore_hypothesis"
    assert to_stream_events({"type": "tool", "node": "planner", "tool": "x"}) == []
    assert to_stream_events({"type": "something_else"}) == []


def test_catalog_search_by_code_name_and_pinyin():
    set_catalog(AShareCatalog.from_stock_list([
        {"code": "300308", "zwjc": "中际旭创", "pinyin": "zjxc", "orgId": "x", "category": "A股"},
        {"code": "300502", "zwjc": "新易盛", "pinyin": "xys", "orgId": "x", "category": "A股"},
        {"code": "600519", "zwjc": "贵州茅台", "pinyin": "gzmt", "orgId": "x", "category": "A股"}]))
    try:
        for q, want in (("3003", "300308.SZ"), ("旭创", "300308.SZ"), ("zjx", "300308.SZ"), ("茅台", "600519.SH"), ("600519", "600519.SH")):
            r = client.get("/api/catalog/search", params={"q": q}).json()
            assert r["catalog_size"] == 3 and want in [x["ticker"] for x in r["results"]], (q, r)
        assert client.get("/api/catalog/search", params={"q": "不存在"}).json()["results"] == []
        assert client.get("/api/catalog/search", params={"q": ""}).json()["results"] == []
    finally:
        set_catalog(None)


def test_catalog_search_marks_stored_companies_and_works_without_a_catalog(monkeypatch):
    import data_platform.ashare_catalog as ac

    monkeypatch.setattr(ac, "CATALOG_PATH", ROOT / "data" / "_no_such_catalog.json")
    set_catalog(None)
    r = client.get("/api/catalog/search", params={"q": "宁德"}).json()
    assert r["catalog_size"] == 0 and any(x["ticker"] == "300750.SZ" and x["local"] for x in r["results"])
