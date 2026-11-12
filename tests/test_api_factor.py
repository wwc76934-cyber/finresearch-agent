"""Factor-validation endpoints behind the web UI (scenario D) and the removal of the legacy front end."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from apps.api_server.main import app  # noqa: E402

client = TestClient(app)


def test_factor_list_and_test():
    meta = client.get("/api/factors").json()
    assert len(meta["factors"]) == 6 and meta["panel_companies"] == 24
    r = client.post("/api/factors/test", json={"factor": "roe_ttm", "universe": "all", "as_of_date": "2025-12-31"})
    assert r.status_code == 200
    d = r.json()
    assert d["n_cross_sections"] >= 6 and len(d["nav"]) == d["n_cross_sections"] + 1
    assert {w["code"] for w in d["warnings"]} >= {"SURVIVORSHIP"}


def test_future_report_is_refused_with_422():
    r = client.post("/api/factors/test", json={"factor": "ocf_to_net_profit", "universe": "all",
                                               "as_of_date": "2025-12-31", "report_period": "2025Q4"})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "PIT_VIOLATION"


def test_bad_input_is_rejected():
    assert client.post("/api/factors/test", json={"factor": "nope"}).status_code == 422
    assert client.post("/api/factors/test", json={"factor": "roe_ttm", "universe": "mars"}).status_code == 422


def test_legacy_front_end_is_gone():
    assert client.get("/legacy").status_code == 404
    assert not (ROOT / "apps" / "web_workspace").exists()


def test_otel_endpoint_reports_disabled_by_default():
    assert client.get("/api/otel/spans").json() == {"enabled": False, "spans": []}
