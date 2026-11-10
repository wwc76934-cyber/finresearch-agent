import importlib.util
from datetime import datetime
from pathlib import Path

from agent_runtime import ResearchRequest, run_research
from evaluation_platform.datasets import load_tasks
from evaluation_platform.evaluators import attribution, task_passed

ROOT = Path(__file__).resolve().parents[1]


def _task(tid):
    return next(t for t in load_tasks() if t["task_id"] == tid)


def test_attribution_needs_replanning():
    t = next(x for x in load_tasks(("lab",)) if x["task_id"] == "L011")  # 990001: profit/cash-flow divergence + receivables
    req = ResearchRequest(ticker=t["ticker"], as_of=datetime.fromisoformat(t["as_of"]), question=t["question"])
    assert attribution(t, run_research(req, "B4", persist=False))["score"] == 1.0
    assert attribution(t, run_research(req, "B4-no_replanning", persist=False))["score"] == 0.0


def test_llm_rubric_does_not_gate_pass():
    scores = {"pit_violation": {"score": 0.0}, "rubric": {"score": 0.5, "judge": "llm"}}
    assert task_passed(scores)
    scores["rubric"]["judge"] = "structural"
    assert not task_passed(scores)


def test_blind_template_builds():
    spec = importlib.util.spec_from_file_location("build_blind", ROOT / "scripts" / "build_blind.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tasks = mod.build(ROOT / "benchmarks" / "blind" / "blind_template.jsonl")
    pit = next(t for t in tasks if t["task_id"] == "B000")
    assert pit["expected"]["numbers"][0]["period"] == "2025Q1"  # H1 report not yet usable at 21:00
    assert {t["category"] for t in tasks} == {"blind_metric", "refusal_trap", "anomaly_attribution"}
