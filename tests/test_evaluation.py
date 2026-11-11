import importlib.util
from datetime import datetime
from pathlib import Path

from agent_runtime import ResearchRequest, run_research
from agent_runtime.state import ArchConfig
from evaluation_platform.datasets import load_tasks
from evaluation_platform.evaluators import attribution, task_passed

ROOT = Path(__file__).resolve().parents[1]


def _task(tid):
    return next(t for t in load_tasks() if t["task_id"] == tid)


def test_attribution_needs_replanning():
    # L013 (990003): impairment and inventory anomalies. The Fundamental subgraph's own cross-check only covers
    # profit/cash-flow and receivables, so these still need the main graph's re-planning to be explained.
    t = next(x for x in load_tasks(("lab",)) if x["task_id"] == "L013")
    req = ResearchRequest(ticker=t["ticker"], as_of=datetime.fromisoformat(t["as_of"]), question=t["question"])
    assert attribution(t, run_research(req, "B4", persist=False))["score"] == 1.0
    assert attribution(t, run_research(req, "B4-no_replanning", persist=False))["score"] == 0.0


def test_subgraph_crosscheck_explains_accounting_identity_breaks_without_replanning():
    # L011 (990001): profit vs cash flow, receivables vs revenue. PRD v1.2 1.1: the subgraph reads MD&A / NOTES itself.
    t = next(x for x in load_tasks(("lab",)) if x["task_id"] == "L011")
    req = ResearchRequest(ticker=t["ticker"], as_of=datetime.fromisoformat(t["as_of"]), question=t["question"])
    assert attribution(t, run_research(req, "B3", persist=False))["score"] == 1.0          # no re-planning in B3
    off = ArchConfig(name="B4-no_crosscheck_no_replanning", replanning=False, subgraph_crosscheck=False)
    assert attribution(t, run_research(req, off, persist=False))["score"] == 0.0           # both off: nothing explains it


def test_llm_rubric_does_not_gate_pass():
    scores = {"pit_violation": {"score": 0.0}, "citation_validity": {"score": 1.0},
              "rubric": {"score": 0.5, "judge": "llm"}}
    assert task_passed(scores)
    scores["rubric"]["judge"] = "structural"
    assert not task_passed(scores)


def test_uncited_answer_fails():
    # B0 used to pass open questions on the PIT check alone
    assert not task_passed({"pit_violation": {"score": 0.0}, "rubric": {"score": 0.9, "judge": "llm"}})
    assert task_passed({"pit_violation": {"score": 0.0}, "refusal": {"score": 1.0}})


def test_blind_template_builds():
    spec = importlib.util.spec_from_file_location("build_blind", ROOT / "scripts" / "build_blind.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tasks = mod.build(ROOT / "benchmarks" / "blind" / "blind_template.jsonl")
    pit = next(t for t in tasks if t["task_id"] == "B000")
    assert pit["expected"]["numbers"][0]["period"] == "2025Q1"  # H1 report not yet usable at 21:00
    assert {t["category"] for t in tasks} == {"blind_metric", "refusal_trap", "anomaly_attribution"}


def test_routine_related_party_filings_are_not_signals():
    from data_platform.calculators import ROUTINE_RPT_RE
    assert ROUTINE_RPT_RE.search("关于预计2026年度日常关联交易的公告")
    assert ROUTINE_RPT_RE.search("关于贵州茅台酒股份有限公司涉及财务公司关联交易的存款、贷款等金融业务的专项说明")
    assert ROUTINE_RPT_RE.search("关联交易管理办法")
    assert not ROUTINE_RPT_RE.search("关于重大关联交易的公告")
    assert not ROUTINE_RPT_RE.search("关于拟向员工持股平台转让控股子公司部分股权暨关联交易的公告")


def test_announcement_boilerplate_is_stripped():
    from agent_runtime.agents.collectors import _strip_boilerplate
    s = ("1证券代码：600519公告编号：临2026-003关于回购股份实施进展的公告本公司董事会及全体董事保证本公告内容不存在任何虚假记载，"
         "并对其内容的真实性、准确性和完整性承担法律责任。重要内容提示：累计已回购金额801,023,304.38元，回购价格区间1,322.60元/股")
    assert _strip_boilerplate(s).startswith("累计已回购金额")


def test_metric_name_windows_are_not_numbers():
    from agent_runtime.agents.verifier import extract_numbers
    assert [n[1] for n in extract_numbers("2026-05-06 PE 3年分位71.43%；最大回撤(1年)为-19.04%")] == [71.43, -19.04]


def test_multidoc_task_set_shape_and_answer_key():
    tasks = load_tasks(("multidoc",))
    assert len(tasks) == 15
    for t in tasks:
        md = t["expected"]["multidoc"]
        assert md["annual"].endswith("2025Q4") and md["min_announcements"] == 2 and len(md["announcement_pool"]) >= 3
        assert t["expected"]["numbers"][0]["value"] is not None
        assert t["as_of"] > "2026-03"          # after the 2025 annual report was public


def test_multidoc_needs_annual_report_notes_and_two_announcements():
    from evaluation_platform.evaluators import multidoc_coverage

    t = load_tasks(("multidoc",))[0]
    req = ResearchRequest(ticker=t["ticker"], as_of=datetime.fromisoformat(t["as_of"]), question=t["question"])
    rep = run_research(req, "B4", persist=False)
    m = multidoc_coverage(t, rep)
    assert m["annual_cited"] and m["announcements_cited"] >= 2 and m["score"] == 1.0


def test_react_transcript_drops_oldest_observations_first():
    from agent_runtime.react_agent import CONTEXT_BUDGET, Observation, _transcript
    from agent_runtime.llm import estimate_tokens

    import agent_runtime.react_agent as ra

    saved, ra.OBS_CHARS = ra.OBS_CHARS, 5000   # force pressure: realistic observations (200-700 tokens) never reach 8k
    obs = {f"O{i}": Observation(f"O{i}", "search_documents", "营业收入应收账款存货" * 500, []) for i in range(1, 9)}
    steps = [{"n": i, "oid": f"O{i}", "action": "search_documents", "args": {}, "thought": "t"} for i in range(1, 9)]
    text = _transcript(steps, obs)
    assert estimate_tokens(text) <= CONTEXT_BUDGET
    dropped = [o for o in obs if obs[o].dropped]
    assert dropped and dropped == [f"O{i}" for i in range(1, len(dropped) + 1)]      # a prefix: the oldest go first
    assert not obs["O8"].dropped
    ra.OBS_CHARS = saved
