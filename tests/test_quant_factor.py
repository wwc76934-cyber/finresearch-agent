"""Scenario D: PIT factor panel, IC / RankIC tool, planner parse, agent run and the 10 evaluation tasks."""
import os
import re
from datetime import datetime

import numpy as np
import pytest

from agent_runtime import ResearchRequest, run_research
from agent_runtime.agents.planner import factor_ask
from data_platform import factors as F
from evaluation_platform.datasets import load_tasks
from evaluation_platform.evaluators import evaluate, task_passed
from evaluation_platform.quant_oracle import factor_stats

os.environ.setdefault("CHECKPOINTER", "memory")


@pytest.fixture(scope="module")
def panel():
    return F.get_panel()


# ---------------------------------------------------------------- point-in-time
def test_panel_covers_24_companies_in_two_universes(panel):
    assert len(panel.tickers("all")) == 24
    assert len(panel.tickers("new_energy")) == 12 and len(panel.tickers("consumer")) == 12
    assert "survivorship_note" in panel.manifest


def test_every_fact_is_visible_only_after_its_filing(panel):
    import pandas as pd

    f = pd.read_csv(F.PANEL_DIR / "financial_facts.csv", parse_dates=["published_at", "available_at"])
    assert (f.available_at > f.published_at).all()          # next trading day 09:30 after the notice date


def test_q1_report_is_not_visible_at_march_31(panel):
    t = datetime(2024, 3, 29, 15)            # last close before 2024-03-31
    for tk in panel.tickers("all"):
        pe = panel.latest_period(tk, t)
        assert pe is not None and pe < "2024-03-31"      # no company can show a 2024Q1 report yet
        assert panel.available_at(tk, "operating_revenue", pe) <= t     # and what it shows was already public
        assert panel.value(tk, "operating_revenue", "2024-03-31", t) is None


def test_forcing_a_future_report_raises_pit_violation():
    with pytest.raises(F.PITViolationError):
        F.evaluate_factor_hypothesis("ocf_to_net_profit", "new_energy", report_period="2024Q1",
                                     cross_section_dates_override=["2024-03-31"])


def test_cross_section_after_as_of_raises_pit_violation():
    with pytest.raises(F.PITViolationError):
        F.evaluate_factor_hypothesis("roe_ttm", as_of_date="2024-12-31", cross_section_dates_override=["2025-03-31"])


def test_as_of_beyond_panel_is_rejected():
    with pytest.raises(F.PanelCoverageError):
        F.evaluate_factor_hypothesis("roe_ttm", as_of_date="2030-01-01")


def test_forward_windows_never_pass_as_of(panel):
    from datetime import datetime as dt

    for as_of in ("2024-06-30", "2025-03-31", "2025-12-31"):
        dates = F.cross_section_dates(panel, dt.fromisoformat(as_of + "T15:00:00"), 60)
        last = panel.last_trading_day_on_or_before(as_of)
        assert all(panel.date_pos[d] + 60 <= panel.date_pos[last] for d in dates)


def test_earlier_as_of_uses_fewer_cross_sections():
    a = F.evaluate_factor_hypothesis("roe_ttm", "all", "2024-06-30", 20, 60, 3)
    b = F.evaluate_factor_hypothesis("roe_ttm", "all", "2025-12-31", 20, 60, 3)
    assert a.n_cross_sections < b.n_cross_sections
    assert max(c.date for c in a.cross_sections) <= "2024-03-31"


# ---------------------------------------------------------------- statistics vs the independent oracle
@pytest.mark.parametrize("factor", F.FACTORS)
def test_tool_matches_independent_oracle(factor):
    r = F.evaluate_factor_hypothesis(factor, "all", "2025-12-31", 8, 60, 3)
    o = factor_stats(factor, "all", "2025-12-31", 8, 60, 3)
    assert r.n_cross_sections == o["n_cs"]
    for got, key in ((r.rank_ic_mean, "rank_ic_mean"), (r.ic_mean, "ic_mean"), (r.rank_ic_t_stat, "rank_ic_t"),
                     (r.rank_ic_p_value, "rank_ic_p"), (r.long_short_cum_return, "ls_cum"),
                     (r.long_short_max_drawdown, "ls_mdd")):
        assert got == pytest.approx(o[key], rel=1e-9, abs=1e-12)


def test_rank_ic_is_spearman_and_bounded():
    r = F.evaluate_factor_hypothesis("ocf_to_net_profit", "all", "2025-12-31", 8, 60, 3)
    assert all(-1 <= c.rank_ic <= 1 and -1 <= c.ic <= 1 for c in r.cross_sections)
    assert r.rank_ic_mean == pytest.approx(np.mean([c.rank_ic for c in r.cross_sections]))
    assert len(r.group_mean_returns) == 3 and len(r.nav) == r.n_cross_sections + 1


def test_warnings_flag_small_samples():
    r = F.evaluate_factor_hypothesis("roe_ttm", "new_energy", "2025-12-31", 3, 20, 3)
    codes = {w["code"] for w in r.warnings}
    assert {"SURVIVORSHIP", "FEW_CROSS_SECTIONS", "SMALL_GROUPS"} <= codes


def test_factor_names_contain_no_digits():
    # a digit in the display name looks like an uncited number to the Tier-1 verifier
    assert not [n for n in F.FACTOR_CN.values() if re.search(r"\d", n)]


# ---------------------------------------------------------------- planner + agent
def test_planner_parses_factor_questions():
    a = factor_ask("在过去2年新能源板块中，经营现金流与净利润比值是否具有更优的下一季度超额收益？三分组。")
    assert a.params["factor_name"] == "ocf_to_net_profit" and a.params["universe"] == "new_energy"
    assert a.params["lookback_quarters"] == 8 and a.params["forward_horizon_days"] == 60
    assert factor_ask("宁德时代2025年第三季度的经营现金流/净利润是多少？") is None   # single-company metric, not a factor test


def test_calendar_dates_are_not_horizons():
    a = factor_ask("在2024年3月31日用2024年一季报的ROE因子对新能源做回测")
    assert a.params["forward_horizon_days"] == 60 and a.params["report_period"] == "2024Q1"


def test_agent_answers_factor_question_with_verified_numbers():
    t = next(x for x in load_tasks(("quant",)) if x["task_id"] == "Q001")
    rep = run_research(ResearchRequest(ticker=t["ticker"], as_of=datetime.fromisoformat(t["as_of"]), question=t["question"]),
                       "B4", persist=False)
    assert rep["scenario"] == "factor_test" and not rep["uncertainties"]
    assert not rep["debate"] and not rep["risk"]          # a universe-level test has nothing to debate
    assert any(k.startswith("factor_rank_ic_") for k in rep["charts"])
    assert task_passed(evaluate(t, rep), t)


def test_pit_trap_is_refused_without_numbers():
    t = next(x for x in load_tasks(("quant",)) if x["task_id"] == "Q009")
    rep = run_research(ResearchRequest(ticker=t["ticker"], as_of=datetime.fromisoformat(t["as_of"]), question=t["question"]),
                       "B4", persist=False)
    assert any("PITViolationError" in q["topic"] for q in rep["open_questions"])
    assert not [c for c in rep["claims"].values() if c["section"] == "answer"]


# ---------------------------------------------------------------- the 10 tasks
def test_quant_task_set_shape():
    tasks = load_tasks(("quant",))
    assert len(tasks) == 10
    cats = {t["category"] for t in tasks}
    assert cats == {"factor_test", "factor_sample_warning", "factor_pit_trap"}
    assert sum(t["category"] == "factor_pit_trap" for t in tasks) == 2


@pytest.mark.parametrize("task_id", [f"Q{i:03d}" for i in range(1, 11)])
def test_b4_passes_every_quant_task(task_id):
    t = next(x for x in load_tasks(("quant",)) if x["task_id"] == task_id)
    rep = run_research(ResearchRequest(ticker=t["ticker"], as_of=datetime.fromisoformat(t["as_of"]), question=t["question"]),
                       "B4", persist=False)
    sc = evaluate(t, rep)
    assert task_passed(sc, t), {k: v for k, v in sc.items() if k in ("numerical_accuracy", "factor_caveats", "refusal", "citation_validity")}
