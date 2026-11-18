"""v2.0 Epic 2: hypothesis-driven planning, bounded ReAct-Lite exploration, and multi-company comparison (scenario C).
Offline: no model and no network. What a model adds to the plan / rewrite step is covered by the validation tests."""
from datetime import date, datetime

import pytest

from agent_runtime import ResearchRequest, run_research
from agent_runtime.agents import hypotheses as H
from agent_runtime.state import ARCHITECTURES

A = datetime(2025, 12, 31, 15)
OPEN_Q = "分析海外建厂的地缘风险与资本开支转化效率对自由现金流的影响，并看客户集中度"


def _tool_calls(rep, name):
    return [t for s in rep["trace"] for t in (s.get("tool_calls") or []) if isinstance(t, dict) and t.get("tool") == name]


# ------------------------------------------------------------------ planning
def test_open_question_becomes_sub_hypotheses_with_their_own_search_phrases():
    hs = H.build(OPEN_Q, "300750.SZ", "full", None, set())
    assert 2 <= len(hs) <= H.MAX_HYPOTHESES
    assert {h.assigned_agent for h in hs} == {"fundamental"}
    qs = [q for h in hs for q in h.custom_rag_queries]
    assert any("境外" in q or "海外" in q for q in qs) and any("在建工程" in q for q in qs) and any("前五名客户" in q for q in qs)
    assert all(h.verification_criterion and h.target_ticker == "300750.SZ" for h in hs)
    assert [h.sub_id for h in hs] == [f"H{i + 1}" for i in range(len(hs))]


def test_lookups_and_standard_questions_get_no_hypotheses():
    assert H.build("宁德时代最新一期累计营收是多少亿元", "300750.SZ", "fast", None, set()) == []
    assert H.build("公司最新一期有哪些财务指标出现背离", "300750.SZ", "standard", None, set()) == []


def test_model_proposals_are_validated_not_repaired():
    good = {"assigned_agent": "news", "hypothesis_statement": "关税政策变化", "custom_rag_queries": ["关税 政策", "出口 管制"],
            "required_metrics": ["gross_margin", "made_up_metric"], "verification_criterion": "公告出现关税相关表述"}
    bad = [{"assigned_agent": "robot", "hypothesis_statement": "x", "custom_rag_queries": ["a b"]},       # unknown agent
           {"assigned_agent": "news", "hypothesis_statement": "", "custom_rag_queries": ["关税 政策"]},     # empty statement
           {"assigned_agent": "news", "hypothesis_statement": "y", "custom_rag_queries": ["x" * 40]},     # phrase too long
           "not a dict"]
    out = H.validate_model_hypotheses([good, *bad], "300750.SZ", {"gross_margin"}, start=1)
    assert len(out) == 1 and out[0].required_metrics == ["gross_margin"]
    assert H.validate_model_hypotheses("nonsense", "300750.SZ", set(), 1) == []


def test_model_proposal_that_repeats_a_rule_phrase_is_dropped_and_total_is_capped():
    rules = H.rule_hypotheses(OPEN_Q, "300750.SZ")
    dup = [h.model_copy(update={"sub_id": "X"}) for h in rules]
    assert len(H.merge(rules, dup)) == len(rules)
    many = [H.SubHypothesis(sub_id=f"M{i}", target_ticker="300750.SZ", assigned_agent="news", hypothesis_statement="s",
                            custom_rag_queries=[f"独立短语{i}"]) for i in range(10)]
    assert len(H.merge(rules, many)) == H.MAX_HYPOTHESES


# ------------------------------------------------------------------ ReAct-Lite
def test_subgraph_searches_for_the_hypotheses_and_every_outcome_is_visible():
    rep = run_research(ResearchRequest(ticker="300750.SZ", as_of=A, question=OPEN_Q, data_mode="offline_golden"), "B4", persist=False)
    assert rep["plan"]["hypotheses"], "the plan should carry the hypotheses"
    calls = _tool_calls(rep, "explore_hypothesis")
    assert calls, "the sub-graph should have searched"
    found = [c for c in rep["claims"].values() if "对应假设[H" in c["text"]]
    missing = [q for q in rep["open_questions"] if q["topic"].startswith("假设[H")]
    assert found or missing                                                 # each hypothesis ends as a cited claim or as an open question
    for c in found:                                                         # a found claim quotes a filing and is PIT-safe
        assert c["evidence_ids"] and all(rep["evidence"][e]["kind"] == "document" for e in c["evidence_ids"])
        assert all(rep["evidence"][e]["available_at"] <= rep["as_of"] for e in c["evidence_ids"])
    assert all(len([x for x in calls if x["args"].get("sub_id") == h["sub_id"] and x["args"].get("step") in (1, 2)]) <= 3
               for h in rep["plan"]["hypotheses"])                          # bounded: two phrases + at most one rewrite


def test_react_lite_can_be_switched_off():
    off = run_research(ResearchRequest(ticker="300750.SZ", as_of=A, question=OPEN_Q, data_mode="offline_golden"),
                       "B4-no_react_lite", persist=False)
    assert ARCHITECTURES["B4-no_react_lite"].react_lite_steps == 0
    assert not _tool_calls(off, "explore_hypothesis")
    assert not [c for c in off["claims"].values() if "对应假设[H" in c["text"]]


def test_a_plain_metric_question_runs_the_fixed_pipeline_unchanged():
    rep = run_research(ResearchRequest(ticker="300750.SZ", as_of=A, question="宁德时代最新一期累计营业收入是多少亿元？", mode="fast",
                                       data_mode="offline_golden"), "B4", persist=False)
    assert rep["plan"]["hypotheses"] == [] and not _tool_calls(rep, "explore_hypothesis")


# ------------------------------------------------------------------ multi-company comparison
def _compare(question="对比宁德时代与比亚迪的盈利质量和估值"):
    return run_research(ResearchRequest(ticker="300750.SZ", as_of=A, question=question, data_mode="offline_golden"), "B4", persist=False)


def test_two_companies_in_the_question_give_one_aligned_matrix():
    rep = _compare()
    cmp_ = rep["comparison"]
    assert [t["ticker"] for t in cmp_["tickers"]] == ["300750.SZ", "002594.SZ"]      # the primary first
    assert cmp_["period"] == "2025Q3" and {r["dimension"] for r in cmp_["rows"]} == {"成长性", "盈利能力", "盈利质量", "财务结构", "估值"}
    assert "## 同业对比" in rep["markdown"] and "宁德时代" in rep["markdown"] and "比亚迪" in rep["markdown"]
    cl = [c for c in rep["claims"].values() if c["section"] == "comparison"]
    assert len(cl) >= 5 and all(c["status"] in ("VERIFIED", "PARTIALLY_SUPPORTED") for c in cl)   # Tier-1 passed every comparison sentence


def test_matrix_numbers_are_the_calculators_not_the_models():
    from data_platform import FinancialCalculator, LineageRecorder, PITDataService

    rep = _compare()
    svc = PITDataService(A, "x")
    calc = FinancialCalculator(svc, LineageRecorder(None, "t"))
    pe = date(2025, 9, 30)
    for t in ("300750.SZ", "002594.SZ"):
        want = f"{calc.growth(t, pe)['revenue_yoy'].value * 100:.2f}%"
        row = next(r for r in rep["comparison"]["rows"] if r["metric"] == "营业收入同比")
        assert row["values"][t] == want


def test_a_single_company_question_is_not_a_comparison():
    rep = run_research(ResearchRequest(ticker="300750.SZ", as_of=A, question="宁德时代的盈利质量如何", data_mode="offline_golden"),
                       "B4", persist=False)
    assert rep["comparison"] is None and not [c for c in rep["claims"].values() if c["section"] == "comparison"]


def test_comparison_names_at_most_four_companies():
    q = "对比宁德时代、比亚迪、隆基绿能、贵州茅台、五粮液、美的集团的成长性"
    rep = _compare(q)
    assert len(rep["comparison"]["tickers"]) == 4


def test_explicit_tickers_field_is_honoured():
    rep = run_research(ResearchRequest(ticker="002594.SZ", tickers=["002594.SZ", "000858.SZ"], as_of=A, question="比较两家公司",
                                       data_mode="offline_golden"), "B4", persist=False)
    assert [t["ticker"] for t in rep["comparison"]["tickers"]] == ["002594.SZ", "000858.SZ"]


@pytest.mark.parametrize("arch,compares", [("B3", True), ("B2", False), ("B1", False)])
def test_graph_variants_compare_and_scripted_baselines_do_not(arch, compares):
    rep = run_research(ResearchRequest(ticker="300750.SZ", as_of=A, question="对比宁德时代与比亚迪的盈利质量", data_mode="offline_golden"),
                       arch, persist=False)
    assert bool(rep.get("comparison")) is compares

# ------------------------------------------------------------------ what counts as evidence
def test_a_sentence_needs_a_topic_specific_word_not_just_generic_ones():
    from agent_runtime.agents.explorer import best_sentence

    terms = ["境外", "收入", "占比", "关税"]
    assert best_sentence("报告期内，受益于终端客户需求，公司营业收入与净利润较上年同期大幅增长，占比持续提高。", terms) is None
    got = best_sentence("公司境外收入占比为90.58%，美国关税政策存在不确定性。", terms)
    assert got and "境外" in got


def test_a_table_row_is_quoted_with_its_column_headings_and_the_literal_line_is_the_focus():
    from agent_runtime.agents.explorer import best_row

    table = ("| 项目 | 2025年 | 占比 |\n| --- | --- | --- |\n| 境外 | 34636523292.07 | 90.58% |\n| 境内 | 3000000000.00 | 9.42% |\n")
    text, focus = best_row(table, ["境外", "收入", "占比", "关税"]) or ("", "")
    assert "境外：2025年 34636523292.07，占比 90.58%" in text and "单位见原表" in text
    assert focus == "| 境外 | 34636523292.07 | 90.58% |" and focus in table
    assert best_row(table, ["前五名客户", "销售额"]) is None


def test_the_judge_reads_the_part_of_the_evidence_that_the_claim_quotes():
    from agent_runtime.agents.verifier import judge_excerpt

    ex = "ANCHOR\n…" + "无关内容。" * 300 + "\n| 在建工程 | 5495242187.20 | 286.46% |\n" + "尾部。" * 300
    out = judge_excerpt("在建工程 5495242187.20", ex)
    assert "5495242187.20" in out and len(out) <= 520
    assert judge_excerpt("短", "很短的证据") == "很短的证据"