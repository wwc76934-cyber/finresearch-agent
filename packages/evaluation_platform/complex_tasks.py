"""complex_open_20 (v2.0 Step 5): 10 open, cross-dimension questions on a company's non-standard disclosures + 10 multi-company /
cross-period tasks. Every expected answer is derived independently of the agent:

* open questions (K001-K010): which topics the filings actually cover is found by searching the stored report text itself
  (regexes on the markdown of the latest periodic reports visible at as_of). The task is only built for topics the filings
  contain, and the agent must turn each such topic into a cited claim.
* comparisons (K011-K016): the matrix values come from FinancialCalculator called directly.
* reviews (K017-K020): a first deep run at an earlier as_of leaves a thesis; the expected status of each falsification
  condition is computed here from the calculators with the condition rules written out again (not from agent code).
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from data_platform.db import ROOT

REAL = ROOT / "benchmarks" / "real_snapshot"
CORE = [("300750.SZ", "宁德时代"), ("002594.SZ", "比亚迪"), ("601012.SH", "隆基绿能"), ("300014.SZ", "亿纬锂能"),
        ("600519.SH", "贵州茅台"), ("000858.SZ", "五粮液"), ("000333.SZ", "美的集团"), ("000651.SZ", "格力电器")]
# topic -> (question fragment, hypothesis statement from agents/hypotheses.py, regex a filing line must match)
TOPICS: dict[str, tuple[str, str, str]] = {
    "customers": ("客户集中度是否过高", "客户集中度与对大客户的依赖", r"前五名客户"),
    "overseas": ("海外业务与关税风险", "海外业务与海外建厂的规模、占比和政策风险", r"^\|\s*(境外|海外)\s*\||(境外|海外)[^|\n]{0,12}收入"),
    "capex": ("资本开支与在建工程转固进度", "资本开支向产能与自由现金流的转化效率", r"^\|\s*在建工程\s*\||购建固定资产"),
    "inventory": ("存货跌价准备计提是否充分", "存货减值计提的充足性", r"存货跌价准备"),
    "rnd": ("研发投入强度与资本化政策", "研发投入强度与资本化政策", r"研发投入资本化|资本化[^|\n]{0,10}研发|研发[^|\n]{0,10}资本化"),
    "cash": ("利润的现金含量与回款质量", "利润的现金含量与回款质量", r"经营活动产生的现金流量净额[^|\n]{0,40}(主要|原因)"),
    "rebate": ("销售返利与价格调整安排", "返利、折扣或价格补偿安排对收入和利润的影响", r"返利"),
}
ORDER = ["customers", "overseas", "capex", "inventory", "cash", "rnd", "rebate"]


def _latest_reports(ticker: str, as_of: datetime) -> list[Path]:
    docs = pd.read_csv(REAL / "documents.csv", parse_dates=["available_at"])
    d = docs[(docs.ticker == ticker) & (docs.doc_type != "announcement") & (docs.available_at <= as_of)
             & ~docs.document_id.str.contains("_v")].sort_values("available_at")
    annual = d[d.doc_type == "annual_report"].tail(1)
    latest = d.tail(1)
    return [ROOT / "benchmarks" / p for p in pd.concat([annual, latest]).path.drop_duplicates()]


def _topics_present(ticker: str, as_of: datetime) -> list[str]:
    text = "\n".join(p.read_text(encoding="utf-8") for p in _latest_reports(ticker, as_of) if p.exists())
    return [k for k in ORDER if re.search(TOPICS[k][2], text, re.M)]


def build_open_tasks() -> list[dict]:
    plan = [(t, n, datetime(2025, 12, 31, 15)) for t, n in CORE] + [(CORE[0][0], CORE[0][1], datetime(2026, 5, 6, 15)),
                                                                     (CORE[4][0], CORE[4][1], datetime(2026, 5, 6, 15))]
    rows: list[dict] = []
    for i, (t, name, as_of) in enumerate(plan):
        present = _topics_present(t, as_of)
        pick = [k for k in present if k not in ("rebate",)][i % 3:] + [k for k in present if k not in ("rebate",)][:i % 3]
        pick = pick[:3]
        if len(pick) < 2:
            continue                                   # a company whose filings hold fewer than two of the topics is skipped
        frags = "；".join(TOPICS[k][0] for k in pick)
        rows.append({"task_id": f"K{len(rows) + 1:03d}", "dataset": "complex", "category": "complex_open", "trap_type": "open_cross_dimension",
                     "ticker": t, "as_of": as_of.isoformat(), "mode": "deep", "fault_injection": {},
                     "question": f"分析{name}：{frags}，并给出多空观点和证伪条件。",
                     "expected": {"hypotheses": {"min": len(pick), "must_find": [TOPICS[k][1] for k in pick], "topics": pick}}})
    return rows


def _fmt_pct(v: Optional[float]) -> str:
    return "数据不足" if v is None else f"{v * 100:.2f}%"


def _comparison_expectation(tickers: list[str], as_of: datetime) -> dict[str, Any]:
    from agent_runtime.context import fmt
    from data_platform import FinancialCalculator, LineageRecorder, PITDataService
    from data_platform.serving import period_label

    svc = PITDataService(as_of, "oracle", data_mode="offline_golden")
    calc = FinancialCalculator(svc, LineageRecorder(None, "oracle"))
    ends = [svc.latest_period(t) for t in tickers]
    assert all(e is not None for e in ends), tickers
    pe: date = min(e for e in ends if e is not None)
    recs = {t: {"ratios": calc.ratios(t, pe), "growth": calc.growth(t, pe), "valuation": calc.valuation(t)} for t in tickers}
    spec = [("成长性", "营业收入同比", "growth", "revenue_yoy"), ("成长性", "归母净利润同比", "growth", "net_profit_yoy"),
            ("盈利能力", "毛利率", "ratios", "gross_margin"), ("盈利能力", "净利率", "ratios", "net_margin"),
            ("盈利能力", "年化ROE", "ratios", "roe_annualized"), ("盈利质量", "经营现金流/净利润", "ratios", "ocf_to_net_profit"),
            ("财务结构", "资产负债率", "ratios", "debt_to_assets"), ("估值", "PE-TTM", "valuation", "pe_ttm"),
            ("估值", "PB-LF", "valuation", "pb_lf"), ("估值", "PE近3年分位", "valuation", "pe_percentile_3y")]
    rows = [{"dimension": d, "metric": m, "values": {t: (fmt(recs[t][fam][k]) if k in recs[t][fam] else "数据不足") for t in tickers}}
            for d, m, fam, k in spec]
    return {"tickers": tickers, "period": period_label(pe), "rows": rows}


def build_comparison_tasks(start: int) -> list[dict]:
    as_of = datetime(2025, 12, 31, 15)
    sets = [(["300750.SZ", "002594.SZ"], "对比宁德时代与比亚迪的盈利质量、成长性和估值"),
            (["600519.SH", "000858.SZ"], "对比贵州茅台与五粮液的盈利能力和估值水平"),
            (["000333.SZ", "000651.SZ"], "对比美的集团与格力电器的财务结构和盈利质量"),
            (["300750.SZ", "300014.SZ", "601012.SH"], "对比宁德时代、亿纬锂能和隆基绿能的成长性与估值"),
            (["600519.SH", "000333.SZ"], "对比贵州茅台与美的集团的盈利能力和现金流质量"),
            (["002594.SZ", "601012.SH", "000651.SZ"], "对比比亚迪、隆基绿能和格力电器的财务结构、成长性与估值")]
    return [{"task_id": f"K{start + i:03d}", "dataset": "complex", "category": "multi_compare", "trap_type": "peer_comparison",
             "ticker": ts[0], "as_of": as_of.isoformat(), "mode": "deep", "fault_injection": {}, "question": q,
             "expected": {"comparison": _comparison_expectation(ts, as_of)}} for i, (ts, q) in enumerate(sets)]


# the four conditions a full run writes (rules restated here, independently of agents/debate.py): metric, direction, threshold rule
def _threshold(metric: str, base: float) -> float:
    return {"gross_margin": base - 0.02, "ocf_to_net_profit": 0.8, "receivable_days": max(base * 1.1, base + 5),
            "interest_debt_ratio": base + 0.05}[metric]


def _expected_review(ticker: str, old_asof: datetime, new_asof: datetime) -> dict[str, str]:
    from data_platform import FinancialCalculator, LineageRecorder, PITDataService

    svc = PITDataService(new_asof, "oracle", data_mode="offline_golden")
    calc = FinancialCalculator(svc, LineageRecorder(None, "oracle"))
    old_pe = PITDataService(old_asof, "oracle", data_mode="offline_golden").latest_period(ticker)
    new_pe = svc.latest_period(ticker)
    assert old_pe is not None and new_pe is not None, ticker
    old, new = calc.ratios(ticker, old_pe), calc.ratios(ticker, new_pe)
    out: dict[str, str] = {}
    for m, lt in (("gross_margin", True), ("ocf_to_net_profit", True), ("receivable_days", False), ("interest_debt_ratio", False)):
        bv, nv = (old[m].value if m in old else None), (new[m].value if m in new else None)
        if bv is None or nv is None:
            continue
        b, n = bv, nv
        thr = _threshold(m, b)
        if (lt and n < thr) or (not lt and n > thr):
            out[m] = "FALSIFIED"
        else:
            progress = (b - n) if lt else (n - b)
            out[m] = "WEAKENED" if progress / abs(b - thr) >= 0.5 else "CONFIRMED"
    return out


def build_review_tasks(start: int) -> list[dict]:
    old, new = datetime(2025, 6, 30, 15), datetime(2025, 12, 31, 15)
    rows = []
    for i, (t, name) in enumerate([("300750.SZ", "宁德时代"), ("600519.SH", "贵州茅台"), ("002594.SZ", "比亚迪"), ("000333.SZ", "美的集团")]):
        rows.append({"task_id": f"K{start + i:03d}", "dataset": "complex", "category": "thesis_review", "trap_type": "cross_period_review",
                     "ticker": t, "as_of": new.isoformat(), "mode": "deep", "fault_injection": {},
                     "question": f"{name}：全面研究基本面、估值与风险，并复盘我此前给出的观点和证伪条件是否被新披露的数据证实或证伪。",
                     "prior_run": {"as_of": old.isoformat(), "question": "全面研究公司基本面、估值与风险，形成包含多空审议与证伪条件的研究报告"},
                     "expected": {"review": _expected_review(t, old, new), "prior_period": "2025Q1", "new_period": "2025Q3"}})
    return rows


def build_complex_tasks() -> list[dict]:
    rows = build_open_tasks()
    rows += build_comparison_tasks(len(rows) + 1)
    rows += build_review_tasks(len(rows) + 1)
    return rows
