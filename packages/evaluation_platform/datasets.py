"""Benchmark Dataset Registry: 30 golden + 50 chaos tasks with an independent oracle.

The oracle recomputes ground truth directly from the snapshot CSV with pandas,
sharing no code with data_platform.calculators, so a calculator bug shows up
as an evaluation failure instead of being copied into the answer key.
Anomaly codes come from the scenario design in build_golden_snapshot.py.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from data_platform.db import ROOT, BenchmarkTaskRow, get_session_factory

GOLDEN = ROOT / "benchmarks" / "golden_snapshot"
TASK_DIR = ROOT / "benchmarks" / "tasks"
VERSION = "v1"
TICKERS = ["990001.SZ", "990002.SZ", "990003.SH", "990004.SH", "990005.SZ", "990006.SZ"]
NAMES = {"990001.SZ": "演示动力电池", "990002.SZ": "演示新能源整车", "990003.SH": "演示锂电材料", "990004.SH": "演示白酒",
         "990005.SZ": "演示光伏组件", "990006.SZ": "演示储能系统"}
SOURCE_PRIORITY = {"demo_filing": 1, "eastmoney_f10": 2, "demo_vendor_b": 5, "cninfo_pdf_parse": 6}
REAL = ROOT / "benchmarks" / "real_snapshot"


# ---------------------------------------------------------------- oracle
class Oracle:
    """Independent ground truth over both snapshots (pandas, shares no code with the calculators)."""

    def __init__(self):
        dirs = [d for d in (GOLDEN, REAL) if (d / "financial_facts.csv").exists()]
        rd = lambda n, dates: pd.concat([pd.read_csv(d / n, parse_dates=dates) for d in dirs], ignore_index=True)  # noqa: E731
        f = rd("financial_facts.csv", ["period_end", "published_at", "available_at"])
        f["prio"] = f["source"].map(SOURCE_PRIORITY).fillna(9)
        self.f = f
        self.p = rd("market_prices.csv", ["trade_date", "available_at"])
        self.docs = rd("documents.csv", ["published_at", "available_at"])

    def _row(self, t, m, pe, as_of, scope="consolidated"):
        d = self.f[(self.f.ticker == t) & (self.f.metric == m) & (self.f.statement_scope == scope)
                   & (self.f.period_end == pd.Timestamp(pe)) & (self.f.available_at <= pd.Timestamp(as_of))
                   & (self.f.published_at_reliable)]
        if d.empty:
            return None
        return d.sort_values(["prio", "revision_id"], ascending=[True, False]).iloc[0]

    def val(self, t, m, pe, as_of, scope="consolidated") -> Optional[float]:
        r = self._row(t, m, pe, as_of, scope)
        return None if r is None else float(r.value)

    def latest(self, t, as_of) -> date:
        d = self.f[(self.f.ticker == t) & (self.f.available_at <= pd.Timestamp(as_of))]
        return d.period_end.max().date()

    @staticmethod
    def prev_q(pe: date) -> Optional[date]:
        return {6: date(pe.year, 3, 31), 9: date(pe.year, 6, 30), 12: date(pe.year, 9, 30)}.get(pe.month)

    def single_q(self, t, m, pe, as_of):
        cur = self._row(t, m, pe, as_of)
        pq = self.prev_q(pe)
        if cur is None:
            return None, "INSUFFICIENT_DATA"
        if pq is None:
            return float(cur.value), "OK"
        prev = self._row(t, m, pq, as_of)
        if prev is None:
            return None, "INSUFFICIENT_DATA"
        if prev.published_at > cur.published_at:
            return None, "RESTATEMENT_BREAKS_SPLIT"
        return float(cur.value - prev.value), "OK"

    def yoy(self, t, m, pe, as_of):
        cur, base = self.val(t, m, pe, as_of), self.val(t, m, date(pe.year - 1, pe.month, pe.day), as_of)
        if cur is None or base is None:
            return None, "INSUFFICIENT_DATA"
        if base <= 0:
            return None, "NEGATIVE_EARNINGS_NA"
        return cur / base - 1, "OK"

    def ttm(self, t, m, pe, as_of):
        if pe.month == 12:
            return self.val(t, m, pe, as_of)
        a, b, c = (self.val(t, m, date(pe.year - 1, 12, 31), as_of), self.val(t, m, date(pe.year - 1, pe.month, pe.day), as_of),
                   self.val(t, m, pe, as_of))
        return None if None in (a, b, c) else a - b + c

    def pe_ttm(self, t, as_of):
        px = self.p[(self.p.ticker == t) & (self.p.available_at <= pd.Timestamp(as_of))].sort_values("trade_date")
        pe = self.latest(t, as_of)
        np_ttm = self.ttm(t, "net_profit", pe, as_of)
        if px.empty or np_ttm is None:
            return None, "INSUFFICIENT_DATA"
        if np_ttm <= 0:
            return None, "NEGATIVE_EARNINGS_NA"
        return float(px.iloc[-1].total_mv) / np_ttm, "OK"

    def future_docs(self, t, as_of) -> list[dict]:
        d = self.docs[(self.docs.ticker == t) & (self.docs.available_at > pd.Timestamp(as_of))]
        return d[["document_id", "title"]].to_dict("records")

    def visible_announcements(self, t, as_of) -> list[str]:
        d = self.docs[(self.docs.ticker == t) & (self.docs.doc_type == "announcement")
                      & (self.docs.available_at <= pd.Timestamp(as_of))]
        return d.document_id.tolist()


def label(pe: date) -> str:
    return f"{pe.year}Q{(pe.month - 1) // 3 + 1}"


def num(key, metric, pe, value, status="OK", scope="consolidated", unit=None, tol_rel=1e-4) -> dict:
    return {"key": key, "metric": metric, "period": label(pe) if isinstance(pe, date) else pe, "scope": scope,
            "value": value, "status": status if value is not None or status != "OK" else "INSUFFICIENT_DATA",
            "unit": unit, "tol_rel": tol_rel}


# designed anomaly scenarios (truth by construction, not by detector output)
DESIGNED_ANOMALIES = {"990001.SZ": ["PROFIT_OCF_DIVERGENCE", "AR_OUTPACE_REVENUE"], "990002.SZ": ["GROSS_MARGIN_SHIFT"],
                      "990003.SH": ["LARGE_IMPAIRMENT", "INVENTORY_DAYS_DETERIORATION"], "990004.SH": [],
                      "990005.SZ": ["INTEREST_DEBT_JUMP"], "990006.SZ": ["RELATED_PARTY_OR_POLICY_CHANGE"]}


def _forbidden(o: Oracle, t, as_of) -> dict:
    fut = o.future_docs(t, as_of)
    seen = set(o.docs[(o.docs.ticker == t) & (o.docs.available_at <= pd.Timestamp(as_of))].title)
    # recurring titles (e.g. monthly 回购进展公告) also exist before as_of, so they prove nothing as keywords
    kw = [d["title"] for d in fut if "公告" in d["title"] and d["title"] not in seen]
    return {"doc_ids": [d["document_id"] for d in fut], "keywords": kw}


def build_tasks() -> tuple[list[dict], list[dict]]:
    o = Oracle()
    golden, chaos = [], []
    gid = iter(range(1, 1000))

    def g(cat, t, as_of, question, expected, mode="deep", trap="none"):
        golden.append({"task_id": f"L{next(gid):03d}", "dataset": "lab", "category": cat, "trap_type": trap,
                       "ticker": t, "as_of": as_of.isoformat(), "question": question, "mode": mode,
                       "fault_injection": {}, "expected": expected})

    def key_facts(t, as_of):
        pe = o.latest(t, as_of)
        rev, np_ = o.val(t, "operating_revenue", pe, as_of), o.val(t, "net_profit", pe, as_of)
        cost, ocf = o.val(t, "operating_cost", pe, as_of), o.val(t, "operating_cash_flow", pe, as_of)
        ry, rys = o.yoy(t, "operating_revenue", pe, as_of)
        ny, nys = o.yoy(t, "net_profit", pe, as_of)
        out = [num("revenue", "operating_revenue", pe, rev), num("revenue_yoy", "revenue_yoy", pe, ry, rys),
               num("net_profit", "net_profit", pe, np_), num("net_profit_yoy", "net_profit_yoy", pe, ny, nys),
               num("gross_margin", "gross_margin", pe, (rev - cost) / rev)]
        out.append(num("ocf_to_np", "ocf_to_net_profit", pe, ocf / np_ if np_ > 0 else None,
                       "OK" if np_ > 0 else "NEGATIVE_EARNINGS_NA"))
        return out

    deep_q = "以研究时点为准，全面研究{n}的基本面、行业竞争、近期公告、财务风险与估值，形成包含多空审议、证伪条件与数据血缘的研究报告。"
    for t, as_of in [(t, datetime(2025, 12, 31, 15)) for t in TICKERS] + \
                    [(t, datetime(2025, 9, 1, 15)) for t in ("990001.SZ", "990002.SZ", "990003.SH", "990006.SZ")]:
        g("deep_research", t, as_of, deep_q.format(n=NAMES[t]),
          {"numbers": key_facts(t, as_of), "forbidden": _forbidden(o, t, as_of), "rubric": True,
           "latest_period": label(o.latest(t, as_of))})
    an_q = "公司最新季报中哪些核心财务指标发生背离或异常变动？结合管理层讨论与专项公告分析潜在风险信号。"
    for t, as_of in [(t, datetime(2025, 12, 31, 15)) for t in TICKERS] + \
                    [(t, datetime(2025, 11, 3, 15)) for t in ("990001.SZ", "990003.SH", "990005.SZ", "990004.SH")]:
        exp = [c for c in DESIGNED_ANOMALIES[t] if not (c == "RELATED_PARTY_OR_POLICY_CHANGE" and as_of < datetime(2025, 9, 16))]
        if t == "990003.SH" and as_of >= datetime(2025, 11, 21):
            exp = exp + ["RELATED_PARTY_OR_POLICY_CHANGE"]  # the correction announcement is visible
        g("anomaly_attribution", t, as_of, an_q,
          {"anomalies": exp, "forbidden": _forbidden(o, t, as_of), "numbers": key_facts(t, as_of)[:2]}, mode="deep")
    ev_q = "结合近期公告分析{n}的重大事项及其影响，并与同行估值进行对比。"
    for t, as_of in [(t, datetime(2025, 12, 31, 15)) for t in TICKERS] + \
                    [("990001.SZ", datetime(2026, 1, 25, 15)), ("990003.SH", datetime(2025, 11, 25, 15)),
                     ("990002.SZ", datetime(2025, 7, 15, 15)), ("990006.SZ", datetime(2025, 9, 20, 15))]:
        pe_v, pe_s = o.pe_ttm(t, as_of)
        g("events_peers", t, as_of, ev_q.format(n=NAMES[t]),
          {"numbers": [num("pe_ttm", "pe_ttm", as_of.date().isoformat(), pe_v, pe_s, tol_rel=1e-3)],
           "events": o.visible_announcements(t, as_of), "forbidden": _forbidden(o, t, as_of)})

    # ------------------------------------------------ chaos
    cid = iter(range(1, 1000))

    def c(cat, t, as_of, question, expected, fault=None, mode="fast"):
        chaos.append({"task_id": f"X{next(cid):03d}", "dataset": "chaos", "category": cat, "trap_type": cat,
                      "ticker": t, "as_of": as_of.isoformat(), "question": question, "mode": mode,
                      "fault_injection": fault or {}, "expected": {"forbidden": _forbidden(o, t, as_of), **expected}})

    A = datetime(2025, 12, 31, 15)
    q3 = date(2025, 9, 30)
    for t in TICKERS:  # 6 single-quarter splits
        v, s = o.single_q(t, "operating_revenue", q3, A)
        c("calc_trap", t, A, "2025年第三季度单季营业收入是多少亿元？", {"numbers": [num("sq_rev", "single_quarter:operating_revenue", q3, v, s)]})
    for t in ("990001.SZ", "990002.SZ", "990004.SH"):  # 3 parent vs consolidated
        c("calc_trap", t, A, "2025年第三季度母公司口径累计营业收入是多少亿元？",
          {"numbers": [num("parent_rev", "operating_revenue", q3, o.val(t, "operating_revenue", q3, A, "parent"), scope="parent")]})
    for t in ("990005.SZ", "990004.SH"):  # 2 loss-maker PE handling
        v, s = o.pe_ttm(t, A)
        c("calc_trap", t, A, "当前PE-TTM是多少倍？", {"numbers": [num("pe", "pe_ttm", A.date().isoformat(), v, s, tol_rel=1e-3)]})
    for t in ("990001.SZ", "990006.SZ"):  # 2 unit conversion
        c("calc_trap", t, A, "2025年第三季度累计归母净利润是多少万元？",
          {"numbers": [num("np_wan", "net_profit", q3, o.val(t, "net_profit", q3, A), unit="万元")]})
    for t, sticky in (("990002.SZ", 0), ("990006.SZ", 3)):  # 2 injected fake numbers
        rev, cost = o.val(t, "operating_revenue", q3, A), o.val(t, "operating_cost", q3, A)
        c("calc_trap", t, A, "2025年第三季度毛利率是多少？", {"numbers": [num("gm", "gross_margin", q3, (rev - cost) / rev)],
                                                     "fault": "fake_number", "expect_downgrade": sticky > 0},
          fault={"fake_number": {"sticky": sticky}})

    # PIT traps (15)
    B = datetime(2025, 10, 28, 21, 0)  # Q3 published 20:00, available next day 09:30
    for t in TICKERS:
        pe = o.latest(t, B)
        v, s = o.yoy(t, "operating_revenue", pe, B)
        c("pit_trap", t, B, "最新一期营收同比是多少？", {"numbers": [num("rev_yoy", "revenue_yoy", pe, v, s)],
                                                "latest_period": label(pe)})
    c("pit_trap", "990001.SZ", A, "公司是否存在监管立案或处罚风险？",
      {"anomalies": [], "fault": "future_evidence"}, fault={"future_evidence": {"document_id": "ann_990001_20260120"}}, mode="deep")
    c("pit_trap", "990001.SZ", A, "全面评估公司治理与合规风险", {"fault": "future_evidence"},
      fault={"future_evidence": {}}, mode="deep")
    C = datetime(2026, 1, 15, 15)
    for i, t in enumerate(TICKERS):
        pe = o.latest(t, C)
        v = o.val(t, "net_profit", pe, C)
        fault = {"future_evidence": {}} if i < 3 else None
        c("pit_trap", t, C, "最新一期累计归母净利润是多少亿元？",
          {"numbers": [num("np", "net_profit", pe, v)], "latest_period": label(pe),
           **({"fault": "future_evidence"} if fault else {})}, fault=fault)
    D = datetime(2025, 4, 26, 22, 0)
    pe = o.latest("990004.SH", D)
    c("pit_trap", "990004.SH", D, "最新一期营业收入是多少亿元？",
      {"numbers": [num("rev", "operating_revenue", pe, o.val("990004.SH", "operating_revenue", pe, D))], "latest_period": label(pe)})

    # conflict / restatement (10)
    for as_of in (datetime(2025, 11, 19, 15), datetime(2025, 11, 25, 15)):
        c("conflict_trap", "990003.SH", as_of, "2025年第三季度累计归母净利润是多少亿元？",
          {"numbers": [num("np", "net_profit", q3, o.val("990003.SH", "net_profit", q3, as_of))]})
        v, s = o.yoy("990003.SH", "net_profit", q3, as_of)
        c("conflict_trap", "990003.SH", as_of, "2025年第三季度净利润同比是多少？",
          {"numbers": [num("np_yoy", "net_profit_yoy", q3, v, s)]})
    q324 = date(2024, 9, 30)
    for as_of in (datetime(2025, 6, 30, 15), datetime(2024, 11, 1, 15)):
        v, s = o.single_q("990002.SZ", "operating_revenue", q324, as_of)
        c("conflict_trap", "990002.SZ", as_of, "2024年第三季度单季营业收入是多少亿元？",
          {"numbers": [num("sq", "single_quarter:operating_revenue", q324, v, s)]})
    q125 = date(2025, 3, 31)
    E = datetime(2025, 6, 30, 15)
    c("conflict_trap", "990002.SZ", E, "2025年第一季度营业收入是多少亿元？",
      {"numbers": [num("rev", "operating_revenue", q125, o.val("990002.SZ", "operating_revenue", q125, E))], "conflict": True})
    c("conflict_trap", "990002.SZ", E, "2025年第一季度营业收入是多少亿元？",
      {"numbers": [num("rev", "operating_revenue", q125, o.val("990002.SZ", "operating_revenue", q125, E))], "conflict": True,
       "fault": "tool_error"}, fault={"tool_error": {"tool": "get_financial_statements", "remaining": 2}})
    h124 = date(2024, 6, 30)
    for as_of in (datetime(2024, 11, 1, 15), datetime(2025, 1, 15, 15)):
        c("conflict_trap", "990002.SZ", as_of, "2024年半年度营业收入是多少亿元？",
          {"numbers": [num("rev", "operating_revenue", h124, o.val("990002.SZ", "operating_revenue", h124, as_of))]})

    # refusal (10)
    refusals = ["下一季度营业收入预测是多少？", "应收账款前五名中客户A的具体名称是什么？", "公司目前在手订单金额是多少？",
                "公司未披露的细分产品毛利率是多少？", "公司的目标价是多少？"]
    for i, q in enumerate(refusals * 2):
        t = TICKERS[i % 6]
        c("refusal_trap", t, A, q, {"must_refuse": True})
    return golden, chaos


# ---------------------------------------------------------------- real A-share golden set (Tier-A)
def real_securities() -> dict[str, str]:
    p = REAL / "securities.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    return dict(zip(d.ticker, d.name))


def oracle_anomalies(o: Oracle, t: str, as_of: datetime) -> list[str]:
    """Independent pandas implementation of the 7 anomaly rules as written in PRD 2.2 / calculators' spec."""
    pe = o.latest(t, as_of)
    ya = date(pe.year - 1, pe.month, pe.day)
    v = lambda m, p=pe: o.val(t, m, p, as_of)  # noqa: E731
    out = []
    ny, _ = o.yoy(t, "net_profit", pe, as_of)
    oy, _ = o.yoy(t, "operating_cash_flow", pe, as_of)
    if ny is not None and oy is not None and ny > 0.2 and oy < -0.3:
        out.append("PROFIT_OCF_DIVERGENCE")
    ar, ar0 = v("accounts_receivable"), v("accounts_receivable", ya)
    ry, _ = o.yoy(t, "operating_revenue", pe, as_of)
    rev_ttm = o.ttm(t, "operating_revenue", pe, as_of)
    material = not rev_ttm or ar is None or ar / rev_ttm >= 0.01   # receivables at least 1% of TTM revenue
    if ar and ar0 and ry is not None and ar0 > 0 and (ar / ar0 - 1) - ry > 0.3 and material:
        out.append("AR_OUTPACE_REVENUE")
    cost_ttm, cost_ttm0 = o.ttm(t, "operating_cost", pe, as_of), o.ttm(t, "operating_cost", ya, as_of)
    inv, inv0 = v("inventory"), v("inventory", ya)
    if inv and inv0 and cost_ttm and cost_ttm0 and (inv / cost_ttm) / (inv0 / cost_ttm0) - 1 > 0.3:
        out.append("INVENTORY_DAYS_DETERIORATION")
    gms, d = [], pe
    for _ in range(8):
        r, _s = o.single_q(t, "operating_revenue", d, as_of)
        c, _s2 = o.single_q(t, "operating_cost", d, as_of)
        if r and c is not None:
            gms.append((r - c) / r)
        d = Oracle.prev_q(d) or date(d.year - 1, 12, 31)
    for idx in (0, 1):
        if idx + 1 >= len(gms):
            break
        cur, prev, hist = gms[idx], gms[idx + 1], gms[idx + 1:]
        sd = float(pd.Series(hist).std(ddof=0)) if len(hist) > 2 else 0.0
        if abs(cur - prev) > 0.04 or (sd > 0 and abs(cur - sum(hist) / len(hist)) > 2 * sd and abs(cur - prev) > 0.02):
            out.append("GROSS_MARGIN_SHIFT")
            break
    imp, np_, ta = v("asset_impairment_loss"), v("net_profit"), v("total_assets")
    if imp is not None and np_ is not None and ta and (imp / max(abs(np_), 1) > 0.3 or imp / ta > 0.02):
        out.append("LARGE_IMPAIRMENT")
    d1, d0, a1, a0 = v("interest_bearing_debt"), v("interest_bearing_debt", ya), ta, v("total_assets", ya)
    if None not in (d1, d0, a1, a0) and a1 and a0 and d1 / a1 - d0 / a0 > 0.08:
        out.append("INTEREST_DEBT_JUMP")
    titles = o.docs[(o.docs.ticker == t) & (o.docs.doc_type == "announcement")
                    & (o.docs.available_at <= pd.Timestamp(as_of))].title.tolist()
    routine = ("日常关联交易", "财务公司", "管理制度", "管理办法", "决策制度", "交易制度", "交易办法")
    if any(k in x for x in titles if not any(r in x for r in routine) for k in ("关联交易", "会计估计变更", "会计差错")):
        out.append("RELATED_PARTY_OR_POLICY_CHANGE")
    return out


def build_real_golden(o: Oracle) -> list[dict]:
    names = real_securities()
    if not names:
        return []
    rows, gid = [], iter(range(1, 1000))
    tick = list(names)

    def g(cat, t, as_of, question, expected, mode="deep"):
        rows.append({"task_id": f"G{next(gid):03d}", "dataset": "golden", "category": cat, "trap_type": "none",
                     "ticker": t, "as_of": as_of.isoformat(), "question": question, "mode": mode, "fault_injection": {},
                     "expected": {"forbidden": _forbidden(o, t, as_of), **expected}})

    def key_facts(t, as_of):
        pe = o.latest(t, as_of)
        rev, np_ = o.val(t, "operating_revenue", pe, as_of), o.val(t, "net_profit", pe, as_of)
        cost, ocf = o.val(t, "operating_cost", pe, as_of), o.val(t, "operating_cash_flow", pe, as_of)
        ry, rys = o.yoy(t, "operating_revenue", pe, as_of)
        ny, nys = o.yoy(t, "net_profit", pe, as_of)
        out = [num("revenue", "operating_revenue", pe, rev), num("revenue_yoy", "revenue_yoy", pe, ry, rys),
               num("net_profit", "net_profit", pe, np_), num("net_profit_yoy", "net_profit_yoy", pe, ny, nys)]
        if rev and cost is not None:
            out.append(num("gross_margin", "gross_margin", pe, (rev - cost) / rev))
        if np_ is not None and ocf is not None:
            out.append(num("ocf_to_np", "ocf_to_net_profit", pe, ocf / np_ if np_ > 0 else None,
                           "OK" if np_ > 0 else "NEGATIVE_EARNINGS_NA"))
        return out

    A = datetime(2025, 12, 31, 15)
    deep_q = "以研究时点为准，全面研究{n}的基本面、行业竞争格局、近期公告、财务风险和估值水平，形成包含多空对抗审议、证伪条件与完整数据血缘追溯的投资研究报告。"
    for t, as_of in [(x, A) for x in tick] + [(tick[0], datetime(2025, 9, 1, 15)), (tick[4], datetime(2025, 9, 1, 15))]:
        g("deep_research", t, as_of, deep_q.format(n=names[t]),
          {"numbers": key_facts(t, as_of), "rubric": True, "latest_period": label(o.latest(t, as_of))})
    an_q = "公司最新一期定期报告中哪些核心财务指标发生背离或异常变动？结合管理层讨论与专项公告分析潜在风险信号。"
    for t, as_of in [(x, datetime(2025, 11, 3, 15)) for x in tick] + [(tick[1], datetime(2025, 9, 1, 15)),
                                                                       (tick[3], datetime(2025, 9, 1, 15))]:
        g("anomaly_attribution", t, as_of, an_q,
          {"anomalies": oracle_anomalies(o, t, as_of), "numbers": key_facts(t, as_of)[:2], "real": True})
    ev_q = "结合近期公告分析{n}的重大事项及其影响，并与同行估值进行对比。"
    for t, as_of in [(x, A) for x in tick] + [(tick[0], datetime(2026, 3, 13, 15)), (tick[4], datetime(2026, 4, 7, 15))]:
        pe_v, pe_s = o.pe_ttm(t, as_of)
        g("events_peers", t, as_of, ev_q.format(n=names[t]),
          {"numbers": [num("pe_ttm", "pe_ttm", as_of.date().isoformat(), pe_v, pe_s, tol_rel=1e-3)],
           "events": o.visible_announcements(t, as_of)})
    return rows


def _annual_has_note(path: str, word: str) -> bool:
    """True when the stored markdown has a note-table heading for `word` under the financial-statement notes."""
    p = ROOT / "benchmarks" / path
    if not p.exists():
        return False
    md = p.read_text(encoding="utf-8")
    i = md.find("## 财务报表附注")
    return i >= 0 and any(l.startswith("###") and word in l for l in md[i:].splitlines())


def build_multidoc_tasks(n: int = 15) -> list[dict]:
    """PRD v1.2 1.1: questions that need the annual-report notes, a later quarter's numbers and at least two
    announcements at once. The answer key is mechanical: Oracle numbers, plus the set of source documents a
    complete answer has to cite (the annual report and >=2 announcements public at as_of)."""
    o = Oracle()
    rows = []
    topics = [
        ("receivable", "应收账款周转天数", "应收账款的回收情况、账龄与客户集中度", "receivable_days"),
        ("inventory", "存货周转天数", "存货规模、跌价准备以及对毛利率和现金流的影响", "inventory_days"),
        ("debt", "有息负债率", "借款与融资安排、偿债压力", "interest_debt_ratio"),
    ]
    secs = real_securities()
    tickers = [t for t in secs if t[:2] != "99"]
    topic_word = {"receivable": "应收账款", "inventory": "存货", "debt": "借款"}
    seen = set()
    for i in range(n * 6):
        if len(rows) >= n:
            break
        t = tickers[i % len(tickers)]
        key, topic, ask, metric = topics[(i // len(tickers) + i) % len(topics)]
        anns = o.docs[(o.docs.ticker == t) & (o.docs.doc_type == "announcement")].sort_values("available_at")
        annual = o.docs[(o.docs.ticker == t) & (o.docs.doc_type == "annual_report") & (~o.docs.document_id.str.contains("_v"))]
        annual = annual[annual.document_id.str.endswith("2025Q4")]
        if len(anns) < 3 or annual.empty:
            continue
        # as_of: the day after the 4th-latest announcement, so >=3 announcements are visible and the annual report is older
        k = min(len(anns) - 1, 2 + (i // len(tickers)) * 2)
        as_of = (anns.iloc[k].available_at + pd.Timedelta(days=1)).replace(hour=15, minute=0, second=0)
        if as_of < annual.iloc[0].available_at or (t, as_of.date(), key) in seen:
            continue
        # an answer key must be satisfiable: the annual report has to contain a note table for this topic
        if not _annual_has_note(annual.iloc[0].path, topic_word[key]):
            continue
        seen.add((t, as_of.date(), key))
        pe = o.latest(t, as_of.to_pydatetime())
        if metric == "receivable_days":
            a, r = o.val(t, "accounts_receivable", pe, as_of), o.ttm(t, "operating_revenue", pe, as_of)
            cur = a / r * 365 if a is not None and r else None
        elif metric == "inventory_days":
            a, r = o.val(t, "inventory", pe, as_of), o.ttm(t, "operating_cost", pe, as_of)
            cur = a / r * 365 if a is not None and r else None
        else:
            a, r = o.val(t, "interest_bearing_debt", pe, as_of), o.val(t, "total_assets", pe, as_of)
            cur = a / r if a is not None and r else None
        if cur is None:
            continue
        visible = anns[anns.available_at <= as_of]
        name = secs[t]
        q = (f"结合{name}2025年年报附注、最新一期（{label(pe)}）的财务数据和近期公告，综合分析{ask}。"
             f"请给出最新的{topic}，并引用年报附注和至少两份近期公告作为依据。")
        rows.append({"task_id": f"M{len(rows) + 1:03d}", "dataset": "multidoc", "category": "multidoc_synthesis",
                     "trap_type": f"multi_source_{key}", "ticker": t, "as_of": as_of.isoformat(), "question": q, "mode": "deep",
                     "fault_injection": {},
                     "expected": {"numbers": [num("level", metric, pe, cur)],
                                  "forbidden": _forbidden(o, t, as_of.to_pydatetime()),
                                  "multidoc": {"annual": annual.iloc[0].document_id, "min_announcements": 2,
                                               "announcement_pool": visible.document_id.tolist()},
                                  "topic": key}})
    return rows


def build_quant_tasks() -> list[dict]:
    """Scenario D (PRD 2.4): 10 factor-validation tasks. Truth comes from quant_oracle (independent of the tool)."""
    from .quant_oracle import factor_stats

    rows: list[dict] = []
    nominal = "300750.SZ"

    def stats_numbers(as_of: str, st: dict) -> list[dict]:
        keys = [("n_cs", "factor_n_cs"), ("rank_ic_mean", "factor_rank_ic_mean"), ("ic_mean", "factor_ic_mean"),
                ("rank_ic_t", "factor_rank_ic_t"), ("rank_ic_p", "factor_rank_ic_p"), ("ls_cum", "factor_ls_cum")]
        return [num(k, m, as_of, st[k]) for k, m in keys if k in st]

    def add(cat, as_of, question, expected, trap="none"):
        rows.append({"task_id": f"Q{len(rows) + 1:03d}", "dataset": "quant", "category": cat, "trap_type": trap,
                     "ticker": nominal, "as_of": f"{as_of}T15:00:00", "question": question, "mode": "deep",
                     "fault_injection": {}, "expected": expected})

    normal = [
        ("ocf_to_net_profit", "all", "在过去2年新能源和大消费两个板块的样本公司中，经营现金流与净利润比值（盈利现金保障）越高，"
                                      "下一季度（60个交易日）的收益是否越好？请做三分组横截面因子检验。"),
        ("roe_ttm", "all", "在过去2年新能源和大消费样本公司里，ROE因子是否具有预测下一季度（60个交易日）收益的能力？请做三分组横截面检验。"),
        ("gross_margin_yoy_delta", "all", "过去2年新能源和大消费样本里，毛利率同比变动因子对下一季度（60个交易日）收益的RankIC表现如何？三分组。"),
        ("accruals_to_assets", "all", "过去2年新能源和大消费样本里，应计项目占总资产比越低盈利质量越高，这个因子能否预测下一季度（60个交易日）收益？三分组。"),
        ("receivable_turnover_yoy", "all", "过去2年新能源和大消费样本里，应收账款周转率同比变动因子是否有预测下一季度（60个交易日）收益的IC？三分组。"),
        ("pe_ttm_inverse", "all", "过去2年新能源和大消费样本里，盈利收益率（市盈率倒数）因子对下一季度（60个交易日）收益的分组表现如何？三分组。"),
    ]
    for fac, uni, q in normal:
        st = factor_stats(fac, uni, "2025-12-31", 8, 60, 3)
        add("factor_test", "2025-12-31", q, {"numbers": stats_numbers("2025-12-31", st), "factor": fac, "universe": uni})
    # as-of discipline: asked at 2024-06-30 only the cross-sections whose 60-day window ended by then may be used
    st = factor_stats("ocf_to_net_profit", "consumer", "2024-06-30", 8, 60, 3)
    add("factor_test", "2024-06-30", "站在2024年6月30日，大消费板块里经营现金流与净利润比值因子，过去2年对下一季度（60个交易日）收益的"
                                    "横截面RankIC如何？三分组。",
        {"numbers": stats_numbers("2024-06-30", st), "factor": "ocf_to_net_profit", "universe": "consumer",
         "note": "only cross-sections whose forward window ends by as_of are usable"}, trap="as_of_discipline")
    st = factor_stats("roe_ttm", "new_energy", "2025-12-31", 3, 20, 3)
    add("factor_sample_warning", "2025-12-31", "新能源板块里ROE因子过去3个季度对下一月（20个交易日）收益的RankIC如何？三分组。",
        {"numbers": stats_numbers("2025-12-31", st), "factor": "roe_ttm", "universe": "new_energy",
         "warnings": ["FEW_CROSS_SECTIONS", "SMALL_GROUPS", "SMALL_UNIVERSE"]}, trap="small_sample")
    add("factor_pit_trap", "2025-12-31", "在2024年3月31日这个截面，用2024年一季报的经营现金流与净利润比值因子，对新能源板块做回测，看下一季度收益。",
        {"must_refuse": True, "pit": "2024Q1 report filed after 2024-03-31"}, trap="pit_future_report")
    add("factor_pit_trap", "2025-12-31", "在2023年9月30日这个截面，用2023年三季报的ROE因子，对大消费板块做回测，看下一季度收益。",
        {"must_refuse": True, "pit": "2023Q3 report filed after 2023-09-30"}, trap="pit_future_report")
    return rows


def write_tasks() -> dict:
    lab, chaos = build_tasks()
    golden = build_real_golden(Oracle())
    quant = build_quant_tasks()
    multidoc = build_multidoc_tasks()
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("lab_v1.jsonl", lab), ("golden_v1.jsonl", golden), ("chaos_v1.jsonl", chaos),
                       ("quant_v1.jsonl", quant), ("multidoc_v1.jsonl", multidoc)):
        with open(TASK_DIR / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    with get_session_factory()() as s:
        for r in lab + golden + chaos + quant + multidoc:
            s.merge(BenchmarkTaskRow(task_id=r["task_id"], dataset=r["dataset"], version=VERSION, snapshot_id=None,
                                     trap_type=r["category"], payload=json.loads(json.dumps(r, default=str))))
        s.commit()
    return {"golden_real": len(golden), "chaos": len(chaos), "lab": len(lab), "quant": len(quant), "multidoc": len(multidoc)}


def load_tasks(datasets=("golden", "chaos"), path: Path = TASK_DIR) -> list[dict]:
    out = []
    for d in datasets:
        p = path / f"{d}_v1.jsonl"
        out += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out
