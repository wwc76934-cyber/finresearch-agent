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
SOURCE_PRIORITY = {"demo_filing": 1, "demo_vendor_b": 5}


# ---------------------------------------------------------------- oracle
class Oracle:
    def __init__(self):
        f = pd.read_csv(GOLDEN / "financial_facts.csv", parse_dates=["period_end", "published_at", "available_at"])
        f["prio"] = f["source"].map(SOURCE_PRIORITY).fillna(9)
        self.f = f
        self.p = pd.read_csv(GOLDEN / "market_prices.csv", parse_dates=["trade_date", "available_at"])
        self.docs = pd.read_csv(GOLDEN / "documents.csv", parse_dates=["published_at", "available_at"])

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
    kw = [d["title"] for d in fut if "公告" in d["title"]]
    return {"doc_ids": [d["document_id"] for d in fut], "keywords": kw}


def build_tasks() -> tuple[list[dict], list[dict]]:
    o = Oracle()
    golden, chaos = [], []
    gid = iter(range(1, 1000))

    def g(cat, t, as_of, question, expected, mode="deep", trap="none"):
        golden.append({"task_id": f"G{next(gid):03d}", "dataset": "golden", "category": cat, "trap_type": trap,
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


def write_tasks() -> dict:
    golden, chaos = build_tasks()
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("golden_v1.jsonl", golden), ("chaos_v1.jsonl", chaos)):
        with open(TASK_DIR / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    with get_session_factory()() as s:
        for r in golden + chaos:
            s.merge(BenchmarkTaskRow(task_id=r["task_id"], dataset=r["dataset"], version=VERSION, snapshot_id=None,
                                     trap_type=r["category"], payload=json.loads(json.dumps(r, default=str))))
        s.commit()
    return {"golden": len(golden), "chaos": len(chaos)}


def load_tasks(datasets=("golden", "chaos"), path: Path = TASK_DIR) -> list[dict]:
    out = []
    for d in datasets:
        p = path / f"{d}_v1.jsonl"
        out += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out
