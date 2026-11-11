"""Deterministic financial calculators. LLMs never compute numbers; they cite MetricRecords.

Every output is a MetricRecord with its input fact ids, formula version and a
status code for economically meaningless cases. Lineage edges are recorded.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta
from statistics import median, pstdev
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from . import catalog
from .lineage import LineageRecorder
from .schema import MetricRecord, StatementScope
from .serving import PITDataService, label_to_end, period_label

CUM_MONTH = {3: 1, 6: 2, 9: 3, 12: 4}
# routine related-party filings (annual budget, finance-company deposits, internal rules): not a signal
ROUTINE_RPT_RE = re.compile(r"日常关联交易|财务公司|(制度|办法)(（|$)|管理制度|管理办法")


class AnomalyFlag(BaseModel):
    code: str
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    metric_ids: list[str]
    message: str
    suggested_query: str
    section_type: str = "NOTES"
    category: str = "rule"                    # "statistical" | "rule" | "semantic"
    statistical_z: Optional[float] = None     # vs the company's own history
    peer_z: Optional[float] = None            # vs peers in the same period
    evidence_quote: str = ""                  # semantic flags: the sentence that triggered it


def prev_quarter_end(d: date) -> Optional[date]:
    q = CUM_MONTH[d.month]
    return None if q == 1 else label_to_end(f"{d.year}Q{q - 1}")


def year_ago(d: date) -> date:
    return date(d.year - 1, d.month, d.day)


class FinancialCalculator:
    def __init__(self, svc: PITDataService, lineage: Optional[LineageRecorder] = None):
        self.svc = svc
        self.lineage = lineage
        self.store: dict[str, MetricRecord] = {}

    # ---------- record helpers ----------
    def _record(self, ticker: str, metric: str, period: str, value: Optional[float], unit: str, inputs: list,
                status: str = "OK", scope: str = "consolidated", extra: Optional[dict] = None,
                suffix: str = "") -> MetricRecord:
        input_ids = [i.fact_id if hasattr(i, "fact_id") else (i.metric_id if hasattr(i, "metric_id") else str(i))
                     for i in inputs]
        avail = [i.available_at for i in inputs if hasattr(i, "available_at")]
        mid = f"m_{ticker[:6]}_{metric}_{period}{'_par' if scope == 'parent' else ''}{suffix}"
        rec = MetricRecord(metric_id=mid, ticker=ticker, metric=metric,
                           value=None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value),
                           unit=unit, status=status, period=period, statement_scope=StatementScope(scope),
                           formula=catalog.formula(metric), formula_version=catalog.formula_version(metric),
                           input_ids=input_ids, available_at=max(avail) if avail else self.svc.as_of,
                           as_of=self.svc.as_of, extra=extra or {})
        self.store[mid] = rec
        if self.lineage:
            for i, iid in zip(inputs, input_ids):
                ptype = "metric" if hasattr(i, "metric_id") else ("price" if iid.startswith("px_") else "fact")
                self.lineage.add(iid, mid, ptype, "metric", f"calc.{metric}", rec.formula_version)
                if ptype == "fact" and getattr(i, "source_document_id", None):
                    self.lineage.add(i.source_document_id, iid, "document", "fact", "ingestion.golden", "v1")
        return rec

    def raw(self, ticker: str, metric: str, pe: date, scope: str = "consolidated"):
        f = self.svc.fact(ticker, metric, pe, scope)
        if f is None:
            return None
        cn = catalog.cn_name(metric)
        rec = self._record(ticker, metric, period_label(pe), f.value, "CNY", [f], scope=scope,
                           extra={"cn": cn, "kind": "fact", "revision_id": f.revision_id, "source": f.source})
        return rec

    # ---------- period math ----------
    def single_quarter(self, ticker: str, metric: str, pe: date) -> MetricRecord:
        cur = self.svc.fact(ticker, metric, pe)
        prev_pe = prev_quarter_end(pe)
        label = period_label(pe)
        if cur is None:
            return self._record(ticker, "single_quarter", label, None, "CNY", [], "INSUFFICIENT_DATA", suffix=f"_{metric}")
        if prev_pe is None:
            return self._record(ticker, "single_quarter", label, cur.value, "CNY", [cur], suffix=f"_{metric}",
                                extra={"base_metric": metric})
        prev = self.svc.fact(ticker, metric, prev_pe)
        if prev is None:
            return self._record(ticker, "single_quarter", label, None, "CNY", [cur], "INSUFFICIENT_DATA", suffix=f"_{metric}")
        if prev.published_at > cur.published_at:
            return self._record(ticker, "single_quarter", label, None, "CNY", [cur, prev], "RESTATEMENT_BREAKS_SPLIT",
                                suffix=f"_{metric}", extra={"base_metric": metric,
                                                            "reason": f"{prev.fact_id} restated after {cur.fact_id}"})
        return self._record(ticker, "single_quarter", label, cur.value - prev.value, "CNY", [cur, prev],
                            suffix=f"_{metric}", extra={"base_metric": metric})

    def ttm(self, ticker: str, metric: str, pe: date, at: Optional[datetime] = None) -> tuple[Optional[float], list]:
        q = CUM_MONTH[pe.month]
        cur = self.svc.fact(ticker, metric, pe, at=at)
        if cur is None:
            return None, []
        if q == 4:
            return cur.value, [cur]
        annual_prev = self.svc.fact(ticker, metric, date(pe.year - 1, 12, 31), at=at)
        cum_prev = self.svc.fact(ticker, metric, year_ago(pe), at=at)
        if not annual_prev or not cum_prev:
            return None, [cur]
        return annual_prev.value - cum_prev.value + cur.value, [cur, annual_prev, cum_prev]

    # ---------- ratios ----------
    def ratios(self, ticker: str, pe: date) -> dict[str, MetricRecord]:
        g = lambda m: self.svc.fact(ticker, m, pe)  # noqa: E731
        lbl = period_label(pe)
        out = {}

        def ratio(name, num, den, status_if_bad="ZERO_DENOMINATOR_NA", den_positive=False):
            if num is None or den is None:
                out[name] = self._record(ticker, name, lbl, None, "ratio", [x for x in (num, den) if x], "INSUFFICIENT_DATA")
            elif den.value == 0 or (den_positive and den.value < 0):
                out[name] = self._record(ticker, name, lbl, None, "ratio", [num, den], status_if_bad)
            else:
                out[name] = self._record(ticker, name, lbl, num.value / den.value, "ratio", [num, den])

        rev, cost = g("operating_revenue"), g("operating_cost")
        if rev and cost:
            out["gross_margin"] = self._record(ticker, "gross_margin", lbl, (rev.value - cost.value) / rev.value,
                                               "ratio", [rev, cost])
        ratio("net_margin", g("net_profit"), rev)
        ratio("rd_ratio", g("rd_expense"), rev)
        ratio("ocf_to_net_profit", g("operating_cash_flow"), g("net_profit"), "NEGATIVE_EARNINGS_NA", den_positive=True)
        ratio("debt_to_assets", g("total_liabilities"), g("total_assets"))
        ratio("interest_debt_ratio", g("interest_bearing_debt"), g("total_assets"))
        # ROE: parent net profit over parent equity when the source provides it (real filings), else total equity
        eq_metric = "parent_equity" if g("parent_equity") else "total_equity"
        np_, eq = g("net_profit"), g(eq_metric)
        eq_begin = self.svc.fact(ticker, eq_metric, date(pe.year - 1, 12, 31))
        if np_ and eq and eq_begin:
            avg_eq = (eq.value + eq_begin.value) / 2
            nq = CUM_MONTH[pe.month]
            if avg_eq <= 0:
                out["roe_annualized"] = self._record(ticker, "roe_annualized", lbl, None, "ratio", [np_, eq, eq_begin], "NEGATIVE_EQUITY_NA")
            else:
                out["roe_annualized"] = self._record(ticker, "roe_annualized", lbl, np_.value * 4 / nq / avg_eq, "ratio", [np_, eq, eq_begin])
        rev_ttm, rin = self.ttm(ticker, "operating_revenue", pe)
        cost_ttm, cin = self.ttm(ticker, "operating_cost", pe)
        ar, inv = g("accounts_receivable"), g("inventory")
        if ar and rev_ttm:
            out["receivable_days"] = self._record(ticker, "receivable_days", lbl, ar.value / rev_ttm * 365, "days", [ar, *rin])
        if inv and cost_ttm:
            out["inventory_days"] = self._record(ticker, "inventory_days", lbl, inv.value / cost_ttm * 365, "days", [inv, *cin])
        return out

    def growth(self, ticker: str, pe: date) -> dict[str, MetricRecord]:
        lbl = period_label(pe)
        out = {}
        for name, m in (("revenue_yoy", "operating_revenue"), ("net_profit_yoy", "net_profit"),
                        ("ocf_yoy", "operating_cash_flow"), ("receivable_yoy", "accounts_receivable")):
            cur, base = self.svc.fact(ticker, m, pe), self.svc.fact(ticker, m, year_ago(pe))
            if not cur or not base:
                out[name] = self._record(ticker, name, lbl, None, "ratio", [x for x in (cur, base) if x], "INSUFFICIENT_DATA")
            elif base.value <= 0:
                out[name] = self._record(ticker, name, lbl, None, "ratio", [cur, base], "NEGATIVE_EARNINGS_NA")
            else:
                out[name] = self._record(ticker, name, lbl, cur.value / base.value - 1, "ratio", [cur, base])
        # QoQ on single quarters (Q1 is compared with the previous Q4 = annual - Q3 cumulative)
        prev_pe = prev_quarter_end(pe) or date(pe.year - 1, 12, 31)
        for name, m in (("revenue_qoq", "operating_revenue"), ("net_profit_qoq", "net_profit")):
            cur, prev = self.single_quarter(ticker, m, pe), self.single_quarter(ticker, m, prev_pe)
            if cur.value is None or prev.value is None:
                out[name] = self._record(ticker, name, lbl, None, "ratio", [cur, prev],
                                         cur.status if cur.status != "OK" else prev.status)
            elif prev.value <= 0:
                out[name] = self._record(ticker, name, lbl, None, "ratio", [cur, prev], "NEGATIVE_EARNINGS_NA")
            else:
                out[name] = self._record(ticker, name, lbl, cur.value / prev.value - 1, "ratio", [cur, prev])
        return out

    def ttm_latest(self, ticker: str, metric: str, at: Optional[datetime] = None) -> tuple[Optional[float], list]:
        """TTM of a flow reported only in some periods (D&A: H1 and annual) from its latest visible period."""
        periods = [p for p in reversed(self.svc.visible_periods(ticker, at))
                   if self.svc.fact(ticker, metric, p, at=at) is not None]
        for p in periods[:2]:
            v, ins = self.ttm(ticker, metric, p, at=at)
            if v is not None:
                return v, ins
        return None, []

    # ---------- valuation ----------
    def _pe_at(self, ticker: str, d: date) -> tuple[Optional[float], str, list]:
        at = datetime(d.year, d.month, d.day, 15, 0)
        px = self.svc.price_on_or_before(ticker, d)
        pe = self.svc.latest_period(ticker, at=at)
        if not px or not pe:
            return None, "INSUFFICIENT_DATA", []
        np_ttm, ins = self.ttm(ticker, "net_profit", pe, at=at)
        if np_ttm is None:
            return None, "INSUFFICIENT_DATA", ins
        if np_ttm <= 0:
            return None, "NEGATIVE_EARNINGS_NA", ins
        return px.total_mv / np_ttm, "OK", ins

    def valuation(self, ticker: str) -> dict[str, MetricRecord]:
        d = self.svc.as_of.date()
        px = self.svc.price_on_or_before(ticker, d)
        pe_end = self.svc.latest_period(ticker)
        if not px or not pe_end:
            return {}
        lbl = d.isoformat()
        price_ref = type("P", (), {"fact_id": f"px_{ticker[:6]}_{px.trade_date.isoformat()}", "available_at": px.available_at})()
        out = {}
        val, status, ins = self._pe_at(ticker, d)
        out["pe_ttm"] = self._record(ticker, "pe_ttm", lbl, val, "x", [price_ref, *ins], status,
                                     extra={"total_mv": px.total_mv, "price_date": px.trade_date.isoformat()})
        eq = self.svc.fact(ticker, "total_equity", pe_end)
        if eq:
            out["pb_lf"] = self._record(ticker, "pb_lf", lbl, px.total_mv / eq.value if eq.value > 0 else None, "x",
                                        [price_ref, eq], "OK" if eq.value > 0 else "NEGATIVE_EQUITY_NA")
        rev_ttm, rin = self.ttm(ticker, "operating_revenue", pe_end)
        if rev_ttm:
            out["ps_ttm"] = self._record(ticker, "ps_ttm", lbl, px.total_mv / rev_ttm, "x", [price_ref, *rin])
        # EV/EBITDA: EBITDA = total profit + interest expense + D&A (all TTM)
        debt, cash = self.svc.fact(ticker, "interest_bearing_debt", pe_end), self.svc.fact(ticker, "cash", pe_end)
        tp, tin = self.ttm(ticker, "total_profit", pe_end)
        ie, iin = self.ttm(ticker, "interest_expense", pe_end)
        da, din = self.ttm_latest(ticker, "depreciation_amortization")
        if None not in (debt, cash, tp, da):
            ebitda = tp + (ie or 0) + da
            ev = px.total_mv + debt.value - cash.value
            out["ev_ebitda"] = self._record(ticker, "ev_ebitda", lbl, ev / ebitda if ebitda > 0 else None, "x",
                                            [price_ref, debt, cash, *tin, *iin, *din],
                                            "OK" if ebitda > 0 else "NEGATIVE_EARNINGS_NA",
                                            extra={"ev": ev, "ebitda_ttm": ebitda})
        # PE percentiles over month-end PIT-correct history
        hist = []
        m = d.replace(day=1)
        for _ in range(60):
            m = (m - timedelta(days=1)).replace(day=1)
            month_end = (m + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            v, st, _ = self._pe_at(ticker, month_end)
            hist.append(v if st == "OK" else None)
        for years, need in ((3, 12), (5, 48)):
            h = [x for x in hist[:years * 12] if x is not None]
            name = f"pe_percentile_{years}y"
            if val is not None and len(h) >= need:
                out[name] = self._record(ticker, name, lbl, sum(1 for x in h if x <= val) / len(h), "ratio",
                                         [out["pe_ttm"]], extra={"n_months": len(h), "min": min(h), "max": max(h),
                                                                 "median": median(h)})
            else:
                out[name] = self._record(ticker, name, lbl, None, "ratio", [out["pe_ttm"]],
                                         status if status != "OK" else "INSUFFICIENT_DATA",
                                         extra={"n_months": len(h), "needed": need})
        return out

    def dcf(self, ticker: str, fcf0: float, growth: float, wacc: float, terminal_growth: float, years: int = 5,
            wacc_grid: tuple = (-0.01, -0.005, 0.0, 0.005, 0.01),
            g_grid: tuple = (-0.01, -0.005, 0.0, 0.005, 0.01)) -> MetricRecord:
        """Optional DCF. All assumptions are explicit inputs (never chosen by an LLM) and a 2-D
        WACC x terminal-growth sensitivity matrix is always returned (PRD 4.3)."""
        shares_f = self.svc.fact(ticker, "shares_outstanding", self.svc.latest_period(ticker))
        if shares_f is None or not shares_f.value:
            return self._record(ticker, "dcf_value_per_share", self.svc.as_of.date().isoformat(), None, "CNY", [],
                                "INSUFFICIENT_DATA")

        def value(w: float, gi: float) -> Optional[float]:
            if w <= gi:
                return None
            pv, f = 0.0, fcf0
            for t in range(1, years + 1):
                f *= 1 + growth
                pv += f / (1 + w) ** t
            tv = f * (1 + gi) / (w - gi)
            return (pv + tv / (1 + w) ** years) / shares_f.value

        base = value(wacc, terminal_growth)
        matrix = [[value(wacc + dw, terminal_growth + dg) for dg in g_grid] for dw in wacc_grid]
        return self._record(ticker, "dcf_value_per_share", self.svc.as_of.date().isoformat(), base, "CNY", [shares_f],
                            "OK" if base is not None else "ZERO_DENOMINATOR_NA",
                            extra={"assumptions": {"fcf0": fcf0, "growth": growth, "wacc": wacc,
                                                   "terminal_growth": terminal_growth, "years": years},
                                   "sensitivity": {"wacc": [wacc + x for x in wacc_grid],
                                                   "terminal_growth": [terminal_growth + x for x in g_grid],
                                                   "value_per_share": matrix}})

    def peer_valuation(self, ticker: str) -> MetricRecord:
        d = self.svc.as_of.date()
        peers = self.svc.industry_peers(ticker)
        vals, used = [], []
        for p in peers:
            v, st, _ = self._pe_at(p.ticker, d)
            if st == "OK":
                vals.append(v)
                used.append(p.ticker)
        return self._record(ticker, "peer_pe_median", d.isoformat(), median(vals) if vals else None, "x", [],
                            "OK" if vals else "INSUFFICIENT_DATA", extra={"peers": used,
                                                                           "peer_values": dict(zip(used, vals))})

    # ---------- risk ----------
    def risk(self, ticker: str) -> dict[str, MetricRecord]:
        px = [p for p in self.svc.prices(ticker) if p.trade_date <= self.svc.as_of.date()][-250:]
        lbl = self.svc.as_of.date().isoformat()
        if len(px) < 60:
            return {}
        closes = np.array([p.close for p in px])
        rets = np.diff(np.log(closes))
        vol = float(rets.std(ddof=1) * math.sqrt(252))
        mdd = float((closes / np.maximum.accumulate(closes) - 1).min())
        ref = type("P", (), {"fact_id": f"px_{ticker[:6]}_{px[0].trade_date}_{px[-1].trade_date}",
                             "available_at": px[-1].available_at})()
        return {"volatility_1y": self._record(ticker, "volatility_1y", lbl, vol, "ratio", [ref]),
                "max_drawdown_1y": self._record(ticker, "max_drawdown_1y", lbl, mdd, "ratio", [ref])}

    # ---------- anomaly detection (7 rules) ----------
    def detect_anomalies(self, ticker: str, pe: date, announcements: list[tuple[str, str]] = (),
                         mode: str = "statistical") -> list[AnomalyFlag]:
        """mode="statistical": year-over-year signals scored against the company's own history and its peers
        (anomaly_stats.py). mode="rules": the original fixed cut-offs, kept so the two can be compared."""
        if mode == "rules":
            return self.detect_rule_anomalies(ticker, pe, announcements)
        from . import anomaly_stats

        flags = anomaly_stats.detect(self, ticker, pe)
        for doc_id, text in announcements:      # rule 7 is document based, not numeric
            if not ROUTINE_RPT_RE.search(text) and any(k in text for k in ("关联交易", "会计估计变更", "会计差错")):
                flags.append(AnomalyFlag(code="RELATED_PARTY_OR_POLICY_CHANGE", severity="MEDIUM", metric_ids=[],
                                         message=f"公告《{text}》涉及关联交易或会计变更", section_type="ANNOUNCEMENT",
                                         suggested_query="关联交易 会计差错更正 会计估计变更", category="rule"))
                break
        return flags

    def detect_rule_anomalies(self, ticker: str, pe: date, announcements: list[tuple[str, str]] = ()) -> list[AnomalyFlag]:
        r = {**self.ratios(ticker, pe), **self.growth(ticker, pe)}
        r_prev = self.ratios(ticker, year_ago(pe))
        flags: list[AnomalyFlag] = []
        v = lambda d, k: d[k].value if k in d and d[k].value is not None else None  # noqa: E731
        npy, ocfy = v(r, "net_profit_yoy"), v(r, "ocf_yoy")
        if npy is not None and ocfy is not None and npy > 0.2 and ocfy < -0.3:
            flags.append(AnomalyFlag(code="PROFIT_OCF_DIVERGENCE", severity="HIGH",
                                     metric_ids=[r["net_profit_yoy"].metric_id, r["ocf_yoy"].metric_id],
                                     message=f"净利润同比{npy:+.1%}而经营现金流同比{ocfy:+.1%}",
                                     suggested_query="经营活动现金流量净额 下降 主要系 回款 原因", section_type="MDA"))
        ary, revy = v(r, "receivable_yoy"), v(r, "revenue_yoy")
        rdays = v(r, "receivable_days")
        # materiality floor: receivables under 1% of TTM revenue (< 3.65 days, e.g. Moutai's 0.1 day)
        # double on a tiny base without saying anything about collection quality
        if ary is not None and revy is not None and ary - revy > 0.3 and (rdays is None or rdays >= 3.65):
            flags.append(AnomalyFlag(code="AR_OUTPACE_REVENUE", severity="HIGH" if ary - revy > 0.5 else "MEDIUM",
                                     metric_ids=[r["receivable_yoy"].metric_id, r["revenue_yoy"].metric_id],
                                     message=f"应收账款同比{ary:+.1%}显著高于营收同比{revy:+.1%}",
                                     suggested_query="应收账款 按欠款方归集 前五名 坏账准备"))
        idays, idays0 = v(r, "inventory_days"), v(r_prev, "inventory_days")
        if idays and idays0 and idays / idays0 - 1 > 0.3:
            flags.append(AnomalyFlag(code="INVENTORY_DAYS_DETERIORATION", severity="MEDIUM",
                                     metric_ids=[r["inventory_days"].metric_id, r_prev["inventory_days"].metric_id],
                                     message=f"存货周转天数由{idays0:.0f}天升至{idays:.0f}天",
                                     suggested_query="存货跌价准备 计提 可变现净值"))
        gm_series = []
        d = pe
        for _ in range(8):
            rev_q = self.single_quarter(ticker, "operating_revenue", d)
            cost_q = self.single_quarter(ticker, "operating_cost", d)
            if rev_q.value and cost_q.value is not None:
                gm_series.append(((rev_q.value - cost_q.value) / rev_q.value, rev_q, cost_q))
            pq = prev_quarter_end(d)
            d = pq if pq else date(d.year - 1, 12, 31)
        if len(gm_series) >= 2:
            # compare the latest *two* quarters: catch a shift in either
            for idx in (0, 1):
                if idx + 1 >= len(gm_series):
                    break
                cur_gm, prev_gm = gm_series[idx][0], gm_series[idx + 1][0]
                hist = [x[0] for x in gm_series[idx + 1:]]
                sd = pstdev(hist) if len(hist) > 2 else 0.0
                delta = cur_gm - prev_gm
                if abs(delta) > 0.04 or (sd > 0 and abs(cur_gm - sum(hist) / len(hist)) > 2 * sd and abs(delta) > 0.02):
                    gm_rec = self._record(ticker, "gross_margin_sq", gm_series[idx][1].period, cur_gm, "ratio",
                                          [gm_series[idx][1], gm_series[idx][2]])
                    prev_rec = self._record(ticker, "gross_margin_sq", gm_series[idx + 1][1].period, prev_gm, "ratio",
                                            [gm_series[idx + 1][1], gm_series[idx + 1][2]])
                    d_rec = self._record(ticker, "gross_margin_sq_change", gm_series[idx][1].period, delta, "ratio",
                                         [gm_rec, prev_rec])
                    flags.append(AnomalyFlag(code="GROSS_MARGIN_SHIFT", severity="HIGH",
                                             metric_ids=[gm_rec.metric_id, prev_rec.metric_id, d_rec.metric_id],
                                             message=f"{gm_rec.period}单季毛利率{cur_gm:.1%}，较上季{prev_gm:.1%}变动{delta:+.1%}",
                                             suggested_query="毛利率 下降 原因 售价 成本", section_type="MDA"))
                    break
        imp, np_, ta = (self.svc.fact(ticker, m, pe) for m in ("asset_impairment_loss", "net_profit", "total_assets"))
        if imp and np_ and ta and (imp.value / max(abs(np_.value), 1) > 0.3 or imp.value / ta.value > 0.02):
            rec = self._record(ticker, "impairment_to_profit", period_label(pe), imp.value / max(abs(np_.value), 1),
                               "ratio", [imp, np_])
            flags.append(AnomalyFlag(code="LARGE_IMPAIRMENT", severity="HIGH", metric_ids=[rec.metric_id],
                                     message=f"资产减值损失占净利润比例{rec.value:.1%}",
                                     suggested_query="存货跌价准备 资产减值 计提"))
        idr, idr0 = v(r, "interest_debt_ratio"), v(r_prev, "interest_debt_ratio")
        if idr is not None and idr0 is not None and idr - idr0 > 0.08:
            flags.append(AnomalyFlag(code="INTEREST_DEBT_JUMP", severity="HIGH",
                                     metric_ids=[r["interest_debt_ratio"].metric_id, r_prev["interest_debt_ratio"].metric_id],
                                     message=f"有息负债率由{idr0:.1%}升至{idr:.1%}",
                                     suggested_query="银行借款 债券 有息负债 融资", section_type="ANNOUNCEMENT"))
        for doc_id, text in announcements:
            # PRD 2.2 rule 7 is about *major* related-party deals or accounting changes. Routine annual
            # related-party budgets, finance-company deposit statements and internal policy documents
            # are filed by almost every large A-share company and are not a signal.
            if ROUTINE_RPT_RE.search(text):
                continue
            if any(k in text for k in ("关联交易", "会计估计变更", "会计差错")):
                flags.append(AnomalyFlag(code="RELATED_PARTY_OR_POLICY_CHANGE", severity="MEDIUM", metric_ids=[],
                                         message=f"公告《{text}》涉及关联交易或会计变更", section_type="ANNOUNCEMENT",
                                         suggested_query="关联交易 会计差错更正 会计估计变更"))
                break
        return flags
