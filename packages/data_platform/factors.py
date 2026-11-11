"""Scenario D: cross-sectional factor validation on a point-in-time panel (PRD v1.2 Epic 4).

evaluate_factor_hypothesis() answers "does factor X predict the next 20/60 trading days of return?" with
IC / RankIC series, IC_IR, a t-test, tercile (or n-quantile) groups and a long-short curve.

Point-in-time rules, enforced in code:
* Cross-sections are the last trading day of each calendar quarter, at the 15:00 close. At cross-section t a
  company's factor uses only financial_facts rows with available_at <= t (latest revision visible at t).
  So at 2024-03-31 the newest report anyone can use is 2023Q3, not 2024Q1 (filed on 2024-04-25 or later).
* Forward returns are only used when the whole forward window ends on or before as_of_date. Returns that
  would need data after as_of are never read.
* Asking for a cross-section after as_of_date, or forcing a report period that is not yet visible at t
  (report_period=...), raises PITViolationError.

The panel is benchmarks/factor_panel (24 companies; see its manifest.json for the survivorship note).
"""
from __future__ import annotations

import json
import math
from bisect import bisect_right
from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from .db import ROOT

PANEL_DIR = ROOT / "benchmarks" / "factor_panel"
FACTORS = ("ocf_to_net_profit", "roe_ttm", "gross_margin_yoy_delta", "accruals_to_assets",
           "receivable_turnover_yoy", "pe_ttm_inverse")
FACTOR_CN = {"ocf_to_net_profit": "盈利现金保障倍数(OCF/净利润,TTM)", "roe_ttm": "净资产收益率(TTM)",
             "gross_margin_yoy_delta": "毛利率同比变动", "accruals_to_assets": "应计项目占总资产比(TTM)",
             "receivable_turnover_yoy": "应收账款周转率同比变动", "pe_ttm_inverse": "盈利收益率(市盈率倒数)"}
# accruals: a lower value means higher earnings quality
HIGHER_IS_BETTER = {f: f != "accruals_to_assets" for f in FACTORS}
MIN_STOCKS = 6            # a cross-section with fewer valid companies is skipped
MONTHS = {3: 1, 6: 2, 9: 3, 12: 4}


class PITViolationError(RuntimeError):
    """A factor request would read data that was not public at the cross-section date."""


class PanelCoverageError(ValueError):
    pass


class CrossSection(BaseModel):
    date: str
    n: int
    ic: float
    rank_ic: float
    report_period: str          # the newest report period used by the median company
    group_returns: list[float]  # mean forward return per group, lowest factor first
    universe_return: float


class FactorValidationReport(BaseModel):
    factor: str
    universe: str
    as_of_date: str
    lookback_quarters: int
    forward_horizon_days: int
    quantiles: int
    higher_is_better: bool
    n_cross_sections: int
    n_stocks_median: int
    ic_mean: Optional[float] = None
    ic_std: Optional[float] = None
    ic_ir: Optional[float] = None
    ic_t_stat: Optional[float] = None
    ic_p_value: Optional[float] = None
    rank_ic_mean: Optional[float] = None
    rank_ic_std: Optional[float] = None
    rank_ic_ir: Optional[float] = None
    rank_ic_t_stat: Optional[float] = None
    rank_ic_p_value: Optional[float] = None
    rank_ic_positive_share: Optional[float] = None
    group_mean_returns: list[float] = Field(default_factory=list)   # lowest factor group first
    long_short_mean_return: Optional[float] = None       # top group minus bottom group, per cross-section
    long_short_cum_return: Optional[float] = None
    long_short_max_drawdown: Optional[float] = None
    top_cum_excess_vs_universe: Optional[float] = None
    significant_5pct: bool = False
    cross_sections: list[CrossSection] = Field(default_factory=list)
    nav: list[dict] = Field(default_factory=list)        # long-short and per-group cumulative net values
    warnings: list[dict] = Field(default_factory=list)   # {"code", "message"}
    summary: str = ""


# ------------------------------------------------------------------ panel
class Panel:
    def __init__(self, directory: Path = PANEL_DIR):
        f = pd.read_csv(directory / "financial_facts.csv", parse_dates=["available_at"])
        self.facts: dict[tuple, list[tuple]] = {}
        for r in f.sort_values("available_at").itertuples(index=False):
            self.facts.setdefault((r.ticker, r.metric, r.period_end), []).append((r.available_at.to_pydatetime(), r.value))
        self.avail_keys = {k: [a for a, _ in v] for k, v in self.facts.items()}
        px = pd.read_csv(directory / "market_prices.csv")
        self.dates = sorted(px.trade_date.unique())
        self.date_pos = {d: i for i, d in enumerate(self.dates)}
        piv = px.pivot(index="trade_date", columns="ticker", values="adj_close").reindex(self.dates)
        self.adj = piv
        self.close = px.pivot(index="trade_date", columns="ticker", values="close").reindex(self.dates)
        self.sec = pd.read_csv(directory / "securities.csv")
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.periods: dict[str, list[str]] = {}
        for (t, m, pe) in self.facts:
            if m == "operating_revenue":
                self.periods.setdefault(t, []).append(pe)
        for t in self.periods:
            self.periods[t].sort()

    def tickers(self, universe: str) -> list[str]:
        if universe == "all":
            return list(self.sec.ticker)
        return list(self.sec[self.sec.universe == universe].ticker)

    def value(self, ticker: str, metric: str, pe: str, t: datetime) -> Optional[float]:
        """Latest revision of (ticker, metric, period) that was public at t."""
        key = (ticker, metric, pe)
        if key not in self.facts:
            return None
        i = bisect_right(self.avail_keys[key], t)
        return self.facts[key][i - 1][1] if i else None

    def available_at(self, ticker: str, metric: str, pe: str) -> Optional[datetime]:
        v = self.facts.get((ticker, metric, pe))
        return v[0][0] if v else None

    def latest_period(self, ticker: str, t: datetime) -> Optional[str]:
        for pe in reversed(self.periods.get(ticker, [])):
            if self.value(ticker, "operating_revenue", pe, t) is not None and \
                    self.value(ticker, "net_profit", pe, t) is not None:
                return pe
        return None

    def last_trading_day_on_or_before(self, d: str) -> Optional[str]:
        i = bisect_right(self.dates, d)
        return self.dates[i - 1] if i else None


@lru_cache(maxsize=1)
def get_panel() -> Panel:
    return Panel()


def _shift_year(pe: str, years: int) -> str:
    y, m, d = pe.split("-")
    return f"{int(y) + years}-{m}-{d}"


def _period_label(pe: str) -> str:
    return f"{pe[:4]}Q{(int(pe[5:7]) - 1) // 3 + 1}"


def _period_from_label(label: str) -> str:
    y, q = label.upper().split("Q")
    return {1: f"{y}-03-31", 2: f"{y}-06-30", 3: f"{y}-09-30", 4: f"{y}-12-31"}[int(q)]


# ------------------------------------------------------------------ factor values
def factor_value(panel: Panel, factor: str, ticker: str, t: datetime,
                 report_period: Optional[str] = None) -> tuple[Optional[float], Optional[str]]:
    """(factor value, report period used) for one company at time t, or (None, None) when it cannot be computed."""
    if report_period:
        pe = _period_from_label(report_period)
        at = panel.available_at(ticker, "operating_revenue", pe)
        if at is None:
            return None, None       # no such report for this company: a data gap, not a time violation
        if at > t:
            raise PITViolationError(
                f"{ticker} {report_period}: report becomes available {at}, after the cross-section {t:%Y-%m-%d %H:%M}")
    else:
        pe = panel.latest_period(ticker, t)
    if pe is None:
        return None, None
    v = lambda m, p: panel.value(ticker, m, p, t)  # noqa: E731
    py, fy = _shift_year(pe, -1), _shift_year(pe[:4] + "-12-31", -1)

    def ttm(m: str) -> Optional[float]:
        cur = v(m, pe)
        if cur is None:
            return None
        if pe.endswith("12-31"):
            return cur
        base, prev = v(m, fy), v(m, py)
        return None if base is None or prev is None else base + cur - prev

    out = None
    if factor == "ocf_to_net_profit":
        np_, ocf = ttm("net_profit"), ttm("operating_cash_flow")
        out = ocf / np_ if np_ and np_ > 0 and ocf is not None else None
    elif factor == "roe_ttm":
        np_, e1, e0 = ttm("net_profit"), v("parent_equity", pe), v("parent_equity", py)
        out = np_ / ((e1 + e0) / 2) if None not in (np_, e1, e0) and (e1 + e0) > 0 else None
    elif factor == "gross_margin_yoy_delta":
        r1, c1, r0, c0 = v("operating_revenue", pe), v("operating_cost", pe), v("operating_revenue", py), v("operating_cost", py)
        out = (1 - c1 / r1) - (1 - c0 / r0) if None not in (r1, c1, r0, c0) and r1 > 0 and r0 > 0 else None
    elif factor == "accruals_to_assets":
        np_, ocf, a1, a0 = ttm("net_profit"), ttm("operating_cash_flow"), v("total_assets", pe), v("total_assets", py)
        out = (np_ - ocf) / ((a1 + a0) / 2) if None not in (np_, ocf, a1, a0) and (a1 + a0) > 0 else None
    elif factor == "receivable_turnover_yoy":
        m_ = MONTHS[int(pe[5:7])]
        r1, r0, ar1, ar0 = v("operating_revenue", pe), v("operating_revenue", py), v("accounts_receivable", pe), v("accounts_receivable", py)
        out = ((r1 / ar1) / (r0 / ar0) - 1) if None not in (r1, r0, ar1, ar0) and ar1 > 0 and ar0 > 0 and r0 > 0 else None
        _ = m_  # same-period comparison: the annualisation factor cancels
    elif factor == "pe_ttm_inverse":
        np_, sh = ttm("net_profit"), v("shares_outstanding", pe)
        d = panel.last_trading_day_on_or_before(t.date().isoformat())
        px = panel.close.at[d, ticker] if d is not None else None
        out = np_ / (px * sh) if None not in (np_, sh) and px and not math.isnan(px) and sh > 0 else None
    else:
        raise ValueError(f"unknown factor {factor}")
    if out is None or (isinstance(out, float) and (math.isnan(out) or math.isinf(out))):
        return None, None
    return float(out), pe


# ------------------------------------------------------------------ cross-sections
def cross_section_dates(panel: Panel, as_of: datetime, horizon: int) -> list[str]:
    """Quarter-end trading days (YYYY-MM-DD) whose forward window ends on or before as_of."""
    last = panel.last_trading_day_on_or_before(as_of.date().isoformat())
    if last is None:
        return []
    out = []
    for y in range(2022, as_of.year + 1):
        for md in ("03-31", "06-30", "09-30", "12-31"):
            d = panel.last_trading_day_on_or_before(f"{y}-{md}")
            if d is None or d < "2022-12-01" or d > as_of.date().isoformat():
                continue
            i = panel.date_pos[d]
            if i + horizon <= panel.date_pos[last]:
                out.append(d)
    return sorted(set(out))


def _max_drawdown(nav: list[float]) -> float:
    peak, mdd = nav[0], 0.0
    for x in nav:
        peak = max(peak, x)
        mdd = min(mdd, x / peak - 1)
    return mdd


def _groups(order: list[str], q: int) -> list[list[str]]:
    return [list(g) for g in np.array_split(np.array(order, dtype=object), q)]


def evaluate_factor_hypothesis(
    factor_name: Literal["ocf_to_net_profit", "roe_ttm", "gross_margin_yoy_delta", "accruals_to_assets",
                         "receivable_turnover_yoy", "pe_ttm_inverse"],
    universe: Literal["all", "new_energy", "consumer"] = "all",
    as_of_date: str = "2025-12-31",
    lookback_quarters: int = 8,
    forward_horizon_days: int = 60,
    quantiles: int = 3,
    report_period: Optional[str] = None,
    cross_section_dates_override: Optional[list[str]] = None,
) -> FactorValidationReport:
    """Cross-sectional IC / RankIC test of one factor. `report_period` (e.g. "2024Q1") and
    `cross_section_dates_override` exist to request things that must fail: using a report that was not yet
    public at the cross-section, or a cross-section after as_of_date. Both raise PITViolationError."""
    if factor_name not in FACTORS:
        raise ValueError(f"unknown factor {factor_name}; choose from {FACTORS}")
    panel = get_panel()
    as_of = datetime.combine(date.fromisoformat(as_of_date), time(15, 0))
    if as_of_date > panel.dates[-1]:
        raise PanelCoverageError(f"panel prices end {panel.dates[-1]}; as_of_date {as_of_date} is beyond it")
    for d in cross_section_dates_override or []:
        if d > as_of_date:
            raise PITViolationError(f"cross-section {d} is after as_of_date {as_of_date}")
    tickers = panel.tickers(universe)
    if not tickers:
        raise ValueError(f"unknown universe {universe}")
    horizon = int(forward_horizon_days)
    if cross_section_dates_override:
        # a calendar date that is not a trading day (e.g. Sunday 2024-03-31) means the last close before it
        dates = sorted({panel.last_trading_day_on_or_before(d) for d in cross_section_dates_override})
        if None in dates:
            raise PanelCoverageError(f"no price history before {cross_section_dates_override}")
    else:
        dates = cross_section_dates(panel, as_of, horizon)[-lookback_quarters:]

    cs: list[CrossSection] = []
    skipped: list[str] = []
    for d in dates:
        t = datetime.combine(date.fromisoformat(d), time(15, 0))
        i = panel.date_pos[d]
        rows = []
        for tk in tickers:
            val, pe = factor_value(panel, factor_name, tk, t, report_period)
            a0 = panel.adj.iat[i, panel.adj.columns.get_loc(tk)]
            j = i + horizon
            a1 = panel.adj.iat[j, panel.adj.columns.get_loc(tk)] if j < len(panel.dates) and panel.dates[j] <= as_of_date else np.nan
            if val is not None and not (np.isnan(a0) or np.isnan(a1)):
                rows.append((tk, val, a1 / a0 - 1, pe))
        if len(rows) < MIN_STOCKS:
            skipped.append(d)
            continue
        rows.sort(key=lambda r: (r[1], r[0]))            # ascending factor, ticker breaks ties
        f = np.array([r[1] for r in rows]); r_ = np.array([r[2] for r in rows])
        ic = float(stats.pearsonr(f, r_)[0]); ric = float(stats.spearmanr(f, r_)[0])
        grp = _groups([r[0] for r in rows], quantiles)
        by = {r[0]: r[2] for r in rows}
        pes = sorted(r[3] for r in rows)
        cs.append(CrossSection(date=d, n=len(rows), ic=ic, rank_ic=ric, report_period=_period_label(pes[len(pes) // 2]),
                               group_returns=[float(np.mean([by[x] for x in g])) for g in grp],
                               universe_return=float(np.mean(r_))))

    rep = FactorValidationReport(
        factor=factor_name, universe=universe, as_of_date=as_of_date, lookback_quarters=lookback_quarters,
        forward_horizon_days=horizon, quantiles=quantiles, higher_is_better=HIGHER_IS_BETTER[factor_name],
        n_cross_sections=len(cs), n_stocks_median=int(np.median([c.n for c in cs])) if cs else 0, cross_sections=cs)
    W = rep.warnings
    W.append({"code": "SURVIVORSHIP", "message": "样本池全是今天仍在上市的公司，退市或被剔除的公司不在其中，质量类因子的结果偏乐观。"})
    if skipped:
        W.append({"code": "SKIPPED_CROSS_SECTIONS", "message": f"{len(skipped)}个截面有效公司不足{MIN_STOCKS}家，已跳过：{', '.join(skipped)}"})
    if len(cs) < 2:
        W.append({"code": "FEW_CROSS_SECTIONS", "message": f"只有{len(cs)}个可用截面，无法做显著性检验。"})
        rep.summary = f"可用截面只有{len(cs)}个，不足以检验。"
        return rep
    if len(cs) < 8:
        W.append({"code": "FEW_CROSS_SECTIONS", "message": f"只有{len(cs)}个截面（建议至少8个），IC均值的标准误很大，结论不可靠。"})
    if rep.n_stocks_median / quantiles < 5:
        W.append({"code": "SMALL_GROUPS", "message": f"每个分组平均只有{rep.n_stocks_median / quantiles:.1f}家公司，分组收益噪声很大。"})
    if horizon > 60:
        W.append({"code": "OVERLAPPING_HORIZON", "message": "持有期超过一个季度，相邻截面的收益窗口重叠，t值被高估。"})
    if rep.n_stocks_median < 20:
        W.append({"code": "SMALL_UNIVERSE", "message": f"每个截面只有约{rep.n_stocks_median}家公司，样本量偏小。"})

    ic = np.array([c.ic for c in cs]); ric = np.array([c.rank_ic for c in cs])
    n = len(cs)
    rep.ic_mean, rep.ic_std = float(ic.mean()), float(ic.std(ddof=1))
    rep.rank_ic_mean, rep.rank_ic_std = float(ric.mean()), float(ric.std(ddof=1))
    rep.ic_ir = rep.ic_mean / rep.ic_std if rep.ic_std > 0 else None
    rep.rank_ic_ir = rep.rank_ic_mean / rep.rank_ic_std if rep.rank_ic_std > 0 else None
    tt = stats.ttest_1samp(ic, 0.0); tr = stats.ttest_1samp(ric, 0.0)
    rep.ic_t_stat, rep.ic_p_value = float(tt.statistic), float(tt.pvalue)
    rep.rank_ic_t_stat, rep.rank_ic_p_value = float(tr.statistic), float(tr.pvalue)
    rep.rank_ic_positive_share = float((ric > 0).mean())
    rep.significant_5pct = bool(rep.rank_ic_p_value < 0.05)

    G = np.array([c.group_returns for c in cs])            # n x quantiles
    U = np.array([c.universe_return for c in cs])
    rep.group_mean_returns = [float(x) for x in G.mean(axis=0)]
    ls = G[:, -1] - G[:, 0]
    nav_ls = np.concatenate([[1.0], np.cumprod(1 + ls)])
    rep.long_short_mean_return = float(ls.mean())
    rep.long_short_cum_return = float(nav_ls[-1] - 1)
    rep.long_short_max_drawdown = float(_max_drawdown(list(nav_ls)))
    rep.top_cum_excess_vs_universe = float(np.prod(1 + G[:, -1]) - np.prod(1 + U))
    navs = [np.concatenate([[1.0], np.cumprod(1 + G[:, g])]) for g in range(quantiles)]
    rep.nav = [{"date": "start", "long_short": 1.0, **{f"G{g + 1}": 1.0 for g in range(quantiles)}}] + [
        {"date": cs[k].date, "long_short": float(nav_ls[k + 1]), **{f"G{g + 1}": float(navs[g][k + 1]) for g in range(quantiles)}}
        for k in range(n)]
    direction = "越高越好" if rep.higher_is_better else "越低越好"
    verdict = "达到5%显著水平" if rep.significant_5pct else "未达到5%显著水平"
    rep.summary = (f"{FACTOR_CN[factor_name]}（{direction}）在{universe}样本池、{n}个季度截面、持有{horizon}个交易日：RankIC均值"
                   f"{rep.rank_ic_mean:+.3f}，t={rep.rank_ic_t_stat:.2f}，p={rep.rank_ic_p_value:.3f}，{verdict}；"
                   f"最高组减最低组每期{rep.long_short_mean_return:+.2%}，累计{rep.long_short_cum_return:+.2%}。")
    return rep
