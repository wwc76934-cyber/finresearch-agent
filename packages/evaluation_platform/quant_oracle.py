"""Independent ground truth for scenario D.

Re-derives every factor statistic straight from the panel CSVs with pandas / numpy. It shares no code with
data_platform.factors (different data structures, ranks via pandas, correlations via numpy), so a bug in the
tool shows up as an evaluation failure instead of being copied into the answer key. Only the *definitions*
(documented in factors.py) are common.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from data_platform.db import ROOT

DIR = ROOT / "benchmarks" / "factor_panel"


@lru_cache(maxsize=1)
def _load():
    f = pd.read_csv(DIR / "financial_facts.csv", parse_dates=["available_at", "period_end"])
    p = pd.read_csv(DIR / "market_prices.csv", parse_dates=["trade_date"])
    s = pd.read_csv(DIR / "securities.csv")
    return f.sort_values(["available_at", "revision_id"]), p, s


def _visible(t: pd.Timestamp) -> pd.DataFrame:
    """(ticker, period_end) x metric table of the latest revision public at t."""
    f, _, _ = _load()
    v = f[f.available_at <= t]
    last = v.groupby(["ticker", "metric", "period_end"], sort=False).value.last()
    return last.unstack("metric")


def _factor_table(factor: str, t: pd.Timestamp) -> pd.Series:
    w = _visible(t)
    out = {}
    for tk, g in w.groupby(level="ticker"):
        g = g.droplevel("ticker")
        ok = g[["operating_revenue", "net_profit"]].dropna().index if {"operating_revenue", "net_profit"} <= set(g.columns) else []
        if len(ok) == 0:
            continue
        pe = ok.max()
        py = pe - pd.DateOffset(years=1)
        fy = pd.Timestamp(year=pe.year - 1, month=12, day=31)

        def cum(m, d):
            return g[m].get(d, np.nan) if m in g.columns else np.nan

        def ttm(m):
            if pe.month == 12:
                return cum(m, pe)
            return cum(m, fy) + cum(m, pe) - cum(m, py)

        v = np.nan
        if factor == "ocf_to_net_profit":
            n, o = ttm("net_profit"), ttm("operating_cash_flow")
            v = o / n if n > 0 else np.nan
        elif factor == "roe_ttm":
            n, e1, e0 = ttm("net_profit"), cum("parent_equity", pe), cum("parent_equity", py)
            v = n / ((e1 + e0) / 2) if (e1 + e0) > 0 else np.nan
        elif factor == "gross_margin_yoy_delta":
            r1, c1, r0, c0 = (cum("operating_revenue", pe), cum("operating_cost", pe),
                              cum("operating_revenue", py), cum("operating_cost", py))
            v = (1 - c1 / r1) - (1 - c0 / r0) if r1 > 0 and r0 > 0 else np.nan
        elif factor == "accruals_to_assets":
            n, o, a1, a0 = ttm("net_profit"), ttm("operating_cash_flow"), cum("total_assets", pe), cum("total_assets", py)
            v = (n - o) / ((a1 + a0) / 2) if (a1 + a0) > 0 else np.nan
        elif factor == "receivable_turnover_yoy":
            r1, r0, a1, a0 = (cum("operating_revenue", pe), cum("operating_revenue", py),
                              cum("accounts_receivable", pe), cum("accounts_receivable", py))
            v = (r1 / a1) / (r0 / a0) - 1 if a1 > 0 and a0 > 0 and r0 > 0 else np.nan
        elif factor == "pe_ttm_inverse":
            _, px, _ = _load()
            day = px[(px.ticker == tk) & (px.trade_date <= t.normalize())].sort_values("trade_date").iloc[-1]
            n, sh = ttm("net_profit"), cum("shares_outstanding", pe)
            v = n / (day.close * sh) if sh > 0 else np.nan
        out[tk] = v
    return pd.Series(out, dtype=float).replace([np.inf, -np.inf], np.nan)


def factor_stats(factor: str, universe: str = "all", as_of_date: str = "2025-12-31", lookback: int = 8,
                 horizon: int = 60, quantiles: int = 3, cs_dates: Optional[list[str]] = None) -> dict:
    f, px, sec = _load()
    tickers = list(sec.ticker if universe == "all" else sec[sec.universe == universe].ticker)
    adj = px.pivot(index="trade_date", columns="ticker", values="adj_close").sort_index()
    cal = list(adj.index)
    as_of = pd.Timestamp(as_of_date)
    last_i = max(i for i, d in enumerate(cal) if d <= as_of)

    def on_or_before(d: pd.Timestamp) -> Optional[int]:
        ids = [i for i, x in enumerate(cal) if x <= d]
        return ids[-1] if ids else None

    if cs_dates:
        idxs = sorted({on_or_before(pd.Timestamp(d)) for d in cs_dates})
    else:
        idxs = []
        for y in range(2022, as_of.year + 1):
            for mm, dd in ((3, 31), (6, 30), (9, 30), (12, 31)):
                q = pd.Timestamp(year=y, month=mm, day=dd)
                i = on_or_before(q)
                if i is None or cal[i] < pd.Timestamp("2022-12-01") or cal[i] > as_of:
                    continue
                if i + horizon <= last_i:
                    idxs.append(i)
        idxs = sorted(set(idxs))[-lookback:]
    ics, rics, grp, uni, per = [], [], [], [], []
    for i in idxs:
        t = cal[i] + pd.Timedelta(hours=15)
        fac = _factor_table(factor, t).reindex(tickers)
        a0 = adj.iloc[i][tickers]
        a1 = adj.iloc[i + horizon][tickers] if i + horizon <= last_i else pd.Series(np.nan, index=tickers)
        df = pd.DataFrame({"f": fac, "r": a1 / a0 - 1}).dropna()
        if len(df) < 6:
            continue
        df = df.reset_index().rename(columns={"index": "tk"}).sort_values(["f", "tk"])
        ics.append(np.corrcoef(df.f, df.r)[0, 1])
        rics.append(np.corrcoef(df.f.rank(), df.r.rank())[0, 1])
        parts = np.array_split(df.r.to_numpy(), quantiles)
        grp.append([p.mean() for p in parts])
        uni.append(df.r.mean())
        per.append(len(df))
    out = {"n_cs": len(ics), "n_stocks": int(np.median(per)) if per else 0}
    if len(ics) < 2:
        return out
    ics, rics, G, U = np.array(ics), np.array(rics), np.array(grp), np.array(uni)
    n = len(ics)

    def tp(x):
        tt = x.mean() / (x.std(ddof=1) / np.sqrt(n))
        return float(tt), float(2 * student_t.sf(abs(tt), n - 1))

    ls = G[:, -1] - G[:, 0]
    nav = np.concatenate([[1.0], np.cumprod(1 + ls)])
    out.update(ic_mean=float(ics.mean()), rank_ic_mean=float(rics.mean()),
               rank_ic_ir=float(rics.mean() / rics.std(ddof=1)), group_means=[float(x) for x in G.mean(axis=0)],
               ls_mean=float(ls.mean()), ls_cum=float(nav[-1] - 1),
               ls_mdd=float((nav / np.maximum.accumulate(nav) - 1).min()))
    out["rank_ic_t"], out["rank_ic_p"] = tp(rics)
    return out
