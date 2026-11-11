"""Statistical-adaptive anomaly detection (PRD v1.2 section 1.2).

Instead of one fixed cut-off per anomaly (e.g. "debt ratio +8pp"), each anomaly is a *signal*: the year-over-year
change of an accounting ratio (same period one year earlier, so seasonality cancels). A signal is unusual when it
is far from

* the company's own history: Z_self = (x - mean(last 8 values)) / max(std, floor/2), and
* its peers in the same period: Z_peer = (x - median(peers)) / max(1.4826 * MAD, floor/2).

A flag needs the bad direction, an economic-materiality floor (a Z of 3 on a 0.01pp move is noise), and
|Z| above the trigger. Severity HIGH needs a larger Z and twice the floor. No company, ticker, period or question
text appears in this module.

The materiality floors are generic accounting magnitudes, listed in SIGNALS, and apply to every company alike.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from statistics import mean, median, stdev
from typing import Callable, Optional

Z_TRIGGER = 1.8
Z_HIGH = 2.5
HISTORY = 8
MIN_HISTORY = 4
MIN_PEERS = 4


# Level gates: a change is only worth testing when the underlying balance matters. Receivables below ~1% of annual
# revenue (under 3.65 days) double on a tiny base without saying anything about collection quality.
LEVEL_GATES = {"AR_OUTPACE_REVENUE": 3.65 / 365, "INVENTORY_DAYS_DETERIORATION": 3.65 / 365}


@dataclass(frozen=True)
class Signal:
    code: str
    label: str
    floor: float            # minimum |change| that matters economically
    direction: int          # +1 only increases are bad, -1 only decreases, 0 both
    unit: str               # how the change is shown: "pp" (difference of ratios) or "pct" (relative change)
    query: str
    section: str


SIGNALS = {
    "PROFIT_OCF_DIVERGENCE": Signal("PROFIT_OCF_DIVERGENCE", "应计项目占总资产比", 0.01, +1, "pp",
                                    "经营活动现金流量净额 下降 主要系 回款 原因", "MDA"),
    "AR_OUTPACE_REVENUE": Signal("AR_OUTPACE_REVENUE", "应收账款周转天数", 0.20, +1, "pct",
                                 "应收账款 按欠款方归集 前五名 坏账准备", "NOTES"),
    "INVENTORY_DAYS_DETERIORATION": Signal("INVENTORY_DAYS_DETERIORATION", "存货周转天数", 0.20, +1, "pct",
                                           "存货跌价准备 计提 可变现净值", "NOTES"),
    "GROSS_MARGIN_SHIFT": Signal("GROSS_MARGIN_SHIFT", "毛利率", 0.015, 0, "pp", "毛利率 变动 原因 售价 成本", "MDA"),
    "LARGE_IMPAIRMENT": Signal("LARGE_IMPAIRMENT", "资产减值损失占总资产比", 0.004, +1, "pp",
                               "存货跌价准备 资产减值 计提", "NOTES"),
    "INTEREST_DEBT_JUMP": Signal("INTEREST_DEBT_JUMP", "有息负债率", 0.02, +1, "pp",
                                 "银行借款 债券 有息负债 融资", "ANNOUNCEMENT"),
}
CUM_MONTH = {3: 1, 6: 2, 9: 3, 12: 4}
# "Large" is a statement about the LEVEL, not the change: an impairment as big as last year's is still large.
# For these codes the signal is the level itself (no year-over-year difference).
LEVEL_SIGNALS = {"LARGE_IMPAIRMENT"}


def _year_ago(d: date) -> date:
    return date(d.year - 1, d.month, d.day)


class FactTable:
    """Latest visible value per (metric, period) for one company, chosen like PITDataService.fact() but without
    logging conflicts or queries (peers are read only for statistics)."""

    def __init__(self, svc, ticker: str):
        best: dict[tuple, object] = {}
        for f in svc._load_facts(ticker):
            if f.statement_scope != "consolidated" or f.available_at > svc.as_of:
                continue
            k = (f.metric, f.period_end)
            cur = best.get(k)
            key = (svc.registry.priority(f.source), -f.revision_id)
            if cur is None or key < cur[0]:
                best[k] = (key, f)
        self.rows = {k: v[1] for k, v in best.items()}
        self.periods = sorted({k[1] for k in self.rows})

    def v(self, metric: str, pe: date) -> Optional[float]:
        f = self.rows.get((metric, pe))
        return None if f is None else float(f.value)

    def f(self, metric: str, pe: date):
        return self.rows.get((metric, pe))


def _level(code: str, t: FactTable, pe: date) -> Optional[float]:
    """The ratio whose year-over-year change is the signal; None when it cannot be computed."""
    g = lambda m: t.v(m, pe)  # noqa: E731
    m = CUM_MONTH[pe.month] * 3
    if code == "PROFIT_OCF_DIVERGENCE":
        np_, ocf, ta = g("net_profit"), g("operating_cash_flow"), g("total_assets")
        return (np_ - ocf) / ta if None not in (np_, ocf, ta) and ta > 0 else None
    if code == "AR_OUTPACE_REVENUE":
        ar, rev = g("accounts_receivable"), g("operating_revenue")
        return ar / (rev * 12 / m) if None not in (ar, rev) and ar > 0 and rev > 0 else None
    if code == "INVENTORY_DAYS_DETERIORATION":
        inv, cost = g("inventory"), g("operating_cost")
        return inv / (cost * 12 / m) if None not in (inv, cost) and inv > 0 and cost > 0 else None
    if code == "GROSS_MARGIN_SHIFT":
        rev, cost = g("operating_revenue"), g("operating_cost")
        return 1 - cost / rev if None not in (rev, cost) and rev > 0 else None
    if code == "LARGE_IMPAIRMENT":
        imp, ta = g("asset_impairment_loss"), g("total_assets")
        return max(imp, 0.0) / ta if None not in (imp, ta) and ta > 0 else None
    if code == "INTEREST_DEBT_JUMP":
        d, ta = g("interest_bearing_debt"), g("total_assets")
        return d / ta if None not in (d, ta) and ta > 0 else None
    return None


def signal_value(code: str, t: FactTable, pe: date) -> Optional[tuple[float, float, float]]:
    """(change shown to the reader, statistic space value, level now) or None."""
    sig = SIGNALS[code]
    now = _level(code, t, pe)
    if now is None:
        return None
    if code in LEVEL_SIGNALS:
        return now, now, now
    before = _level(code, t, _year_ago(pe))
    if before is None:
        return None
    if sig.unit == "pp":
        return now - before, now - before, now
    if before <= 0:
        return None
    return now / before - 1, math.log(now / before), now      # log space for the Z statistic


def _floor_in_stat_space(sig: Signal) -> float:
    return math.log(1 + sig.floor) if sig.unit == "pct" else sig.floor


def detect(calc, ticker: str, pe: date) -> list:
    """Return AnomalyFlag objects (with MetricRecords registered in calc.store)."""
    from .calculators import AnomalyFlag
    from .serving import period_label

    svc = calc.svc
    mine = FactTable(svc, ticker)
    secs = {s.ticker: s for s in svc.securities()}
    mine_demo = bool(secs[ticker].is_demo) if ticker in secs else False
    peers = [t for t, s in secs.items() if t != ticker and bool(s.is_demo) == mine_demo]   # real vs fictitious never mix
    peer_tables = [FactTable(svc, p) for p in peers]
    flags = []
    hist_periods = [p for p in mine.periods if p < pe][-(HISTORY + 4):]
    for code, sig in SIGNALS.items():
        cur = signal_value(code, mine, pe)
        if cur is None:
            continue
        shown, x, level = cur
        if code in LEVEL_GATES and level < LEVEL_GATES[code]:
            continue
        floor = _floor_in_stat_space(sig)
        if abs(x) < floor or (sig.direction > 0 and x < 0) or (sig.direction < 0 and x > 0):
            continue
        hist = [h[1] for p in hist_periods if (h := signal_value(code, mine, p)) is not None][-HISTORY:]
        z_self = None
        if len(hist) >= MIN_HISTORY:
            sd = max(stdev(hist), floor / 2)
            z_self = (x - mean(hist)) / sd
        pv = [h[1] for t in peer_tables if (h := signal_value(code, t, pe)) is not None]
        z_peer = None
        if len(pv) >= MIN_PEERS:
            med = median(pv)
            mad = 1.4826 * median([abs(v - med) for v in pv])
            z_peer = (x - med) / max(mad, floor / 2)
        zs = [z for z in (z_self, z_peer) if z is not None]
        if not zs or max(abs(z) for z in zs) <= Z_TRIGGER:
            continue
        zmax = max(abs(z) for z in zs)
        sev = "HIGH" if zmax > Z_HIGH and abs(x) >= 2 * floor else "MEDIUM"
        lbl = period_label(pe)
        ins = [f for m in ("net_profit", "operating_cash_flow", "total_assets", "accounts_receivable", "operating_revenue",
                           "inventory", "operating_cost", "asset_impairment_loss", "interest_bearing_debt")
               for f in (mine.f(m, pe), mine.f(m, _year_ago(pe))) if f is not None and m in _USED[code]]
        r_sig = calc._record(ticker, f"anom_{code.lower()}", lbl, shown, "ratio", ins,
                             extra={"cn": sig.label + "同比变化", "level": level, "stat_space_value": x})
        ids = [r_sig.metric_id]
        text = f"{sig.label}同比{'变动' if sig.unit == 'pp' else ''}{shown:+.2%}" if sig.unit == "pp" else \
            f"{sig.label}同比{shown:+.1%}"
        if z_self is not None:
            r_zs = calc._record(ticker, f"anom_{code.lower()}_z_self", lbl, z_self, "stat", [r_sig],
                                extra={"cn": "相对自身历史的Z值"})
            ids.append(r_zs.metric_id)
            text += f"，相对自身历史Z={z_self:+.2f}"
        if z_peer is not None:
            r_zp = calc._record(ticker, f"anom_{code.lower()}_z_peer", lbl, z_peer, "stat", [r_sig],
                                extra={"cn": "相对同行的Z值"})
            ids.append(r_zp.metric_id)
            text += f"，相对同行Z={z_peer:+.2f}"
        flags.append(AnomalyFlag(code=code, severity=sev, metric_ids=ids, message=text, suggested_query=sig.query,
                                 section_type=sig.section, category="statistical", statistical_z=z_self, peer_z=z_peer))
    return flags


_USED = {
    "PROFIT_OCF_DIVERGENCE": {"net_profit", "operating_cash_flow", "total_assets"},
    "AR_OUTPACE_REVENUE": {"accounts_receivable", "operating_revenue"},
    "INVENTORY_DAYS_DETERIORATION": {"inventory", "operating_cost"},
    "GROSS_MARGIN_SHIFT": {"operating_revenue", "operating_cost"},
    "LARGE_IMPAIRMENT": {"asset_impairment_loss", "total_assets"},
    "INTEREST_DEBT_JUMP": {"interest_bearing_debt", "total_assets"},
}
