"""Build the PIT factor panel for scenario D (PRD v1.2 Epic 4): 24 A-share companies, statements 2020Q1-2025Q4,
daily prices with adjusted closes. Statements only (no PDFs), so it is cheap to rebuild.

  python scripts/build_factor_panel.py       # needs network once; responses are cached in data/raw_cache/
Output: benchmarks/factor_panel/{securities,financial_facts,market_prices}.csv + manifest.json

Point-in-time rules (stricter than the real snapshot, because there is no cninfo timestamp for the 16 new companies):
* published_at = Eastmoney NOTICE_DATE (a date). The time of day is unknown, so available_at is the NEXT trading
  day 09:30 after it. A factor never sees a report on the day it was filed.
* Eastmoney's UPDATE_DATE is about NOTICE_DATE + 1 year for almost every row (the period is re-issued as the
  prior-year comparative in the next year's report), so it is NOT a restatement signal and is not used to hide data.
* Restatements are handled where they were confirmed: for the 8 companies whose original filing PDFs were parsed
  (benchmarks/real_snapshot/filing_crosscheck.json, 11 periods), the PDF value is revision 1 visible from the filing
  date and the Eastmoney value is revision 2 visible from UPDATE_DATE.
* KNOWN LIMIT: for the 16 added companies no PDF check exists. A period restated after its first filing may carry
  the restated number from the filing date on. Among the 8 checked companies 11 of 216 periods were restated.
"""
import csv
import json
import sys
from bisect import bisect_right
from datetime import date, datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_real_snapshot import CUM, flows, instants  # noqa: E402
from data_platform.providers import EastmoneyStatements, HttpCache, YahooPrices  # noqa: E402
from fetch_real import COMPANIES  # noqa: E402

OUT = ROOT / "benchmarks" / "factor_panel"
# ticker, name, SW level 1, SW level 2, universe
NEW = [
    ("300274.SZ", "阳光电源", "电力设备", "逆变器", "new_energy"),
    ("002460.SZ", "赣锋锂业", "有色金属", "锂", "new_energy"),
    ("002709.SZ", "天赐材料", "电力设备", "锂电材料", "new_energy"),
    ("002812.SZ", "恩捷股份", "电力设备", "锂电材料", "new_energy"),
    ("300450.SZ", "先导智能", "电力设备", "锂电设备", "new_energy"),
    ("600438.SH", "通威股份", "电力设备", "光伏产业链", "new_energy"),
    ("002129.SZ", "TCL中环", "电力设备", "光伏硅片", "new_energy"),
    ("601633.SH", "长城汽车", "汽车", "乘用车", "new_energy"),
    ("600887.SH", "伊利股份", "食品饮料", "乳制品", "consumer"),
    ("603288.SH", "海天味业", "食品饮料", "调味品", "consumer"),
    ("000568.SZ", "泸州老窖", "食品饮料", "白酒", "consumer"),
    ("600809.SH", "山西汾酒", "食品饮料", "白酒", "consumer"),
    ("002304.SZ", "洋河股份", "食品饮料", "白酒", "consumer"),
    ("600690.SH", "海尔智家", "家用电器", "白色家电", "consumer"),
    ("002032.SZ", "苏泊尔", "家用电器", "小家电", "consumer"),
    ("600132.SH", "重庆啤酒", "食品饮料", "啤酒", "consumer"),
]
UNIVERSE_OF = {"电力设备": "new_energy", "汽车": "new_energy", "食品饮料": "consumer", "家用电器": "consumer"}
FIRST_PERIOD = date(2020, 3, 31)
CROSSCHECK = ROOT / "benchmarks" / "real_snapshot" / "filing_crosscheck.json"
PRICE_START, PRICE_END = datetime(2021, 12, 1), datetime(2026, 11, 9)


def next_open(days: list[date], d: date) -> datetime:
    """09:30 of the first trading day strictly after d."""
    i = bisect_right(days, d)
    return datetime.combine(days[i] if i < len(days) else d + timedelta(days=1), time(9, 30))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    http = HttpCache()
    em, yh = EastmoneyStatements(http), YahooPrices(http)
    universe = [(t, n, l1, l2, UNIVERSE_OF[l1]) for t, n, l1, l2, _ in COMPANIES] + NEW
    cc = json.loads(CROSSCHECK.read_text(encoding="utf-8")) if CROSSCHECK.exists() else {"checks": [], "restated_periods": []}
    restated = {(r["ticker"], r["period"]): set(r["metrics"]) for r in cc["restated_periods"]}
    filing_val = {(c["ticker"], c["period"], c["metric"]): c["filing_pdf"] for c in cc["checks"]}
    facts, prices, secs, stats = [], [], [], {"periods": 0, "restated_revisions": 0}
    for t, name, l1, l2, uni in universe:
        bars = yh.daily(t.replace(".SH", ".SS"), PRICE_START, PRICE_END)
        days = sorted({b["date"] for b in bars})
        secs.append([t, name, l1, l2, uni])
        inc = {r["REPORT_DATE"][:10]: r for r in em.fetch(t, "income")}
        bal = {r["REPORT_DATE"][:10]: r for r in em.fetch(t, "balance")}
        cfl = {r["REPORT_DATE"][:10]: r for r in em.fetch(t, "cashflow")}
        n_late = 0
        for rd in sorted(set(inc) | set(bal)):
            pe = date.fromisoformat(rd)
            if pe < FIRST_PERIOD or pe.month not in CUM:
                continue
            i, b, c = inc.get(rd, {}), bal.get(rd, {}), cfl.get(rd, {})
            src = i or b or c
            notice = datetime.fromisoformat(src["NOTICE_DATE"][:10]).date()
            upd = datetime.fromisoformat((i.get("UPDATE_DATE") or src["NOTICE_DATE"])[:10]).date()
            avail = next_open(days, notice)
            stats["periods"] += 1
            values = {**{m: (CUM[pe.month], v) for m, v in flows(i, c).items()},
                      **{m: ("instant", v) for m, v in instants(b).items()}}
            for m, (ptype, v) in values.items():
                if m in restated.get((t, rd), ()) and (t, rd, m) in filing_val:
                    # confirmed restatement: original filing value first, restated value from UPDATE_DATE
                    facts.append([t, m, rd, ptype, f"{filing_val[(t, rd, m)]:.2f}", notice.isoformat() + "T00:00:00",
                                  avail.isoformat(), 1, "false", "cninfo_pdf_parse"])
                    facts.append([t, m, rd, ptype, f"{v:.2f}", upd.isoformat() + "T00:00:00",
                                  next_open(days, upd).isoformat(), 2, "true", "eastmoney_f10"])
                    stats["restated_revisions"] += 1
                    n_late += 1
                else:
                    facts.append([t, m, rd, ptype, f"{v:.2f}", notice.isoformat() + "T00:00:00",
                                  avail.isoformat(), 1, "false", "eastmoney_f10"])
        for bar in bars:
            prices.append([t, bar["date"].isoformat(), round(bar["close"], 4), round(bar["adj_close"], 6), bar["volume"],
                           datetime.combine(bar["date"], time(15, 0)).isoformat()])
        print(f"{t} {name}: bars={len(bars)} restated_revisions={n_late}", flush=True)

    def write(name, header, rows):
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    write("securities.csv", ["ticker", "name", "industry_sw_l1", "industry_sw_l2", "universe"], secs)
    write("financial_facts.csv", ["ticker", "metric", "period_end", "period_type", "value", "published_at", "available_at",
                                  "revision_id", "is_restated", "source"], facts)
    write("market_prices.csv", ["ticker", "trade_date", "close", "adj_close", "volume", "available_at"], prices)
    manifest = {"built_at": datetime.now().isoformat(timespec="seconds"), "n_securities": len(secs), "n_facts": len(facts),
                "n_prices": len(prices), **stats,
                "universe_counts": {u: sum(1 for s in secs if s[4] == u) for u in ("new_energy", "consumer")},
                "survivorship_note": "All 24 companies are listed today. Companies delisted or removed from these "
                                     "industries during 2023-2025 are not in the panel, so factor results are optimistic "
                                     "for quality-type factors. This is a small illustrative sample, not a research-grade universe."}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
