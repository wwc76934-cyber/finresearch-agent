"""Step 2 of the real snapshot: turn the cached raw data into benchmarks/real_snapshot/ (same layout as the
demo Golden Snapshot, so the same ingestion, PIT serving, RAG and evaluation code runs on it).

  python scripts/fetch_real.py --pdfs      # once, needs network
  python scripts/build_real_snapshot.py    # offline from data/raw_cache/

PIT rules used here
* published_at of a periodic report = the cninfo disclosure date of that report (date-only entries are the
  official 披露日, so they are treated as disclosed before the open); if the report is older than the
  announcement window, Eastmoney NOTICE_DATE is used.
* available_at = the first trading-day 09:30 at or after published_at (announcements time-stamped after
  09:30 become usable the next trading day).
* Eastmoney values can be restated later (its UPDATE_DATE moves). Revenue and net profit are cross-checked
  against the consolidated income statement parsed from the original PDF. A mismatch becomes two revisions:
  rev 1 = original filing value at the original date, rev 2 = Eastmoney value at UPDATE_DATE.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from bisect import bisect_left
from datetime import date, datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_platform.providers import CACHE, Cninfo, EastmoneyStatements, HttpCache, YahooPrices  # noqa: E402
from fetch_real import COMPANIES, END, KEY_WORDS, REPORT_PERIODS, START, periodic_kind  # noqa: E402
from rag_engine.pdf_parser import parse_pdf, statement_value  # noqa: E402

OUT = ROOT / "benchmarks" / "real_snapshot"
RETRIEVED = datetime(2026, 11, 10, 12, 0)
KIND_DOC = {"A": "annual_report", "H1": "semi_annual_report", "Q1": "q1_report", "Q3": "q3_report"}
KIND_END = {"A": (12, 31), "H1": (6, 30), "Q1": (3, 31), "Q3": (9, 30)}
CUM = {3: "cumulative_q1", 6: "cumulative_h1", 9: "cumulative_q3", 12: "annual"}
# metric -> (statement, row patterns) read from the original filing PDF (first number = current period)
FILING_ROWS = {
    "operating_revenue": ("合并利润表", (r"其中营业收入", r"营业收入")),
    "operating_cost": ("合并利润表", (r"其中营业成本", r"营业成本")),
    "net_profit": ("合并利润表", (r"归属于母公司(股东|所有者)的净利润",)),
    "operating_cash_flow": ("合并现金流量表", (r"经营活动产生的现金流量净额",)),
    "total_assets": ("合并资产负债表", (r"资产总计",)),
    "total_liabilities": ("合并资产负债表", (r"负债合计",)),
    "total_equity": ("合并资产负债表", (r"(所有者|股东)权益合计",)),
    "parent_equity": ("合并资产负债表", (r"归属于母公司(所有者|股东)权益合计",)),
    "accounts_receivable": ("合并资产负债表", (r"应收账款",)),
    "inventory": ("合并资产负债表", (r"存货",)),
    "cash": ("合并资产负债表", (r"货币资金",)),
}
CROSSCHECK_KEYS = ("operating_revenue", "net_profit")   # used to decide whether a period was restated


def num(row: dict, *keys) -> float | None:
    for k in keys:
        v = row.get(k)
        if v is not None:
            return float(v)
    return None


def flows(inc: dict, cf: dict) -> dict[str, float]:
    imp = None
    if inc.get("ASSET_IMPAIRMENT_INCOME") is not None or inc.get("CREDIT_IMPAIRMENT_INCOME") is not None:
        imp = -(num(inc, "ASSET_IMPAIRMENT_INCOME") or 0) - (num(inc, "CREDIT_IMPAIRMENT_INCOME") or 0)
    elif inc.get("ASSET_IMPAIRMENT_LOSS") is not None:
        imp = num(inc, "ASSET_IMPAIRMENT_LOSS")
    da = [num(cf, k) for k in ("FA_IR_DEPR", "IA_AMORTIZE", "LPE_AMORTIZE", "USERIGHT_ASSET_AMORTIZE")]
    out = {"operating_revenue": num(inc, "OPERATE_INCOME", "TOTAL_OPERATE_INCOME"),
           "operating_cost": num(inc, "OPERATE_COST"), "rd_expense": num(inc, "RESEARCH_EXPENSE"),
           "asset_impairment_loss": imp, "net_profit": num(inc, "PARENT_NETPROFIT"),
           "operating_cash_flow": num(cf, "NETCASH_OPERATE"), "total_profit": num(inc, "TOTAL_PROFIT"),
           "interest_expense": num(inc, "FE_INTEREST_EXPENSE"),
           # D&A is only in the indirect cash-flow supplement of half-year and annual reports
           "depreciation_amortization": sum(d for d in da if d is not None) if any(d is not None for d in da) else None}
    return {k: v for k, v in out.items() if v is not None}


def instants(b: dict) -> dict[str, float]:
    debt = [num(b, k) for k in ("SHORT_LOAN", "LONG_LOAN", "BOND_PAYABLE", "NONCURRENT_LIAB_1YEAR")]
    out = {"accounts_receivable": num(b, "ACCOUNTS_RECE"), "inventory": num(b, "INVENTORY"),
           "total_assets": num(b, "TOTAL_ASSETS"), "total_liabilities": num(b, "TOTAL_LIABILITIES"),
           "total_equity": num(b, "TOTAL_EQUITY"), "parent_equity": num(b, "TOTAL_PARENT_EQUITY"),
           "interest_bearing_debt": sum(d for d in debt if d is not None) if any(d is not None for d in debt) else None,
           "goodwill": num(b, "GOODWILL") or 0.0, "cash": num(b, "MONETARYFUNDS"),
           "shares_outstanding": num(b, "SHARE_CAPITAL")}
    return {k: v for k, v in out.items() if v is not None}


class Calendar:
    def __init__(self, days: list[date]):
        self.days = sorted(set(days))

    def open_at_or_after(self, ts: datetime) -> datetime:
        d = ts.date()
        if ts.time() > time(9, 30):
            d += timedelta(days=1)
        i = bisect_left(self.days, d)
        d = self.days[i] if i < len(self.days) else d
        return datetime.combine(d, time(9, 30))


def md_cache(adjunct: str, pdf: bytes, title: str) -> tuple[str, dict]:
    p = CACHE / "md" / (hashlib.sha256(adjunct.encode()).hexdigest()[:24] + ".md")
    if p.exists():
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        return p.read_text(encoding="utf-8"), meta
    t0 = datetime.now()
    md, stats = parse_pdf(pdf, title)
    print(f"  parsed {title} pages={stats['pages']} kept={stats['kept_pages']} tables={stats['tables']} "
          f"{(datetime.now() - t0).seconds}s", flush=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md, encoding="utf-8")
    p.with_suffix(".json").write_text(json.dumps(stats), encoding="utf-8")
    return md, stats


def ann_text(pdf: bytes, title: str, limit: int = 3000) -> str:
    import pymupdf

    doc = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        text = "".join(doc[i].get_text("text") for i in range(min(doc.page_count, 4)))
    finally:
        doc.close()
    body = "".join(l.strip() for l in text.splitlines() if l.strip())[:limit]
    return f"# {title}\n\n<!-- page 1 -->\n## 公告正文\n{body}\n"


def main() -> None:
    http = HttpCache()
    em, cn, yh = EastmoneyStatements(http), Cninfo(http), YahooPrices(http)
    (OUT / "documents").mkdir(parents=True, exist_ok=True)
    facts, prices, docs, checks, restated_periods = [], [], [], [], []
    secs = []
    for t, name, l1, l2, template in COMPANIES:
        code, mkt = t[:6], t[-2:]
        bars = yh.daily(t.replace(".SH", ".SS"), datetime(2021, 12, 1), datetime(2026, 11, 9))
        cal = Calendar([b["date"] for b in bars])
        secs.append([t, name, "2000-01-01", l1, l2, template, "false"])
        inc = {r["REPORT_DATE"][:10]: r for r in em.fetch(t, "income")}
        bal = {r["REPORT_DATE"][:10]: r for r in em.fetch(t, "balance")}
        cfl = {r["REPORT_DATE"][:10]: r for r in em.fetch(t, "cashflow")}
        anns = cn.announcements(code, mkt, START, END)
        # ---- periodic reports: latest version per period is the reference document
        by_period: dict[str, list[dict]] = {}
        for a in anns:
            k = periodic_kind(a["announcementTitle"])
            if k and "".join(k) in REPORT_PERIODS:
                by_period.setdefault("".join(k), []).append(a)
        filing_vals: dict[str, dict] = {}
        for per, versions in sorted(by_period.items()):
            versions.sort(key=lambda a: a["announcementTime"])
            y, kind = per[:4], per[4:]
            pe = date(int(y), *KIND_END[kind])
            for v_no, a in enumerate(versions, 1):
                md, stats = md_cache(a["adjunctUrl"], cn.pdf(a["adjunctUrl"]), f"{name}（{t}）{a['announcementTitle']}")
                did = f"doc_{code}_{pe.year}Q{(pe.month - 1) // 3 + 1}" + ("" if v_no == 1 else f"_v{v_no}")
                pub = Cninfo.published_at(a)
                (OUT / "documents" / f"{did}.md").write_text(md, encoding="utf-8")
                docs.append({"document_id": did, "ticker": t, "title": f"{name}{a['announcementTitle']}",
                             "doc_type": KIND_DOC[kind], "period_end": pe.isoformat(), "published_at": pub.isoformat(),
                             "available_at": cal.open_at_or_after(pub).isoformat(),
                             "content_hash": hashlib.sha256(md.encode()).hexdigest(),
                             "path": f"real_snapshot/documents/{did}.md", "version": v_no})
                if v_no == 1:
                    filing_vals[pe.isoformat()] = {"doc": did, "published": pub, "pages": stats.get("pages"),
                                                   **{m: statement_value(md, st_name, rows)
                                                      for m, (st_name, rows) in FILING_ROWS.items()}}
        # ---- key announcements
        key = [a for a in anns if not periodic_kind(a["announcementTitle"])
               and any(k in a["announcementTitle"] for k in KEY_WORDS)][:12]
        for a in key:
            pub = Cninfo.published_at(a)
            did = f"ann_{code}_{a['announcementId']}"
            text = ann_text(cn.pdf(a["adjunctUrl"]), a["announcementTitle"])
            (OUT / "documents" / f"{did}.md").write_text(text, encoding="utf-8")
            docs.append({"document_id": did, "ticker": t, "title": a["announcementTitle"], "doc_type": "announcement",
                         "period_end": "", "published_at": pub.isoformat(),
                         "available_at": cal.open_at_or_after(pub).isoformat(),
                         "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                         "path": f"real_snapshot/documents/{did}.md", "version": 1})
        # ---- statements -> facts
        shares_hist = []
        for rd in sorted(set(inc) | set(bal)):
            pe = date.fromisoformat(rd)
            if pe < date(2021, 3, 31) or pe.month not in CUM:
                continue
            i, b, c = inc.get(rd, {}), bal.get(rd, {}), cfl.get(rd, {})
            src = i or b or c
            fv = filing_vals.get(rd)
            notice = datetime.fromisoformat(src["NOTICE_DATE"][:10])
            pub = fv["published"] if fv else notice
            avail = cal.open_at_or_after(pub)
            upd = datetime.fromisoformat((i.get("UPDATE_DATE") or src["NOTICE_DATE"])[:10])
            doc_id = fv["doc"] if fv else None
            base = dict(ticker=t, scope="consolidated", period_end=pe, published=pub, available=avail,
                        doc=doc_id, rev=1, reliable=True, source="eastmoney_f10")
            values = {**{m: (CUM[pe.month], v) for m, v in flows(i, c).items()},
                      **{m: ("instant", v) for m, v in instants(b).items()}}
            late_update = upd.date() > notice.date() + timedelta(days=30)
            diffs = {m: abs(fv[m] - v) / max(abs(v), 1) for m, (_, v) in values.items()
                     if fv and fv.get(m) is not None}
            for m, d_ in diffs.items():
                checks.append({"ticker": t, "period": rd, "metric": m, "eastmoney": values[m][1], "filing_pdf": fv[m],
                               "rel_diff": d_, "update_date": upd.date().isoformat()})
            restated_metrics = [m for m, d_ in diffs.items() if d_ > 0.003 and late_update]
            if restated_metrics:
                # Eastmoney overwrote part of this period with later restated figures (e.g. the 2024 质保费用
                # reclassification into 营业成本). Each metric readable from the original PDF gets revision 1 =
                # filing value, revision 2 = Eastmoney value at UPDATE_DATE. Metrics the PDF cannot confirm are
                # made visible from UPDATE_DATE only, so no restated number leaks into an earlier as_of.
                restated_periods.append({"ticker": t, "period": rd, "update_date": upd.date().isoformat(),
                                         "metrics": restated_metrics})
            base_late = {**base, "published": upd, "available": cal.open_at_or_after(upd), "doc": None}
            for m, (ptype, v) in values.items():
                row = {**base, "metric": m, "period_type": ptype, "value": v}
                if m in diffs and diffs[m] > 0.003:
                    if late_update:
                        facts.append({**row, "value": fv[m]})
                        facts.append({**row, **{k: base_late[k] for k in ("published", "available", "doc")}, "rev": 2})
                    else:   # same-date disagreement between two sources -> keep both, conflict is reported
                        facts.append(row)
                        facts.append({**row, "value": fv[m], "source": "cninfo_pdf_parse"})
                elif m not in diffs and restated_metrics:
                    facts.append({**row, **{k: base_late[k] for k in ("published", "available", "doc")}})
                else:
                    facts.append(row)
            if b.get("SHARE_CAPITAL"):
                shares_hist.append((avail, float(b["SHARE_CAPITAL"])))
        shares_hist.sort()
        for bar in bars:
            ts = datetime.combine(bar["date"], time(15, 0))
            vis = [s for at, s in shares_hist if at <= ts] or [shares_hist[0][1]]
            sh = vis[-1]
            prices.append([t, bar["date"].isoformat(), round(bar["open"], 3), round(bar["high"], 3), round(bar["low"], 3),
                           round(bar["close"], 3), bar["volume"], round(bar["volume"] / sh * 100, 4),
                           round(bar["close"] * sh, 2), round(bar["close"] * sh, 2), ts.isoformat()])
        print(t, name, "periods", len(by_period), "docs", sum(1 for d in docs if d["ticker"] == t), flush=True)

    with open(OUT / "securities.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "name", "list_date", "industry_sw_l1", "industry_sw_l2", "template", "is_demo"])
        w.writerows(secs)
    with open(OUT / "financial_facts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fact_id", "ticker", "metric", "statement_scope", "period_type", "value", "unit", "period_end",
                    "published_at", "available_at", "retrieved_at", "source", "source_document_id", "revision_id",
                    "is_restated", "published_at_reliable"])
        for x in facts:
            pe = x["period_end"]
            fid = f"fact_{x['ticker'][:6]}_{pe:%Y%m%d}_{x['metric']}_con_{x['source'][-6:]}_v{x['rev']}"
            w.writerow([fid, x["ticker"], x["metric"], "consolidated", x["period_type"], f"{x['value']:.2f}", "CNY",
                        pe.isoformat(), x["published"].isoformat(), x["available"].isoformat(), RETRIEVED.isoformat(),
                        x["source"], x["doc"] or "", x["rev"], "true" if x["rev"] > 1 else "false", "true"])
    with open(OUT / "market_prices.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "trade_date", "open", "high", "low", "close", "volume", "turnover_rate", "total_mv",
                    "float_mv", "available_at"])
        w.writerows(prices)
    with open(OUT / "documents.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(docs[0].keys()))
        w.writeheader()
        w.writerows(docs)
    (OUT / "filing_crosscheck.json").write_text(json.dumps({"checks": checks, "restated_periods": restated_periods},
                                                           ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("restated periods:", restated_periods)
    ok = sum(1 for c in checks if c["rel_diff"] <= 0.003)
    print(f"facts={len(facts)} prices={len(prices)} docs={len(docs)} crosscheck={ok}/{len(checks)} within 0.3%")


if __name__ == "__main__":
    import faulthandler

    faulthandler.enable()  # PyMuPDF crashes in native code would otherwise vanish without a trace
    main()
