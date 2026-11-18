"""Collect the data that only AKShare can supply from your network, into data/akshare_cache/.

Run these where AKShare works (this project's own network blocks it). The app never calls AKShare itself; it reads
the files below, so you can run this once a week and copy the folder over.

  pip install akshare
  python scripts/akshare_collect.py industry                      # peers: industry members with market cap (~2-4 min)
  python scripts/akshare_collect.py prices --tickers 430047.BJ 601899.SH   # daily prices (Beijing listings; any ticker)
  python scripts/akshare_collect.py all --tickers 430047.BJ        # both

Files written
  data/akshare_cache/industry_members.json  {"generated_at", "boards": {board: [{code, name, mktcap}]}, "by_code": {code: board}}
  data/akshare_cache/prices_<code>.json     [{date, open, high, low, close, volume}]  (unadjusted, last 3 years)

AKShare calls used (all Eastmoney-backed)
  stock_board_industry_name_em()            list of industry boards
  stock_board_industry_cons_em(symbol)      members of one board, with 总市值
  stock_zh_a_hist(symbol, period="daily")   daily bars, Shanghai / Shenzhen / Beijing
"""
import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "akshare_cache"


def industry() -> None:
    import akshare as ak

    boards = ak.stock_board_industry_name_em()
    names = boards["板块名称"].tolist()
    print(f"{len(names)} industry boards")
    members, by_code = {}, {}
    for i, name in enumerate(names, 1):
        for attempt in range(3):
            try:
                df = ak.stock_board_industry_cons_em(symbol=name)
                break
            except Exception as exc:  # noqa: BLE001 - flaky public endpoint
                print(f"  retry {name}: {type(exc).__name__}")
                time.sleep(2 * (attempt + 1))
        else:
            continue
        rows = [{"code": str(r["代码"]).zfill(6), "name": r["名称"], "mktcap": float(r["总市值"]) if r.get("总市值") == r.get("总市值") else None}
                for _, r in df.iterrows()] if "总市值" in df.columns else [
            {"code": str(r["代码"]).zfill(6), "name": r["名称"], "mktcap": None} for _, r in df.iterrows()]
        members[name] = rows
        for r in rows:
            by_code.setdefault(r["code"], name)
        print(f"  [{i}/{len(names)}] {name}: {len(rows)}", flush=True)
        time.sleep(0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "industry_members.json").write_text(json.dumps({"generated_at": datetime.now().isoformat(), "boards": members,
                                                         "by_code": by_code}, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT / "industry_members.json", "companies:", len(by_code))


def prices(tickers: list[str]) -> None:
    import akshare as ak

    OUT.mkdir(parents=True, exist_ok=True)
    start = (date.today() - timedelta(days=3 * 365 + 10)).strftime("%Y%m%d")
    for t in tickers:
        code = t[:6]
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=date.today().strftime("%Y%m%d"), adjust="")
        rows = [{"date": str(r["日期"])[:10], "open": float(r["开盘"]), "high": float(r["最高"]), "low": float(r["最低"]),
                 "close": float(r["收盘"]), "volume": float(r["成交量"]) * 100} for _, r in df.iterrows()]
        (OUT / f"prices_{code}.json").write_text(json.dumps(rows), encoding="utf-8")
        print(t, len(rows), "bars")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["industry", "prices", "all"])
    ap.add_argument("--tickers", nargs="*", default=[])
    a = ap.parse_args()
    if a.what in ("industry", "all"):
        industry()
    if a.what in ("prices", "all"):
        if not a.tickers:
            sys.exit("--tickers is required for prices")
        prices(a.tickers)
