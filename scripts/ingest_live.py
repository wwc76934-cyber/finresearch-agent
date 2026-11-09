"""Live ingestion via AKShare (needs network and `pip install akshare`).

Currently loads daily bars for real A-share tickers into market_prices.
Statement ingestion is intentionally not wired: AKShare statement endpoints do
not carry a reliable 披露时间, and facts without published_at would be
pit_unsafe. Join them with 巨潮 announcement timestamps before enabling.

  python scripts/ingest_live.py 300750.SZ 20220101 20251231 --shares 4.4e9
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from data_platform.db import MarketPriceRow, get_session_factory  # noqa: E402
from data_platform.ingestion import AkshareProvider  # noqa: E402
from data_platform.registry import DataSourceRegistry  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--shares", type=float, required=True, help="total shares, used for market value")
    a = ap.parse_args()
    reg = DataSourceRegistry()
    reg.sources["akshare"].enabled = True
    bars = AkshareProvider(reg).prices(a.ticker, a.start, a.end, a.shares)
    with get_session_factory()() as s:
        for b in bars:
            s.merge(MarketPriceRow(**b.model_dump()))
        s.commit()
    print(f"loaded {len(bars)} bars for {a.ticker}")
