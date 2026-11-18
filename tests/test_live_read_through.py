"""Read-Through ingestion (v2.0 Epic 1): a company that is not stored locally is fetched on demand, written with PIT
timestamps, and covered by a watermark. Network-free: a fake provider stands in for Eastmoney / cninfo / Yahoo.
Set RUN_LIVE_TESTS=1 to also run the two tests that call the real endpoints."""
import os
from collections import Counter
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from data_platform import PITDataService
from data_platform.ashare_catalog import AShareCatalog, set_catalog
from data_platform.db import FinancialFactRow, IngestionWatermarkRow, MarketPriceRow, SecurityRow, get_session_factory
from data_platform.live import LiveMarketProvider, ReadThroughIngestor, set_provider_factory

NOW = datetime(2026, 11, 18, 12, 0)
CATALOG = AShareCatalog.from_stock_list([
    {"code": "300308", "zwjc": "中际旭创", "pinyin": "zjxc", "orgId": "9900022016", "category": "A股"},
    {"code": "601899", "zwjc": "紫金矿业", "pinyin": "zjky", "orgId": "gssh0601899", "category": "A股"},
    {"code": "600519", "zwjc": "贵州茅台", "pinyin": "gzmt", "orgId": "gssh0600519", "category": "A股"},
    {"code": "000001", "zwjc": "平安银行", "pinyin": "payh", "orgId": "gssz0000001", "category": "A股"},
])
# (REPORT_DATE, NOTICE_DATE): the notice is always later than the period end
PERIODS = [("2024-12-31", "2025-03-18"), ("2025-03-31", "2025-04-24"), ("2025-06-30", "2025-08-20"), ("2025-09-30", "2025-10-28")]


class FakeProvider(LiveMarketProvider):
    def __init__(self, drop_notice=(), fail=()):
        self.calls = Counter()
        self.drop_notice, self.fail = set(drop_notice), set(fail)

    def _boom(self, name):
        self.calls[name] += 1
        if name in self.fail:
            raise RuntimeError(f"{name} is down")

    def statements(self, secucode, periods=24):
        self._boom("statements")
        inc, bal, cf = [], [], []
        for i, (rd, nd) in enumerate(PERIODS):
            notice = None if rd in self.drop_notice else nd + " 00:00:00"
            inc.append({"REPORT_DATE": rd + " 00:00:00", "NOTICE_DATE": notice, "OPERATE_INCOME": 1e9 * (i + 1),
                        "OPERATE_COST": 6e8 * (i + 1), "PARENT_NETPROFIT": 1e8 * (i + 1), "TOTAL_PROFIT": 1.2e8})
            bal.append({"REPORT_DATE": rd + " 00:00:00", "NOTICE_DATE": notice, "TOTAL_ASSETS": 5e9, "TOTAL_LIABILITIES": 2e9,
                        "TOTAL_EQUITY": 3e9, "TOTAL_PARENT_EQUITY": 2.9e9, "SHARE_CAPITAL": 8e8, "ACCOUNTS_RECE": 4e8,
                        "INVENTORY": 3e8, "MONETARYFUNDS": 1e9})
            cf.append({"REPORT_DATE": rd + " 00:00:00", "NOTICE_DATE": notice, "NETCASH_OPERATE": 9e7 * (i + 1)})
        return {"income": inc[::-1], "balance": bal[::-1], "cashflow": cf[::-1]}

    def org_info(self, secucode):
        self._boom("org_info")
        return {"EM2016": "通信-通信设备-通信网络设备及器件", "LISTING_DATE": "2012-04-10 00:00:00"}

    def prices(self, symbol, start, end):
        self._boom("prices")
        d, out = start, []
        while d <= min(end, NOW.date()):
            if d.weekday() < 5:
                out.append({"date": d, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1e6})
            d += timedelta(days=1)
        return out

    def announcements(self, entry, start, end, max_pages=3):
        self._boom("announcements")
        ts = int((datetime(2025, 11, 5, 18, 30) - timedelta(hours=8)).replace(tzinfo=timezone.utc).timestamp() * 1000)  # 18:30 Beijing
        return [{"announcementId": "9001", "announcementTitle": "关于回购公司股份方案的公告", "announcementTime": ts,
                 "adjunctUrl": "finalpage/2025-11-05/9001.PDF"}]

    def announcement_text(self, ann, pages=4, limit=3000):
        self._boom("announcement_text")
        return "公司拟以自有资金回购股份，回购金额不低于人民币1亿元，回购期限为董事会审议通过之日起12个月内。"


@pytest.fixture()
def env(tmp_path):
    url = f"sqlite:///{(tmp_path / 'live.db').as_posix()}"
    set_catalog(CATALOG)
    yield url, tmp_path
    set_catalog(None)
    set_provider_factory(None)


def ingestor(env, provider=None, clock=lambda: NOW):
    url, tmp = env
    return ReadThroughIngestor(url, provider=provider or FakeProvider(), catalog=CATALOG, doc_dir=tmp / "docs", clock=clock)


def test_catalog_resolves_code_name_and_pinyin():
    assert CATALOG.resolve("300308.SZ").name == "中际旭创"
    assert CATALOG.resolve("601899").ticker == "601899.SH"
    assert CATALOG.resolve("紫金矿业").code == "601899"
    assert CATALOG.resolve("gzmt").code == "600519"
    assert CATALOG.resolve("不存在的公司") is None
    assert [e.code for e in CATALOG.extract("对比中际旭创和紫金矿业的盈利质量，再看600519")] == ["300308", "601899", "600519"]
    assert len(CATALOG.extract("中际旭创 紫金矿业 贵州茅台 平安银行 600519", limit=2)) == 2


def test_new_company_is_stored_with_real_disclosure_dates(env):
    ing = ingestor(env)
    rep = ing.ensure("300308.SZ", datetime(2025, 12, 31, 15), "hybrid_live")
    assert rep.status in ("fetched", "partial") and rep.counts["facts"] > 0 and rep.counts["prices"] > 0
    assert rep.counts["documents"] == 1 and rep.counts["industry"] == "通信/通信设备"
    with get_session_factory(env[0])() as s:
        facts = list(s.scalars(select(FinancialFactRow).where(FinancialFactRow.ticker == "300308.SZ")))
        sec = s.get(SecurityRow, "300308.SZ")
    assert sec.name == "中际旭创" and sec.is_demo is False and sec.template == "manufacturing"
    assert facts and all(f.published_at.date() > f.period_end for f in facts)     # never the period end
    q3 = next(f for f in facts if f.period_end == date(2025, 9, 30))
    assert q3.published_at == datetime(2025, 10, 28) and q3.available_at == datetime(2025, 10, 29, 9, 30)
    assert q3.pit_estimated is False and q3.source == "eastmoney_f10_live"


def test_pit_filter_hides_a_report_until_the_next_trading_day(env):
    ingestor(env).ensure("300308.SZ", datetime(2025, 12, 31, 15), "hybrid_live")
    day_of = PITDataService(datetime(2025, 10, 28, 18, 0), "x", env[0])
    nxt = PITDataService(datetime(2025, 10, 29, 10, 0), "x", env[0])
    assert day_of.fact("300308.SZ", "operating_revenue", date(2025, 9, 30)) is None
    assert nxt.fact("300308.SZ", "operating_revenue", date(2025, 9, 30)) is not None
    assert PITDataService(datetime(2025, 7, 1), "x", env[0]).fact("300308.SZ", "operating_revenue", date(2025, 6, 30)) is None


def test_announcement_after_the_open_waits_for_the_next_session(env):
    ingestor(env).ensure("300308.SZ", datetime(2025, 12, 31, 15), "hybrid_live")
    with get_session_factory(env[0])() as s:
        from data_platform.db import SourceDocumentRow
        d = s.scalars(select(SourceDocumentRow).where(SourceDocumentRow.ticker == "300308.SZ")).one()
    assert d.published_at.date() == date(2025, 11, 5) and d.available_at > d.published_at
    assert d.available_at.time().hour == 9 and d.available_at.date() == date(2025, 11, 6)


def test_missing_notice_date_uses_statutory_deadline_and_is_flagged(env):
    rep = ingestor(env, FakeProvider(drop_notice={"2025-06-30"})).ensure("300308.SZ", datetime(2025, 12, 31, 15), "hybrid_live")
    with get_session_factory(env[0])() as s:
        h1 = list(s.scalars(select(FinancialFactRow).where(FinancialFactRow.ticker == "300308.SZ",
                                                             FinancialFactRow.period_end == date(2025, 6, 30))))
    assert h1 and all(f.pit_estimated and f.published_at == datetime(2025, 8, 31) for f in h1)
    assert rep.status == "partial" and any("statutory deadline" in w for w in rep.warnings)


def test_watermark_stops_repeat_fetches_until_the_request_outruns_it(env):
    p = FakeProvider()
    ing = ingestor(env, p)
    assert ing.ensure("300308.SZ", datetime(2025, 12, 31, 15), "hybrid_live").status != "hit"
    calls = dict(p.calls)
    assert ing.ensure("300308.SZ", datetime(2025, 12, 31, 15), "hybrid_live").status == "hit"
    assert dict(p.calls) == calls
    with get_session_factory(env[0])() as s:
        wm = s.get(IngestionWatermarkRow, "300308.SZ")
    assert wm.max_as_of_date == NOW.date() and wm.source_status in ("ok", "partial")
    later = ingestor(env, p, clock=lambda: NOW + timedelta(days=3))
    assert later.ensure("300308.SZ", NOW + timedelta(days=3), "hybrid_live").status != "hit"   # a newer day: fetch again
    assert ing.ensure("300308.SZ", datetime(2025, 12, 31, 15), "online_force_refresh").status != "hit"


def test_offline_golden_never_touches_the_provider(env):
    p = FakeProvider()
    rep = ingestor(env, p).ensure("300308.SZ", datetime(2025, 12, 31, 15), "offline_golden")
    assert rep.status == "skipped_offline" and not p.calls


def test_unknown_security_is_reported_not_invented(env):
    assert ingestor(env).ensure("999999.SZ", datetime(2025, 12, 31, 15), "hybrid_live").status == "unknown_security"


def test_one_source_down_gives_partial_not_a_crash(env):
    rep = ingestor(env, FakeProvider(fail={"prices"})).ensure("300308.SZ", datetime(2025, 12, 31, 15), "hybrid_live")
    assert rep.status == "partial" and rep.counts["facts"] > 0 and rep.counts["prices"] == 0
    assert any(w.startswith("prices:") for w in rep.warnings)
    all_down = ingestor(env, FakeProvider(fail={"statements", "org_info", "prices", "announcements"})).ensure(
        "601899.SH", datetime(2025, 12, 31, 15), "hybrid_live")
    assert all_down.status == "failed"


def test_financial_sector_company_is_marked_and_warned(env):
    class Bank(FakeProvider):
        def org_info(self, secucode):
            return {"EM2016": "银行-国有大型银行Ⅱ-国有大型银行Ⅲ", "LISTING_DATE": "1991-04-03 00:00:00"}
    rep = ingestor(env, Bank()).ensure("000001.SZ", datetime(2025, 12, 31, 15), "hybrid_live")
    with get_session_factory(env[0])() as s:
        assert s.get(SecurityRow, "000001.SZ").template == "bank"
    assert any("template=bank" in w for w in rep.warnings)


def test_end_to_end_research_run_on_a_company_the_repo_never_had(env):
    from agent_runtime import ResearchRequest, run_research

    set_provider_factory(lambda: FakeProvider())
    req = ResearchRequest(ticker="中际旭创", as_of=datetime(2025, 12, 31, 15), data_mode="hybrid_live", mode="fast",
                          question="中际旭创最新一期累计营业收入是多少亿元？")
    rep = run_research(req, "B4", db_url=env[0], persist=False)
    assert rep["ticker"] == "300308.SZ" and rep["data_mode"] == "hybrid_live"
    assert rep["live_ingestion"]["status"] in ("fetched", "partial")
    assert rep["claims"] and all(e["available_at"] <= rep["as_of"] for e in rep["evidence"].values())
    again = run_research(req, "B4", db_url=env[0], persist=False)
    assert again["live_ingestion"]["status"] == "hit"


@pytest.mark.skipif(not os.environ.get("RUN_LIVE_TESTS"), reason="set RUN_LIVE_TESTS=1 to call the real endpoints")
@pytest.mark.parametrize("ticker", ["300308.SZ", "601899.SH"])
def test_real_endpoints_for_companies_outside_the_snapshot(env, ticker):
    cat = AShareCatalog.load(refresh=True)
    set_catalog(cat)
    ing = ReadThroughIngestor(env[0], catalog=cat, doc_dir=env[1] / "docs")
    rep = ing.ensure(ticker, datetime(2025, 12, 31, 15), "hybrid_live")
    assert rep.status in ("fetched", "partial"), rep.as_dict()
    with get_session_factory(env[0])() as s:
        facts = list(s.scalars(select(FinancialFactRow).where(FinancialFactRow.ticker == ticker)))
        n_px = len(list(s.scalars(select(MarketPriceRow.trade_date).where(MarketPriceRow.ticker == ticker))))
    assert len(facts) > 100 and n_px > 500
    assert all(f.published_at.date() > f.period_end and f.available_at > f.published_at for f in facts)
    svc = PITDataService(datetime(2025, 12, 31, 15), "live", env[0])
    assert svc.fact(ticker, "operating_revenue", date(2025, 9, 30)) is not None
    assert svc.fact(ticker, "operating_revenue", date(2025, 12, 31)) is None     # annual report not yet public
