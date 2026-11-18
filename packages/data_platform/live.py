"""Read-Through ingestion: fetch a company that is not in the local database (or whose local data is too old for the
requested as_of) from public sources, store it with PIT timestamps, and remember what is now covered.

Sources that work from this project's network: Eastmoney F10 datacenter (statements, NOTICE_DATE, industry),
cninfo (announcements, orgId from the catalog), Yahoo (daily bars). AKShare and Eastmoney's push2 are blocked here.

PIT rules
* published_at of a statement period = Eastmoney NOTICE_DATE (the first disclosure date). The period end is never
  used as a publication time. When NOTICE_DATE is missing the statutory deadline is used and the fact is marked
  `pit_estimated`.
* available_at = 09:30 of the first trading day AFTER NOTICE_DATE (the vendor gives a date, not a time, so the
  same day is not assumed).
* announcements have a real timestamp: usable from the next 09:30 if published after 09:30.
* Known limit: Eastmoney returns the vendor's latest value for each past period. If a company later restated a
  period, an earlier as_of can see the restated number. For the 8 core companies this was cross-checked against
  filing PDFs; for live companies it is not, and the ingestion report says so.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time as _time
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
from sqlalchemy import delete, select

from .ashare_catalog import AShareCatalog, CatalogEntry, get_catalog
from .db import (ROOT, DocumentChunkRow, FinancialFactRow, IngestionWatermarkRow, MarketPriceRow, SecurityRow,
                 SourceDocumentRow, get_session_factory)
from .em_mapping import CUM, flows, instants

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FinResearchAgent/2.0 research"}
EM_URL = "https://datacenter.eastmoney.com/securities/api/data/get"
CNINFO_QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC = "http://static.cninfo.com.cn/"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
EM_STATEMENTS = {"income": ("RPT_F10_FINANCE_GINCOME", "APP_F10_GINCOME"),
                 "balance": ("RPT_F10_FINANCE_GBALANCE", "F10_FINANCE_GBALANCE"),
                 "cashflow": ("RPT_F10_FINANCE_GCASHFLOW", "APP_F10_GCASHFLOW")}
LIVE_SOURCE = "eastmoney_f10_live"
LIVE_DOC_DIR = ROOT / "data" / "live_docs"
KEY_WORDS = ("合同", "关联交易", "回购", "担保", "处罚", "立案", "减值", "更正", "业绩预告", "业绩快报", "债券", "收购", "诉讼",
             "利润分配", "募集", "重大")
PERIODIC = re.compile(r"(年度报告|半年度报告|第一季度报告|第三季度报告|年报|半年报|一季报|三季报)")
DEADLINE = {3: (4, 30), 6: (8, 31), 9: (10, 31), 12: (4, 30)}   # statutory disclosure deadlines (annual: next year)
PERIODIC_CATEGORY = {3: "category_yjdbg_szsh;", 6: "category_bndbg_szsh;", 9: "category_sjdbg_szsh;", 12: "category_ndbg_szsh;"}
PERIODIC_DOC = {3: "q1_report", 6: "semi_annual_report", 9: "q3_report", 12: "annual_report"}
SKIP_TITLE = ("摘要", "英文", "取消", "提示性")
CONSUMER_L1 = {"食品饮料", "家电", "商贸零售", "社会服务", "美容护理", "纺织服饰", "农林牧渔", "轻工制造", "休闲服务"}
BEIJING = timedelta(hours=8)


# ----------------------------------------------------------------------------- providers
class LiveMarketProvider:
    """Thin HTTP client for the three public sources. Short timeouts, one retry, no disk cache (live data)."""

    def __init__(self, timeout: float = 4.0, client: Optional[httpx.Client] = None):
        self.client = client or httpx.Client(headers=UA, timeout=timeout, follow_redirects=True)

    def _json(self, method: str, url: str, **kw) -> Any:
        last: Optional[Exception] = None
        for i in range(2):
            try:
                r = self.client.request(method, url, **kw)
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
                _time.sleep(0.4 * (i + 1))
        raise RuntimeError(f"{url}: {last}")

    def statements(self, secucode: str, periods: int = 24, prefix: str = "G") -> dict[str, list[dict]]:
        """prefix: G general companies, B banks, I insurers, S securities firms (Eastmoney's own report families)."""
        out: dict[str, list[dict]] = {}
        for kind, (rtype, sty) in EM_STATEMENTS.items():
            if prefix != "G":
                rtype, sty = rtype.replace("_G", f"_{prefix}"), "ALL"
            body = self._json("GET", EM_URL, params={"type": rtype, "sty": sty, "filter": f'(SECUCODE="{secucode}")',
                                                    "p": 1, "ps": periods, "sr": -1, "st": "REPORT_DATE",
                                                    "source": "HSF10", "client": "PC"})
            out[kind] = (body.get("result") or {}).get("data") or []
        return out

    def org_info(self, secucode: str) -> dict:
        body = self._json("GET", EM_URL, params={"type": "RPT_F10_BASIC_ORGINFO", "sty": "ALL",
                                                "filter": f'(SECUCODE="{secucode}")', "client": "PC", "source": "HSF10"})
        rows = (body.get("result") or {}).get("data") or []
        return rows[0] if rows else {}

    def prices(self, symbol: str, start: date, end: date) -> list[dict]:
        body = self._json("GET", YAHOO.format(sym=symbol), params={
            "period1": int(datetime.combine(start, time()).replace(tzinfo=timezone.utc).timestamp()),
            "period2": int(datetime.combine(end, time()).replace(tzinfo=timezone.utc).timestamp()), "interval": "1d"})
        res = body["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        out: list[dict] = []
        for i, ts in enumerate(res.get("timestamp") or []):
            if q["close"][i] is None:
                continue
            d = (datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None) + BEIJING).date()
            out.append({"date": d, "open": q["open"][i], "high": q["high"][i], "low": q["low"][i], "close": q["close"][i],
                        "volume": q["volume"][i] or 0})
        return out

    def announcements(self, entry: CatalogEntry, start: date, end: date, max_pages: int = 3) -> list[dict]:
        out: list[dict] = []
        for page in range(1, max_pages + 1):
            body = self._json("POST", CNINFO_QUERY, data={
                "stock": f"{entry.code},{entry.org_id}", "tabName": "fulltext", "pageSize": 30, "pageNum": page,
                "column": "sse" if entry.market == "SH" else "szse", "category": "", "plate": "", "searchkey": "",
                "seDate": f"{start}~{end}", "isHLtitle": "true"})
            anns = body.get("announcements") or []
            out += anns
            if not body.get("hasMore") or not anns:
                break
        return out

    def periodic_announcements(self, entry: CatalogEntry, start: date, end: date, category: str) -> list[dict]:
        out: list[dict] = []
        for page in (1, 2):
            body = self._json("POST", CNINFO_QUERY, data={
                "stock": f"{entry.code},{entry.org_id}", "tabName": "fulltext", "pageSize": 30, "pageNum": page,
                "column": "sse" if entry.market == "SH" else "szse", "category": category, "plate": "", "searchkey": "",
                "seDate": f"{start}~{end}", "isHLtitle": "true"})
            anns = body.get("announcements") or []
            out += anns
            if not body.get("hasMore") or not anns:
                break
        return out

    def pdf_bytes(self, ann: dict) -> bytes:
        r = self.client.get(CNINFO_STATIC + ann["adjunctUrl"], timeout=40)
        r.raise_for_status()
        return r.content

    def announcement_text(self, ann: dict, pages: int = 4, limit: int = 3000) -> str:
        r = self.client.get(CNINFO_STATIC + ann["adjunctUrl"])
        r.raise_for_status()
        import pymupdf

        doc = pymupdf.open(stream=r.content, filetype="pdf")
        try:
            text = "".join(doc[i].get_text("text") for i in range(min(doc.page_count, pages)))
        finally:
            doc.close()
        return "".join(l.strip() for l in text.splitlines() if l.strip())[:limit]


# ----------------------------------------------------------------------------- AKShare cache
AKSHARE_DIR = ROOT / "data" / "akshare_cache"


class AkshareCache:
    """Data that only AKShare can supply from the user's network (industry members with market cap, prices for markets
    Yahoo does not cover). `scripts/akshare_collect.py` writes it; nothing here calls AKShare."""

    def __init__(self, root: Path = AKSHARE_DIR):
        self.root = root
        self._members: Optional[dict] = None

    def _load_members(self) -> dict:
        if self._members is None:
            f = self.root / "industry_members.json"
            self._members = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        return self._members

    def has_industry(self) -> bool:
        return bool(self._load_members().get("boards"))

    def board_of(self, code: str) -> Optional[str]:
        return (self._load_members().get("by_code") or {}).get(code)

    def peers(self, code: str, n: int = 5) -> list[str]:
        """Codes of the n largest (by market cap) other members of the company's industry board."""
        m = self._load_members()
        board = self.board_of(code)
        if not board:
            return []
        rows = [r for r in m["boards"].get(board, []) if r["code"] != code and r.get("mktcap")]
        rows.sort(key=lambda r: -float(r["mktcap"]))
        return [r["code"] for r in rows[:n]]

    def prices(self, code: str) -> list[dict]:
        f = self.root / f"prices_{code}.json"
        if not f.exists():
            return []
        return [{**r, "date": date.fromisoformat(r["date"])} for r in json.loads(f.read_text(encoding="utf-8"))]


# ----------------------------------------------------------------------------- helpers
class TradingCalendar:
    def __init__(self, days: list[date]):
        self.days = sorted(set(days))

    def next_open_after(self, d: date) -> datetime:
        """09:30 of the first trading day strictly after d."""
        i = bisect_right(self.days, d)
        if i < len(self.days):
            return datetime.combine(self.days[i], time(9, 30))
        nxt = d + timedelta(days=1)
        while nxt.weekday() >= 5:      # no bars that late (or none at all): next weekday
            nxt += timedelta(days=1)
        return datetime.combine(nxt, time(9, 30))

    def open_at_or_after(self, ts: datetime) -> datetime:
        d = ts.date() + (timedelta(days=1) if ts.time() > time(9, 30) else timedelta())
        i = bisect_right(self.days, d - timedelta(days=1))
        if i < len(self.days):
            return datetime.combine(self.days[i], time(9, 30))
        while d.weekday() >= 5:
            d += timedelta(days=1)
        return datetime.combine(d, time(9, 30))


def statutory_deadline(period_end: date) -> datetime:
    m, d = DEADLINE[period_end.month]
    return datetime(period_end.year + (1 if period_end.month == 12 else 0), m, d)


def template_of(l1: str, l2: str, l3: str = "") -> str:
    """Eastmoney's EM2016 industry path, e.g. 金融-银行-国有银行, 金融-非银行金融-证券, 食品饮料-饮料-白酒."""
    if l1 == "金融" and l2 == "银行":
        return "bank"
    if l1 == "金融" and "保险" in (l2 + l3):
        return "insurance"
    if l1 == "金融" and "证券" in (l2 + l3):
        return "broker"
    return "consumer" if l1 in CONSUMER_L1 else "manufacturing"


def parse_date(v: Any) -> Optional[date]:
    if not v:
        return None
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def yahoo_symbol(entry: CatalogEntry) -> Optional[str]:
    return {"SH": f"{entry.code}.SS", "SZ": f"{entry.code}.SZ"}.get(entry.market)


@dataclass
class IngestReport:
    ticker: str
    status: str                       # hit | fetched | partial | failed | skipped_offline | unknown_security
    counts: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    messages: list[str] = field(default_factory=list)
    degraded: int = 0                 # sources that failed or fell back (drives status=partial)

    def as_dict(self) -> dict:
        return {"ticker": self.ticker, "status": self.status, "counts": self.counts, "warnings": self.warnings,
                "elapsed_s": round(self.elapsed_s, 2)}


_PROVIDER_FACTORY: Optional[Callable[[], LiveMarketProvider]] = None


def set_provider_factory(factory: Optional[Callable[[], LiveMarketProvider]]) -> None:
    """Tests inject a provider that never touches the network."""
    global _PROVIDER_FACTORY
    _PROVIDER_FACTORY = factory


def prepare_live(query: str, as_of: datetime, mode: str, db_url: Optional[str] = None,
                 listener: Optional[Callable[[dict], None]] = None) -> Optional[IngestReport]:
    """Make sure the company named by `query` (code, name or pinyin initials) is in the local database as of
    `as_of`. Returns None when the mode is offline or the query is not in the catalog."""
    if mode == "offline_golden":
        return None
    cat = get_catalog()
    entry = cat.resolve(query) if cat else None
    if entry is None:
        return None
    return make_ingestor(db_url, cat).ensure(entry.ticker, as_of, mode, listener)


def make_ingestor(db_url: Optional[str] = None, catalog: Optional[AShareCatalog] = None) -> "ReadThroughIngestor":
    provider = _PROVIDER_FACTORY() if _PROVIDER_FACTORY else None
    return ReadThroughIngestor(db_url, provider=provider, catalog=catalog or get_catalog())


# ----------------------------------------------------------------------------- the ingestor
class ReadThroughIngestor:
    _locks: dict[str, threading.Lock] = {}
    _guard = threading.Lock()

    def __init__(self, db_url: Optional[str] = None, provider: Optional[LiveMarketProvider] = None,
                 catalog: Optional[AShareCatalog] = None, doc_dir: Path = LIVE_DOC_DIR,
                 clock: Callable[[], datetime] = datetime.now, akshare: Optional[AkshareCache] = None):
        self.db_url = db_url
        self.Session = get_session_factory(db_url)
        self.provider = provider or LiveMarketProvider()
        self._catalog = catalog
        self.doc_dir = doc_dir
        self.clock = clock
        self.akshare = akshare or AkshareCache()

    @property
    def catalog(self) -> Optional[AShareCatalog]:
        return self._catalog or get_catalog()

    # ------------------------------------------------------------ coverage
    def _seed_watermark(self, ticker: str) -> Optional[IngestionWatermarkRow]:
        """A company loaded from a snapshot has no watermark yet: derive it from what is stored."""
        with self.Session() as s:
            facts = s.execute(select(FinancialFactRow.available_at).where(FinancialFactRow.ticker == ticker)).scalars().all()
            bars = s.execute(select(MarketPriceRow.trade_date).where(MarketPriceRow.ticker == ticker)).scalars().all()
            if not facts and not bars:
                return None
            lo = min([f.date() for f in facts] + list(bars))
            hi = max([f.date() for f in facts] + list(bars))
            row = IngestionWatermarkRow(ticker=ticker, covered_from=lo, max_as_of_date=hi, updated_at=self.clock(),
                                        source_status="seeded", details={"origin": "local snapshot"})
            s.merge(row)
            s.commit()
            return row

    def coverage(self, ticker: str) -> Optional[IngestionWatermarkRow]:
        with self.Session() as s:
            row = s.get(IngestionWatermarkRow, ticker)
        return row or self._seed_watermark(ticker)

    def needs_fetch(self, ticker: str, as_of: datetime, mode: str) -> bool:
        if mode == "online_force_refresh":
            return True
        wm = self.coverage(ticker)
        if wm is None:
            return True
        day = min(as_of.date(), self.clock().date())     # nothing newer than today exists to fetch
        return not (wm.covered_from <= day <= wm.max_as_of_date)

    # ------------------------------------------------------------ entry point
    def ensure(self, ticker: str, as_of: datetime, mode: str = "hybrid_live",
               listener: Optional[Callable[[dict], None]] = None, with_peers: bool = True) -> IngestReport:
        if mode == "offline_golden":
            return IngestReport(ticker, "skipped_offline")
        entry = self.catalog.get(ticker) if self.catalog else None
        with self._guard:
            lock = self._locks.setdefault(ticker, threading.Lock())
        with lock:
            if not self.needs_fetch(ticker, as_of, mode):
                return IngestReport(ticker, "hit")
            if entry is None:
                return IngestReport(ticker, "unknown_security", warnings=["not in the A-share catalog and not stored locally"])
            return self._fetch(entry, as_of, listener or (lambda ev: None), with_peers=with_peers)

    # ------------------------------------------------------------ fetch + store
    def _fetch(self, entry: CatalogEntry, as_of: datetime, emit: Callable[[dict], None], lite: bool = False,
               with_peers: bool = True) -> IngestReport:
        """lite: statements, industry and prices only (used for peers; no announcements, no peers of peers)."""
        t0 = _time.perf_counter()
        t = entry.ticker
        rep = IngestReport(t, "fetched")
        emit({"type": "live_ingestion_status", "ticker": t,
              "message": f"正在从东方财富/巨潮资讯同步 {t} {entry.name} 的历史财报与 PIT 时间戳..."})
        end = min(as_of.date(), self.clock().date())

        def result(fut, name, default):
            if fut is None:
                rep.degraded += 1
                rep.warnings.append(f"{name}: no source for {entry.market}")
                return default
            try:
                return fut.result()
            except Exception as exc:  # noqa: BLE001 - one source down must not stop the run
                rep.degraded += 1
                rep.warnings.append(f"{name}: {type(exc).__name__}: {str(exc)[:120]}")
                return default
        with ThreadPoolExecutor(max_workers=5) as ex:
            f_org = ex.submit(self.provider.org_info, t)
            sym = yahoo_symbol(entry)
            f_px = ex.submit(self.provider.prices, sym, end - timedelta(days=3 * 365 + 10), end + timedelta(days=1)) if sym else None
            f_an = None if lite else ex.submit(self.provider.announcements, entry, end - timedelta(days=548), end)
            org = result(f_org, "org_info", {})
            parts = [x for x in str(org.get("EM2016") or "").split("-") if x]
            template = template_of(*(parts + ["", "", ""])[:3]) if parts else "manufacturing"
            prefix = {"bank": "B", "insurance": "I", "broker": "S"}.get(template, "G")
            stmts = result(ex.submit(self.provider.statements, t, 24, prefix), "statements", {})
            bars = result(f_px, "prices", []) if sym else []
            anns = [] if lite else result(f_an, "announcements", [])
        if not bars:       # Yahoo has no Beijing listings and can be down: fall back to the AKShare cache
            bars = [b for b in self.akshare.prices(entry.code) if b["date"] <= end]
            if bars:
                rep.warnings.append("prices: from the AKShare cache")
            elif entry.market == "BJ" or sym is None:
                rep.warnings.append("prices: run `python scripts/akshare_collect.py prices --tickers %s` to add them" % t)
        cal = TradingCalendar([b["date"] for b in bars])
        sec = self._store_security(entry, org)
        rep.counts["facts"] = self._store_facts(t, stmts, cal, rep)
        rep.counts["prices"] = self._store_prices(t, bars, stmts)
        emit({"type": "live_ingestion_status", "ticker": t, "message": f"{t}: 财报 {rep.counts['facts']} 条、行情 {rep.counts['prices']} 条已入库"})
        rep.counts["documents"] = 0 if lite else self._store_documents(entry, anns, cal, rep)
        rep.counts["industry"] = f"{sec.industry_sw_l1}/{sec.industry_sw_l2}"
        if sec.template in ("bank", "insurance", "broker"):
            rep.warnings.append(f"template={sec.template}: statements come from the {prefix}-family reports; gross margin, "
                                "inventory and receivable-days style ratios do not apply and stay empty")
        rep.warnings.append("statement values are the vendor's latest per period and were not cross-checked against the "
                            "filing PDF unless a periodic report was parsed (see details.restatement_checks)")
        got = rep.counts["facts"] or rep.counts["prices"] or rep.counts["documents"]
        rep.status = ("partial" if rep.degraded else "fetched") if got else "failed"
        self._set_watermark(t, as_of, end, bars, rep)
        if with_peers and not lite:
            rep.counts["peers"] = self._ensure_peers(entry, as_of, emit, rep)
        rep.elapsed_s = _time.perf_counter() - t0
        emit({"type": "live_ingestion_status", "ticker": t, "status": rep.status, "counts": rep.counts,
              "message": f"{t} 同步完成（{rep.elapsed_s:.1f}s，{rep.status}）"})
        return rep

    def _ensure_peers(self, entry: CatalogEntry, as_of: datetime, emit: Callable[[dict], None], rep: IngestReport,
                      n: int = 5) -> list[str]:
        """The n largest companies of the same industry board, fetched in lite mode so peer valuation has data.
        The member list needs AKShare (scripts/akshare_collect.py industry); without it only locally stored
        companies of the same industry can be peers."""
        if not self.akshare.has_industry():
            rep.warnings.append("peers: no industry member list; run `python scripts/akshare_collect.py industry` "
                                "(peers fall back to companies already stored locally)")
            return []
        cat = self.catalog
        codes = self.akshare.peers(entry.code, n)
        peers = [cat.by_code[c] for c in codes if cat and c in cat.by_code]
        todo = [p for p in peers if self.needs_fetch(p.ticker, as_of, "hybrid_live")]
        if todo:
            emit({"type": "live_ingestion_status", "ticker": entry.ticker,
                  "message": f"正在同步同业可比公司：{'、'.join(p.name for p in todo)}"})
            with ThreadPoolExecutor(max_workers=min(len(todo), 5)) as ex:
                list(ex.map(lambda p: self._fetch(p, as_of, lambda ev: None, lite=True, with_peers=False), todo))
        return [p.ticker for p in peers]

    # ------------------------------------------------------------ Tier-B: a whole periodic report, on demand
    def _calendar(self, ticker: str) -> TradingCalendar:
        with self.Session() as s:
            days = list(s.execute(select(MarketPriceRow.trade_date).where(MarketPriceRow.ticker == ticker)).scalars())
        return TradingCalendar(days)

    def ensure_periodic_report(self, ticker: str, period_end: date, as_of: datetime,
                               emit: Optional[Callable[[dict], None]] = None) -> dict:
        """Download and parse the periodic report of one period (all notes become searchable), store it, and compare the
        statement totals with the vendor figures already stored. Slow (tens of seconds): only called when a sub-graph
        needs a note it cannot find."""
        from rag_engine.chunking import chunk_document
        from rag_engine.index import get_retriever
        from rag_engine.pdf_parser import parse_pdf, statement_value

        emit = emit or (lambda ev: None)
        entry = self.catalog.get(ticker) if self.catalog else None
        if entry is None:
            return {"status": "unknown_security"}
        did = f"doc_{entry.code}_{period_end.year}Q{(period_end.month - 1) // 3 + 1}"
        with self.Session() as s:
            have = s.get(SourceDocumentRow, did)
            if have is not None and have.available_at <= as_of:
                return {"status": "cached", "document_id": did}
        emit({"type": "live_ingestion_status", "ticker": ticker, "message": f"正在下载并解析 {period_end} 定期报告全文（按需）..."})
        try:
            anns = self.provider.periodic_announcements(entry, period_end + timedelta(days=1),
                                                        min(as_of.date(), self.clock().date()), PERIODIC_CATEGORY[period_end.month])
            anns = [a for a in anns if str(period_end.year) in a["announcementTitle"]
                    and not any(w in a["announcementTitle"] for w in SKIP_TITLE)]
            if not anns:
                return {"status": "not_found", "document_id": did}
            ann = max(anns, key=lambda a: a["announcementTime"])
            md, stats = parse_pdf(self.provider.pdf_bytes(ann), f"{entry.name}（{ticker}）{ann['announcementTitle']}")
        except Exception as exc:  # noqa: BLE001 - network, download or parse failure: the caller records an open question
            return {"status": "failed", "document_id": did, "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        pub = datetime.fromtimestamp(ann["announcementTime"] / 1000, tz=timezone.utc).replace(tzinfo=None) + BEIJING
        avail = self._calendar(ticker).open_at_or_after(pub)
        self.doc_dir.mkdir(parents=True, exist_ok=True)
        (self.doc_dir / f"{did}.md").write_text(md, encoding="utf-8")
        checks = self._crosscheck(ticker, period_end, md, statement_value)
        with self.Session() as s:
            s.merge(SourceDocumentRow(
                document_id=did, ticker=ticker, title=f"{entry.name}{ann['announcementTitle']}", doc_type=PERIODIC_DOC[period_end.month],
                period_end=period_end, published_at=pub, available_at=avail, content_hash=hashlib.sha256(md.encode()).hexdigest(),
                path=f"../data/live_docs/{did}.md", version=1))
            s.execute(delete(DocumentChunkRow).where(DocumentChunkRow.document_id == did))
            for c in chunk_document(did, ticker, f"{period_end.year}Q{(period_end.month - 1) // 3 + 1}",
                                    PERIODIC_DOC[period_end.month], md, avail, anchors=True):
                s.add(DocumentChunkRow(**c.__dict__))
            wm = s.get(IngestionWatermarkRow, ticker)
            if wm is not None:
                details = dict(wm.details or {})
                details["restatement_checks"] = [c for c in details.get("restatement_checks", []) if c["period"] != str(period_end)] + checks
                wm.details = details
            s.commit()
        get_retriever.cache_clear()
        return {"status": "parsed", "document_id": did, "pages": stats.get("pages"), "tables": stats.get("tables"),
                "restatement_checks": checks}

    def _crosscheck(self, ticker: str, period_end: date, md: str, statement_value: Callable[..., Optional[float]]) -> list[dict]:
        rows = {"operating_revenue": ("合并利润表", (r"其中营业收入", r"营业收入")),
                "net_profit": ("合并利润表", (r"归属于母公司(股东|所有者)的净利润",))}
        out: list[dict] = []
        with self.Session() as s:
            for metric, (stmt, pats) in rows.items():
                f = s.scalars(select(FinancialFactRow).where(FinancialFactRow.ticker == ticker, FinancialFactRow.metric == metric,
                                                             FinancialFactRow.period_end == period_end,
                                                             FinancialFactRow.source == LIVE_SOURCE)).first()
                filing = statement_value(md, stmt, pats)
                if f is None or filing is None:
                    continue
                diff = abs(filing - f.value) / max(abs(f.value), 1.0)
                out.append({"period": str(period_end), "metric": metric, "vendor": f.value, "filing_pdf": filing,
                            "rel_diff": round(diff, 5), "restatement_suspect": diff > 0.003})
        return out

    def _store_security(self, entry: CatalogEntry, org: dict) -> SecurityRow:
        em = str(org.get("EM2016") or "")
        parts = [p for p in em.split("-") if p]
        l1, l2, l3 = (parts + ["未知", "未知", ""])[:3] if parts else ("未知", "未知", "")
        listed = parse_date(org.get("LISTING_DATE")) or date(2000, 1, 1)
        row = SecurityRow(ticker=entry.ticker, name=entry.name, list_date=listed, delisted=False, industry_sw_l1=l1,
                          industry_sw_l2=l2, template=template_of(l1, l2, l3), is_demo=False)
        with self.Session() as s:
            s.merge(row)
            s.commit()
        return row

    def _store_facts(self, ticker: str, stmts: dict, cal: TradingCalendar, rep: IngestReport) -> int:
        inc = {r["REPORT_DATE"][:10]: r for r in stmts.get("income", [])}
        bal = {r["REPORT_DATE"][:10]: r for r in stmts.get("balance", [])}
        cfl = {r["REPORT_DATE"][:10]: r for r in stmts.get("cashflow", [])}
        now = self.clock()
        with self.Session() as s:
            have = set(s.execute(select(FinancialFactRow.period_end).where(FinancialFactRow.ticker == ticker)).scalars())
            rows, estimated = [], 0
            for rd in sorted(set(inc) | set(bal)):
                pe = date.fromisoformat(rd)
                if pe.month not in CUM or pe in have:
                    continue           # keep what a better source already stored for that period
                i, b, c = inc.get(rd, {}), bal.get(rd, {}), cfl.get(rd, {})
                notice = parse_date((i or b or c).get("NOTICE_DATE"))
                est = notice is None or notice <= pe       # a notice date on/before period end is not a disclosure date
                pub = datetime.combine(notice, time()) if (notice and not est) else statutory_deadline(pe)
                estimated += est
                avail = cal.next_open_after(pub.date())
                values = {**{m: (CUM[pe.month], v) for m, v in flows(i, c).items()},
                          **{m: ("instant", v) for m, v in instants(b).items()}}
                for m, (ptype, v) in values.items():
                    rows.append(FinancialFactRow(
                        fact_id=f"fact_{ticker[:6]}_{pe:%Y%m%d}_{m}_con_live_v1", ticker=ticker, metric=m,
                        statement_scope="consolidated", period_type=ptype, value=float(v), unit="CNY", period_end=pe,
                        published_at=pub, available_at=avail, retrieved_at=now, source=LIVE_SOURCE,
                        source_document_id=None, revision_id=1, is_restated=False, pit_unsafe=False, pit_estimated=est))
            s.add_all(rows)
            s.commit()
        if estimated:
            rep.degraded += 1
            rep.warnings.append(f"{estimated} period(s) had no usable NOTICE_DATE: statutory deadline used (pit_estimated)")
        return len(rows)

    def _store_prices(self, ticker: str, bars: list[dict], stmts: dict) -> int:
        if not bars:
            return 0
        shares = sorted((parse_date(r.get("NOTICE_DATE")) or date.min, float(r["SHARE_CAPITAL"]))
                        for r in stmts.get("balance", []) if r.get("SHARE_CAPITAL"))
        first = shares[0][1] if shares else None
        out = 0
        with self.Session() as s:
            known = set(s.execute(select(MarketPriceRow.trade_date).where(MarketPriceRow.ticker == ticker)).scalars())
            for bar in bars:
                if bar["date"] in known:
                    continue
                vis = [v for d, v in shares if d <= bar["date"]]
                sh = vis[-1] if vis else first
                mv = round(bar["close"] * sh, 2) if sh else 0.0
                s.add(MarketPriceRow(
                    ticker=ticker, trade_date=bar["date"], adjust="none", open=round(bar["open"], 3), high=round(bar["high"], 3),
                    low=round(bar["low"], 3), close=round(bar["close"], 3), volume=float(bar["volume"]),
                    turnover_rate=round(bar["volume"] / sh * 100, 4) if sh else 0.0, total_mv=mv, float_mv=mv,
                    available_at=datetime.combine(bar["date"], time(15, 0))))
                out += 1
            s.commit()
        return out

    def _store_documents(self, entry: CatalogEntry, anns: list[dict], cal: TradingCalendar, rep: IngestReport,
                         max_docs: int = 6) -> int:
        """Tier-A: key (non-periodic) announcements, first pages only. Whole periodic reports are Tier-B (on demand)."""
        from rag_engine.chunking import chunk_document
        from rag_engine.index import get_retriever

        key = [a for a in anns if not PERIODIC.search(a["announcementTitle"])
               and any(k in a["announcementTitle"] for k in KEY_WORDS)]
        key.sort(key=lambda a: -a["announcementTime"])
        key = key[:max_docs]
        if not key:
            return 0
        self.doc_dir.mkdir(parents=True, exist_ok=True)

        def one(a: dict):
            try:
                return a, self.provider.announcement_text(a)
            except Exception as exc:  # noqa: BLE001
                rep.degraded += 1
                rep.warnings.append(f"announcement text {a['announcementId']}: {type(exc).__name__}")
                return a, None
        with ThreadPoolExecutor(max_workers=6) as ex:
            texts = list(ex.map(one, key))
        n = 0
        with self.Session() as s:
            for a, body in texts:
                if not body:
                    continue
                did = f"ann_{entry.code}_{a['announcementId']}"
                pub = datetime.fromtimestamp(a["announcementTime"] / 1000, tz=timezone.utc).replace(tzinfo=None) + BEIJING
                avail = cal.open_at_or_after(pub)
                md = f"# {a['announcementTitle']}\n\n<!-- page 1 -->\n## 公告正文\n{body}\n"
                (self.doc_dir / f"{did}.md").write_text(md, encoding="utf-8")
                s.merge(SourceDocumentRow(
                    document_id=did, ticker=entry.ticker, title=a["announcementTitle"], doc_type="announcement",
                    period_end=None, published_at=pub, available_at=avail,
                    content_hash=hashlib.sha256(md.encode()).hexdigest(), path=f"../data/live_docs/{did}.md", version=1))
                s.execute(delete(DocumentChunkRow).where(DocumentChunkRow.document_id == did))
                for c in chunk_document(did, entry.ticker, pub.date().isoformat(), "announcement", md, avail, anchors=True):
                    s.add(DocumentChunkRow(**c.__dict__))
                n += 1
            s.commit()
        if n:
            get_retriever.cache_clear()     # the in-memory index is rebuilt from the database on next use
        return n

    def _set_watermark(self, ticker: str, as_of: datetime, end: date, bars: list[dict], rep: IngestReport) -> None:
        with self.Session() as s:
            lows = [r for r in s.execute(select(FinancialFactRow.available_at).where(
                FinancialFactRow.ticker == ticker)).scalars()]
        lo = min([d.date() for d in lows] + [b["date"] for b in bars] + [as_of.date()])
        status = {"fetched": "ok", "partial": "partial", "failed": "failed"}.get(rep.status, "ok")
        with self.Session() as s:
            s.merge(IngestionWatermarkRow(ticker=ticker, covered_from=lo, max_as_of_date=max(end, self.clock().date()),
                                          updated_at=self.clock(), source_status=status,
                                          details={"counts": rep.counts, "warnings": rep.warnings}))
            s.commit()
