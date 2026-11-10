"""Live provider adapters for real A-share data (network required).

Eastmoney F10 datacenter: consolidated income / balance / cash-flow statements, PIT key NOTICE_DATE.
cninfo (巨潮资讯): announcement list and original PDFs, PIT key announcementTime (millisecond timestamp).
Yahoo Finance chart API: unadjusted daily OHLCV, available at 15:00 on the trade date.

Every raw response is cached under data/raw_cache/ so a snapshot can be rebuilt offline and
the build is reproducible. Baostock was evaluated but its login hung from this network.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

from .db import ROOT

CACHE = ROOT / "data" / "raw_cache"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FinResearchAgent/0.1 research"}
EM_URL = "https://datacenter.eastmoney.com/securities/api/data/get"
EM_REPORTS = {"income": ("RPT_F10_FINANCE_GINCOME", "APP_F10_GINCOME"),
              "balance": ("RPT_F10_FINANCE_GBALANCE", "F10_FINANCE_GBALANCE"),
              "cashflow": ("RPT_F10_FINANCE_GCASHFLOW", "APP_F10_GCASHFLOW")}


class HttpCache:
    def __init__(self, min_interval: float = 0.6):
        self.client = httpx.Client(headers=UA, timeout=30, follow_redirects=True)
        self.min_interval = min_interval
        self._last = 0.0

    def _path(self, key: str, ext: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()[:24]
        return CACHE / ext / f"{h}.{ext}"

    def _throttle(self) -> None:
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def request(self, method: str, url: str, *, params=None, data=None, binary: bool = False, attempts: int = 5):
        key = json.dumps([method, url, params, data], sort_keys=True, ensure_ascii=False)
        p = self._path(key, "bin" if binary else "json")
        if p.exists():
            return p.read_bytes() if binary else json.loads(p.read_text(encoding="utf-8"))
        last: Optional[Exception] = None
        for i in range(attempts):
            try:
                self._throttle()
                r = self.client.request(method, url, params=params, data=data)
                r.raise_for_status()
                p.parent.mkdir(parents=True, exist_ok=True)
                if binary:
                    p.write_bytes(r.content)
                    return r.content
                body = r.json()
                p.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
                return body
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
                time.sleep(1.5 * 2 ** i)
        raise RuntimeError(f"{url} failed after {attempts} attempts: {last}")


class EastmoneyStatements:
    def __init__(self, http: HttpCache):
        self.http = http

    def fetch(self, secucode: str, kind: str) -> list[dict[str, Any]]:
        rtype, sty = EM_REPORTS[kind]
        rows, page = [], 1
        while True:
            body = self.http.request("GET", EM_URL, params={
                "type": rtype, "sty": sty, "filter": f'(SECUCODE="{secucode}")', "p": page, "ps": 50,
                "sr": -1, "st": "REPORT_DATE", "source": "HSF10", "client": "PC"})
            res = body.get("result") or {}
            rows += res.get("data") or []
            if page >= (res.get("pages") or 1):
                return rows
            page += 1


class Cninfo:
    QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    STATIC = "http://static.cninfo.com.cn/"

    def __init__(self, http: HttpCache):
        self.http = http
        self._orgs: dict[str, str] = {}

    def org_id(self, code: str, market: str) -> str:
        if not self._orgs:
            js = self.http.request("GET", "http://www.cninfo.com.cn/new/data/szse_stock.json")
            for s in js.get("stockList", []):
                self._orgs[s["code"]] = s["orgId"]
        if code not in self._orgs:  # fallback: site search
            hits = self.http.request("POST", "http://www.cninfo.com.cn/new/information/topSearch/query",
                                     data={"keyWord": code, "maxNum": 5})
            for h in hits or []:
                if h.get("code") == code:
                    self._orgs[code] = h["orgId"]
        return self._orgs[code]

    def announcements(self, code: str, market: str, start: str, end: str, category: str = "") -> list[dict]:
        column = "sse" if market == "SH" else "szse"
        out, page = [], 1
        while True:
            body = self.http.request("POST", self.QUERY, data={
                "stock": f"{code},{self.org_id(code, market)}", "tabName": "fulltext", "pageSize": 30, "pageNum": page,
                "column": column, "category": category, "seDate": f"{start}~{end}", "isHLtitle": "true",
                "searchkey": "", "plate": ""})
            anns = body.get("announcements") or []
            out += anns
            if not body.get("hasMore") or not anns or page > 40:
                return out
            page += 1

    def pdf(self, adjunct_url: str) -> bytes:
        return self.http.request("GET", self.STATIC + adjunct_url, binary=True)

    @staticmethod
    def published_at(ann: dict) -> datetime:
        # announcementTime is epoch ms in UTC; convert to Beijing local wall time (UTC+8), naive
        ts = datetime.fromtimestamp(ann["announcementTime"] / 1000, tz=timezone.utc)
        return (ts.replace(tzinfo=None) + (datetime(2000, 1, 1, 8) - datetime(2000, 1, 1)))


class YahooPrices:
    URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"

    def __init__(self, http: HttpCache):
        self.http = http

    def daily(self, symbol: str, start: datetime, end: datetime) -> list[dict]:
        body = self.http.request("GET", self.URL.format(sym=symbol), params={
            "period1": int(start.replace(tzinfo=timezone.utc).timestamp()),
            "period2": int(end.replace(tzinfo=timezone.utc).timestamp()), "interval": "1d", "events": "div,splits"})
        res = body["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        out = []
        for i, ts in enumerate(res.get("timestamp") or []):
            if q["close"][i] is None:
                continue
            d = (datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
                 + (datetime(2000, 1, 1, 8) - datetime(2000, 1, 1))).date()
            out.append({"date": d, "open": q["open"][i], "high": q["high"][i], "low": q["low"][i],
                        "close": q["close"][i], "volume": q["volume"][i] or 0})
        return out
