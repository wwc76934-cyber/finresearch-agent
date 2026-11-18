"""Whole-market A-share security dictionary (about 6,200 listings) for code / name / pinyin-initials lookup.

Source: cninfo's public stock list (`szse_stock.json`, which holds SH, SZ and BJ listings). It is cached in
`data/a_share_catalog.json`; a cache older than 7 days is refreshed when the network allows and used as is when
it does not. Industry is not in this list: it is read from Eastmoney F10 when a company is first ingested.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import httpx

from .db import ROOT

CATALOG_PATH = ROOT / "data" / "a_share_catalog.json"
CNINFO_LIST = "http://www.cninfo.com.cn/new/data/szse_stock.json"
MAX_AGE_S = 7 * 86400


def market_of(code: str) -> str:
    if code.startswith(("6", "900")):
        return "SH"
    if code.startswith(("4", "8", "92")):
        return "BJ"
    return "SZ"        # 000/001/002/003/300/301 and 200 (B shares)


@dataclass(frozen=True)
class CatalogEntry:
    code: str
    name: str
    pinyin: str
    org_id: str
    category: str = "A股"

    @property
    def market(self) -> str:
        return market_of(self.code)

    @property
    def ticker(self) -> str:
        return f"{self.code}.{self.market}"


class AShareCatalog:
    def __init__(self, entries: list[CatalogEntry]):
        self.entries = entries
        self.by_code = {e.code: e for e in entries}
        self._by_pinyin: dict[str, list[CatalogEntry]] = {}
        for e in entries:
            self._by_pinyin.setdefault(e.pinyin.lower(), []).append(e)
        self._names = sorted(entries, key=lambda e: -len(e.name))

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------ lookup
    def get(self, ticker_or_code: str) -> Optional[CatalogEntry]:
        m = re.match(r"^\s*(\d{6})(?:\.(?:SZ|SH|BJ))?\s*$", ticker_or_code, re.I)
        return self.by_code.get(m.group(1)) if m else None

    def resolve(self, query: str) -> Optional[CatalogEntry]:
        """One security from a code (`600519`, `600519.SH`), a Chinese short name (exact, then contained in the
        text) or pinyin initials (`gzmt`). Ambiguous initials return None."""
        q = query.strip()
        hit = self.get(q)
        if hit:
            return hit
        found = self.extract(q, limit=1)
        if found:
            return found[0]
        if re.fullmatch(r"[A-Za-z]{3,8}", q):
            cands = self._by_pinyin.get(q.lower(), [])
            if len(cands) == 1:
                return cands[0]
        return None

    def extract(self, text: str, limit: int = 4) -> list[CatalogEntry]:
        """Securities mentioned in free text, in order of appearance. 6-digit codes always count; names need at
        least 3 characters (2-character names such as "中信" collide with ordinary words)."""
        hits: dict[str, tuple[int, CatalogEntry]] = {}
        for m in re.finditer(r"(?<!\d)(\d{6})(?:\.(?:SZ|SH|BJ))?(?!\d)", text, re.I):
            e = self.by_code.get(m.group(1))
            if e and e.code not in hits:
                hits[e.code] = (m.start(), e)
        taken: list[tuple[int, int]] = []
        for e in self._names:
            if len(e.name) < 3 or e.code in hits:
                continue
            i = text.find(e.name)
            if i >= 0 and not any(a <= i < b for a, b in taken):
                hits[e.code] = (i, e)
                taken.append((i, i + len(e.name)))
        return [e for _, e in sorted(hits.values(), key=lambda x: x[0])][:limit]

    def search(self, q: str, limit: int = 8) -> list[CatalogEntry]:
        """Type-ahead: code prefix, name containing the text, or pinyin-initials prefix. Exact matches first."""
        q = q.strip()
        if not q:
            return []
        ql = q.lower()
        scored: list[tuple[int, CatalogEntry]] = []
        for e in self.entries:
            if e.code == q or e.name == q:
                score = 0
            elif e.code.startswith(q) or e.name.startswith(q):
                score = 1
            elif q in e.name:
                score = 2
            elif e.pinyin.lower().startswith(ql) and ql.isalpha():
                score = 3
            else:
                continue
            scored.append((score, e))
        scored.sort(key=lambda x: (x[0], x[1].code))
        return [e for _, e in scored[:limit]]

    # ------------------------------------------------------------ persistence
    @classmethod
    def from_stock_list(cls, stock_list: list[dict]) -> "AShareCatalog":
        return cls([CatalogEntry(code=s["code"], name=s["zwjc"], pinyin=s.get("pinyin", ""), org_id=s.get("orgId", ""),
                                 category=s.get("category", "A股"))
                    for s in stock_list if re.fullmatch(r"\d{6}", s.get("code", "")) and s.get("category") != "CDR"])

    def save(self, path: Path = CATALOG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": time.time(), "entries": [asdict(e) for e in self.entries]},
                                   ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CATALOG_PATH, refresh: bool = True, timeout: float = 8.0) -> "AShareCatalog":
        cached = None
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            cached = (raw.get("fetched_at", 0), cls([CatalogEntry(**e) for e in raw["entries"]]))
            if not refresh or time.time() - cached[0] < MAX_AGE_S:
                return cached[1]
        try:
            r = httpx.get(CNINFO_LIST, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True)
            r.raise_for_status()
            cat = cls.from_stock_list(r.json()["stockList"])
            if len(cat) < 1000:
                raise ValueError(f"catalog suspiciously small: {len(cat)}")
            cat.save(path)
            return cat
        except (httpx.HTTPError, ValueError, KeyError):
            if cached:
                return cached[1]
            raise


_CATALOG: Optional[AShareCatalog] = None


def get_catalog() -> Optional[AShareCatalog]:
    """The process-wide catalog, or None when there is no cache and no network (callers fall back to the local DB)."""
    global _CATALOG
    if _CATALOG is None:
        try:
            _CATALOG = AShareCatalog.load()
        except Exception:  # noqa: BLE001 - offline and never cached
            return None
    return _CATALOG


def cached_catalog() -> Optional[AShareCatalog]:
    """The catalog if it is already in memory or in the cache file; never touches the network (for the first byte of a
    stream and for type-ahead search)."""
    global _CATALOG
    if _CATALOG is None and CATALOG_PATH.exists():
        try:
            _CATALOG = AShareCatalog.load(refresh=False)
        except (ValueError, KeyError, OSError):
            return None
    return _CATALOG


def set_catalog(cat: Optional[AShareCatalog]) -> None:
    """Tests inject a small catalog here."""
    global _CATALOG
    _CATALOG = cat
