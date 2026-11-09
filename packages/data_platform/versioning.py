"""Data Versioning: immutable DataSnapshotManifest per research run."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import func, select

from .db import (ROOT, DataSnapshotRow, FinancialFactRow, MarketPriceRow, SourceDocumentRow, get_session_factory)
from .schema import SnapshotManifest

GOLDEN = ROOT / "benchmarks" / "golden_snapshot"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def create_snapshot(as_of: datetime, tickers: list[str], db_url: Optional[str] = None) -> SnapshotManifest:
    Session = get_session_factory(db_url)
    tables = {"financial_facts": FinancialFactRow, "market_prices": MarketPriceRow, "source_documents": SourceDocumentRow}
    watermarks, counts = {}, {}
    with Session() as s:
        for name, model in tables.items():
            cond = (model.available_at <= as_of) & (model.ticker.in_(tickers))
            mx = s.scalar(select(func.max(model.available_at)).where(cond))
            watermarks[name] = mx.isoformat() if mx else ""
            counts[name] = s.scalar(select(func.count()).select_from(model).where(cond)) or 0
    hashes = {p.name: sha256_file(p) for p in sorted(GOLDEN.glob("*.csv"))} if GOLDEN.exists() else {}
    body = {"as_of": as_of.isoformat(), "tickers": sorted(tickers), "watermarks": watermarks, "row_counts": counts,
            "file_hashes": hashes, "source_versions": {"demo_filing": "golden_v1", "calculators": "v1"}}
    mhash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    manifest = SnapshotManifest(snapshot_id=f"snap_{as_of:%Y%m%d%H%M}_{mhash[:8]}", as_of=as_of,
                                created_at=datetime.now().replace(microsecond=0), tickers=sorted(tickers),
                                watermarks=watermarks, row_counts=counts, file_hashes=hashes,
                                source_versions=body["source_versions"], manifest_hash=mhash)
    with Session() as s:
        if not s.get(DataSnapshotRow, manifest.snapshot_id):
            s.add(DataSnapshotRow(snapshot_id=manifest.snapshot_id, as_of=as_of, created_at=manifest.created_at,
                                  manifest=json.loads(manifest.model_dump_json())))
            s.commit()
    return manifest


def list_snapshots(db_url: Optional[str] = None) -> list[dict]:
    with get_session_factory(db_url)() as s:
        return [r.manifest for r in s.scalars(select(DataSnapshotRow).order_by(DataSnapshotRow.created_at.desc()))]
