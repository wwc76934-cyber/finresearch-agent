"""Data Source Registry: priorities, rate limits, health and fallback routing."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

CONFIG = Path(__file__).parent / "config" / "data_sources.yaml"


class DataSourceConfig(BaseModel):
    source_id: str
    kind: str
    priority: int
    rate_limit_per_min: int = 60
    enabled: bool = True
    path: Optional[str] = None
    description: str = ""


class DataSourceRegistry:
    def __init__(self, path: Path = CONFIG):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.sources = {s["source_id"]: DataSourceConfig(**s) for s in raw["sources"]}
        self.health: dict[str, str] = {k: "unknown" for k in self.sources}
        self._calls: dict[str, list[float]] = {}

    def priority(self, source_id: str) -> int:
        cfg = self.sources.get(source_id)
        return cfg.priority if cfg else 99

    def enabled(self) -> list[DataSourceConfig]:
        return sorted([s for s in self.sources.values() if s.enabled and self.health.get(s.source_id) != "down"],
                      key=lambda s: s.priority)

    def route(self, kind: Optional[str] = None) -> DataSourceConfig:
        """Pick the best healthy, enabled source (optionally of one kind)."""
        for s in self.enabled():
            if kind is None or s.kind == kind:
                return s
        raise LookupError(f"no healthy source for kind={kind}")

    def mark(self, source_id: str, status: str) -> None:
        self.health[source_id] = status

    def acquire(self, source_id: str) -> None:
        """Simple sliding-window rate limiter."""
        limit = self.sources[source_id].rate_limit_per_min
        if limit <= 0:
            return
        now = time.time()
        window = [t for t in self._calls.get(source_id, []) if now - t < 60]
        if len(window) >= limit:
            time.sleep(60 - (now - window[0]))
        window.append(time.time())
        self._calls[source_id] = window
