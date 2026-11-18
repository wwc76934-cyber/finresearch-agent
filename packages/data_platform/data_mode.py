"""Where a run may read data from.

offline_golden       only the frozen local snapshots. Every benchmark and test runs in this mode, so results stay
                     reproducible.
hybrid_live          local database first; a company that is missing, or whose local data stops before the requested
                     as_of, is fetched on demand and stored (Read-Through). The default for the API and web UI.
online_force_refresh always refetch, even when the local watermark covers the request.
"""
from __future__ import annotations

import os
from typing import Literal

DataMode = Literal["offline_golden", "hybrid_live", "online_force_refresh"]
MODES = ("offline_golden", "hybrid_live", "online_force_refresh")


def default_data_mode() -> DataMode:
    """FINRESEARCH_DATA_MODE overrides the default; tests and benchmark scripts set it to offline_golden."""
    v = os.environ.get("FINRESEARCH_DATA_MODE", "hybrid_live")
    if v not in MODES:
        raise ValueError(f"FINRESEARCH_DATA_MODE={v!r}; choose one of {MODES}")
    return v  # type: ignore[return-value]
