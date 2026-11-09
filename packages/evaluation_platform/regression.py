"""Regression Suite: bad cases found in experiments are frozen as JSONL and re-run by pytest."""
from __future__ import annotations

import json
from datetime import datetime

from data_platform.db import ROOT

from .datasets import load_tasks

REG = ROOT / "benchmarks" / "regression" / "regression_v1.jsonl"


def load() -> list[dict]:
    if not REG.exists():
        return []
    return [json.loads(l) for l in REG.read_text(encoding="utf-8").splitlines() if l.strip()]


def promote(task_ids: list[str], reason: str) -> int:
    """Freeze tasks (by id) into the regression set with a note on the bug they caught."""
    tasks = {t["task_id"]: t for t in load_tasks()}
    existing = {r["task_id"] for r in load()}
    REG.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(REG, "a", encoding="utf-8") as f:
        for tid in task_ids:
            if tid in tasks and tid not in existing:
                f.write(json.dumps({**tasks[tid], "regression_reason": reason,
                                    "promoted_at": datetime.now().isoformat()}, ensure_ascii=False) + "\n")
                n += 1
    return n
