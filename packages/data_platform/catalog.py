"""Data Catalog: metric dictionary and industry templates."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG = Path(__file__).parent / "config" / "metrics.yaml"


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def cn_name(metric: str) -> str:
    cat = load_catalog()
    for section in ("base_items", "derived"):
        if metric in cat[section]:
            return cat[section][metric]["cn"]
    return metric


def formula(metric: str) -> str:
    return load_catalog()["derived"].get(metric, {}).get("formula", "raw fact")


def formula_version(metric: str) -> str:
    return load_catalog()["derived"].get(metric, {}).get("version", "v1")


def template_metrics(template: str) -> list[str]:
    return load_catalog()["templates"].get(template, [])
