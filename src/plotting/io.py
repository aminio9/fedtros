from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return None


def load_jsonl_if_exists(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_existing(run_dir: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
        nested = next(run_dir.rglob(name), None)
        if nested is not None:
            return nested
    return None
