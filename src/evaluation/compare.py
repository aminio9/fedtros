from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from src.utils.config import resolve_path

logger = logging.getLogger(__name__)


def _load_run_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "evaluation_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_csv = run_dir / "metrics.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        if not df.empty:
            return df.iloc[-1].dropna().to_dict()
    metrics_jsonl = run_dir / "metrics.jsonl"
    if metrics_jsonl.exists():
        lines = [
            line for line in metrics_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if lines:
            return json.loads(lines[-1])
    raise FileNotFoundError(f"No metrics file found in run directory: {run_dir}")


def compare_runs(cfg: DictConfig, *, project_root: Path) -> Path:
    if not cfg.runs:
        raise ValueError("Provide runs=[outputs/run1,outputs/run2] to compare_runs.py.")
    rows = []
    for run in cfg.runs:
        run_dir = resolve_path(project_root, run)
        metrics = _load_run_metrics(run_dir)
        row = {"run_dir": str(run_dir), **metrics}
        rows.append(row)
    df = pd.DataFrame(rows)
    output_dir = resolve_path(project_root, cfg.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "run_comparison.csv"
    df.to_csv(output_path, index=False)
    logger.info("Saved run comparison table to %s", output_path)
    return output_path
