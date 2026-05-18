from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.utils.config import resolve_path

logger = logging.getLogger(__name__)


def _run_metadata(run_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"run_dir": str(run_dir)}
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata.update(json.loads(metadata_path.read_text(encoding="utf-8")))

    config_path = run_dir / "resolved_config.yaml"
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        metadata.update(
            {
                "method": OmegaConf.select(cfg, "experiment.method"),
                "dataset": OmegaConf.select(cfg, "dataset.name"),
                "seed": OmegaConf.select(cfg, "seed"),
                "alpha": OmegaConf.select(cfg, "dataset.preprocessing.alpha"),
                "num_clients": OmegaConf.select(cfg, "federated.num_clients"),
                "strategy": OmegaConf.select(cfg, "federated.strategy.name"),
            }
        )
    return metadata


def _longify_metrics_frame(
    df: pd.DataFrame,
    *,
    metadata: dict[str, Any],
    source_file: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    index_columns = {"epoch", "step", "round", "global_step", "timestamp_utc"}
    numeric_columns = [
        col for col in df.columns if col not in index_columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_columns:
        return pd.DataFrame()

    round_column = next((col for col in ("round", "epoch", "step", "global_step") if col in df.columns), None)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        base = dict(metadata)
        base["source_file"] = source_file
        if round_column is not None:
            base["round"] = row.get(round_column)
        for metric_name in numeric_columns:
            rows.append(
                {
                    **base,
                    "metric_name": metric_name,
                    "metric_value": row[metric_name],
                }
            )
    return pd.DataFrame(rows)


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
    comparison_frames: list[pd.DataFrame] = []
    for run in cfg.runs:
        run_dir = resolve_path(project_root, run)
        metadata = _run_metadata(run_dir)
        metrics = _load_run_metrics(run_dir)
        row = {**metadata, **metrics}
        rows.append(row)

        history_path = run_dir / "federated_history.csv"
        if history_path.exists():
            history_df = pd.read_csv(history_path)
            if not history_df.empty:
                frame = history_df.copy()
                for key, value in metadata.items():
                    frame[key] = value
                frame["source_file"] = "federated_history.csv"
                comparison_frames.append(frame)

        metrics_csv = run_dir / "metrics.csv"
        if metrics_csv.exists():
            metrics_df = pd.read_csv(metrics_csv)
            frame = _longify_metrics_frame(
                metrics_df,
                metadata=metadata,
                source_file="metrics.csv",
            )
            if not frame.empty:
                comparison_frames.append(frame)
    df = pd.DataFrame(rows)
    output_dir = resolve_path(project_root, cfg.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "run_comparison.csv"
    df.to_csv(output_path, index=False)
    logger.info("Saved run comparison table to %s", output_path)
    if comparison_frames:
        comparison_path = output_dir / "comparison_metrics.csv"
        pd.concat(comparison_frames, ignore_index=True).to_csv(comparison_path, index=False)
        logger.info("Saved comparison metrics table to %s", comparison_path)
    return output_path
