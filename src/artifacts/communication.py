from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import OmegaConf

from src.utils.config import resolve_path

logger = logging.getLogger(__name__)

MODEL_KEY_ORDER = (
    "prior_net",
    "recognition_net",
    "value_net_main",
    "value_net_target",
    "generation_net",
)


def _tensor_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_nbytes(item) for item in value)
    return 0


def estimate_checkpoint_parameter_bytes(checkpoint: dict[str, Any]) -> int:
    total = sum(_tensor_nbytes(checkpoint.get(key)) for key in MODEL_KEY_ORDER)
    return total if total > 0 else _tensor_nbytes(checkpoint)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return {}


def _load_resolved_config(run_dir: Path) -> Any:
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.exists():
        return None
    try:
        return OmegaConf.load(config_path)
    except Exception as exc:
        logger.warning("Could not load resolved config %s: %s", config_path, exc)
        return None


def _load_monitoring_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "fmrl_ava_monitoring.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except Exception as exc:
                logger.warning("Skipping malformed monitor line in %s: %s", path, exc)
    return records


def _load_round_frame(run_dir: Path) -> pd.DataFrame:
    history_path = run_dir / "federated_history.csv"
    if history_path.exists():
        try:
            return pd.read_csv(history_path)
        except Exception as exc:
            logger.warning("Could not read %s: %s", history_path, exc)
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        try:
            return pd.read_csv(metrics_path)
        except Exception as exc:
            logger.warning("Could not read %s: %s", metrics_path, exc)
    return pd.DataFrame()


def _metric_candidates() -> tuple[str, ...]:
    return (
        "accuracy",
        "test/accuracy",
        "federated/global_accuracy",
        "federated/accuracy",
        "openset_overall_acc",
    )


def _select_round_accuracy(df: pd.DataFrame, round_col: str) -> pd.DataFrame:
    if df.empty or "metric_name" not in df.columns or "metric_value" not in df.columns:
        return pd.DataFrame()

    candidates = _metric_candidates()
    metric_df = df.loc[df["metric_name"].astype(str).isin(candidates)].copy()
    if metric_df.empty:
        return pd.DataFrame()

    metric_df["_metric_priority"] = metric_df["metric_name"].map(
        {metric: idx for idx, metric in enumerate(candidates)}
    )
    metric_df = metric_df.sort_values([round_col, "_metric_priority"])
    metric_df = metric_df.groupby(round_col, as_index=False).first()
    return metric_df[[round_col, "metric_value", "metric_name"]]


def _communication_bytes_per_round(
    *,
    method: str,
    num_clients: int,
    model_parameter_bytes: int,
    monitoring_records: list[dict[str, Any]],
) -> dict[int, int]:
    if method.lower() != "fmrl_ava":
        return {}

    selected_by_logical_round: dict[int, int] = {}
    for record in monitoring_records:
        event = str(record.get("event", ""))
        logical_round = record.get("logical_round")
        if logical_round is None:
            server_round = record.get("server_round")
            if server_round is None:
                continue
            logical_round = int((int(server_round) + 1) // 2)
        logical_round = int(logical_round)
        if event == "phase_b_aggregation":
            uploads = record.get("uploads", [])
            selected_by_logical_round[logical_round] = int(len(uploads))
        elif event == "phase_a_selection" and logical_round not in selected_by_logical_round:
            selected_by_logical_round[logical_round] = int(record.get("selected_clients", num_clients))

    return {
        logical_round: int(model_parameter_bytes * (num_clients + (2 * selected_clients)))
        for logical_round, selected_clients in selected_by_logical_round.items()
    }


def build_communication_metrics(
    *,
    run_dir: Path,
    project_root: Path,
    history_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Estimate cumulative model-traffic and round accuracy for a single run."""
    cfg = _load_resolved_config(run_dir)
    if cfg is None:
        return pd.DataFrame()

    method = str(OmegaConf.select(cfg, "experiment.method", default=""))
    num_clients = int(OmegaConf.select(cfg, "federated.num_clients", default=0) or 0)
    if num_clients <= 0:
        return pd.DataFrame()

    checkpoint_candidates = [
        resolve_path(project_root, run_dir / "best_model.pt"),
        resolve_path(project_root, run_dir / "latest_checkpoint.pt"),
        resolve_path(project_root, run_dir / "global_model_latest.pt"),
    ]
    checkpoint_path = next((path for path in checkpoint_candidates if path.exists()), None)
    if checkpoint_path is None:
        checkpoint_dir = OmegaConf.select(cfg, "checkpointing.dir", default=None)
        if checkpoint_dir:
            for candidate_name in ("best_model.pt", "latest_checkpoint.pt", "global_model_latest.pt"):
                candidate = resolve_path(project_root, Path(checkpoint_dir) / candidate_name)
                if candidate.exists():
                    checkpoint_path = candidate
                    break
    if checkpoint_path is None:
        return pd.DataFrame()

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning("Could not load checkpoint %s: %s", checkpoint_path, exc)
        return pd.DataFrame()

    model_parameter_bytes = estimate_checkpoint_parameter_bytes(checkpoint)
    if model_parameter_bytes <= 0:
        return pd.DataFrame()

    if history_frame is None:
        history_frame = _load_round_frame(run_dir)
    if history_frame is None or history_frame.empty:
        return pd.DataFrame()

    if "round" not in history_frame.columns and "logical_round" not in history_frame.columns:
        return pd.DataFrame()

    frame = history_frame.copy()
    if "logical_round" not in frame.columns and "round" in frame.columns:
        if method.lower() == "fmrl_ava":
            frame["logical_round"] = ((frame["round"].astype(int) + 1) // 2).astype(int)
        else:
            frame["logical_round"] = frame["round"].astype(int)

    round_col = "logical_round"
    if "metric_name" not in frame.columns:
        if "accuracy" in frame.columns:
            frame = frame[[round_col, "accuracy"]].dropna().copy()
            frame = frame.rename(columns={"accuracy": "metric_value"})
            frame["metric_name"] = "accuracy"
        elif "test/accuracy" in frame.columns:
            frame = frame[[round_col, "test/accuracy"]].dropna().copy()
            frame = frame.rename(columns={"test/accuracy": "metric_value"})
            frame["metric_name"] = "test/accuracy"
        else:
            return pd.DataFrame()
    else:
        frame = _select_round_accuracy(frame, round_col)
        if frame.empty:
            return pd.DataFrame()

    frame = frame.sort_values(round_col).copy()
    monitoring_records = _load_monitoring_records(run_dir)
    bytes_per_round = _communication_bytes_per_round(
        method=method,
        num_clients=num_clients,
        model_parameter_bytes=model_parameter_bytes,
        monitoring_records=monitoring_records,
    )
    if not bytes_per_round:
        per_round_bytes = model_parameter_bytes * (2 * num_clients)
        bytes_per_round = {int(round_id): int(per_round_bytes) for round_id in frame[round_col]}

    cumulative = 0
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        round_id = int(row[round_col])
        cumulative += int(bytes_per_round.get(round_id, model_parameter_bytes * (2 * num_clients)))
        rows.append(
            {
                "method": method,
                "round": round_id,
                "cumulative_mb": cumulative / 1_000_000.0,
                "accuracy": float(row["metric_value"]),
                "source_run_dir": str(run_dir),
            }
        )

    return pd.DataFrame(rows)
