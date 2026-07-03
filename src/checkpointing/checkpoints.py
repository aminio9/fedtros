from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.agents.agent import Agent

logger = logging.getLogger(__name__)

VALIDATION_PREFIXES = ("val/", "validation/")
VALIDATION_COMPONENTS = (
    "val/macro_f1",
    "val/balanced_accuracy",
    "validation/macro_f1",
    "validation/balanced_accuracy",
    "val/open_set/auroc",
    "val/open_set/unknown_f1",
    "validation/open_set/auroc",
    "validation/open_set/unknown_f1",
)


@dataclass
class CheckpointState:
    epoch: int
    global_step: int
    metrics: dict[str, Any]
    best_metric: float | None = None


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def build_agent_checkpoint(
    agent: Agent,
    cfg: DictConfig,
    state: CheckpointState,
) -> dict[str, Any]:
    payload = {
        "epoch": int(state.epoch),
        "round": int(state.epoch),
        "global_step": int(state.global_step),
        "metrics": state.metrics,
        "best_metric": state.best_metric,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "prior_net": agent.prior_net.state_dict(),
        "recognition_net": agent.recognition_net.state_dict(),
        "value_net_main": agent.value_net_main.state_dict(),
        "value_net_target": agent.value_net_target.state_dict(),
        "generation_net": (
            agent.generation_net.state_dict() if agent.generation_net is not None else None
        ),
        "optimizer_prior": agent.optimizer_prior.state_dict(),
        "optimizer_q_rl": agent.optimizer_q_rl.state_dict(),
    }
    if bool(cfg.checkpointing.include_rng_state):
        payload["rng_state"] = _rng_state()
    return payload


def _config_payload(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _config_hash(cfg: DictConfig) -> str:
    payload = json.dumps(_config_payload(cfg), sort_keys=True, default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def _selected_known_labels(cfg: DictConfig) -> list[str]:
    labels = OmegaConf.select(cfg, "dataset.preprocessing.known_labels", default=[])
    return [str(label) for label in labels or []]


def _selected_unknown_labels(cfg: DictConfig) -> list[str]:
    source_labels = OmegaConf.select(cfg, "dataset.source_labels", default=[])
    known_labels = set(_selected_known_labels(cfg))
    return [str(label) for label in source_labels or [] if str(label) not in known_labels]


def is_validation_metric(metric_name: str | None) -> bool:
    if not metric_name:
        return False
    return str(metric_name).startswith(VALIDATION_PREFIXES)


def select_checkpoint_metric(
    metrics: dict[str, Any],
    *,
    monitor_metric: str,
) -> tuple[str, float] | None:
    """Select a validation-only checkpoint metric from a metric dictionary."""
    if monitor_metric == "combined_validation_score":
        values = []
        for key in VALIDATION_COMPONENTS:
            if key in metrics:
                try:
                    values.append(float(metrics[key]))
                except (TypeError, ValueError):
                    continue
        if values:
            return monitor_metric, float(np.mean(values))
        return None

    if not is_validation_metric(monitor_metric):
        return None
    value = metrics.get(monitor_metric)
    if value is None:
        return None
    try:
        return monitor_metric, float(value)
    except (TypeError, ValueError):
        return None


def build_checkpoint_metadata(
    cfg: DictConfig,
    state: CheckpointState,
    *,
    checkpoint_path: str | Path,
    selected_metric_name: str | None = None,
    selected_metric_value: float | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_path": str(checkpoint_path),
        "epoch": int(state.epoch),
        "round": int(state.epoch),
        "global_step": int(state.global_step),
        "selected_metric_name": selected_metric_name,
        "selected_metric_value": selected_metric_value,
        "config_hash": _config_hash(cfg),
        "seed": OmegaConf.select(cfg, "seed", default=None),
        "known_labels": _selected_known_labels(cfg),
        "unknown_labels": _selected_unknown_labels(cfg),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_checkpoint_metadata(
    cfg: DictConfig,
    path: str | Path,
    state: CheckpointState,
    *,
    selected_metric_name: str | None = None,
    selected_metric_value: float | None = None,
    is_best: bool = False,
) -> None:
    checkpoint_path = Path(path)
    metadata = build_checkpoint_metadata(
        cfg,
        state,
        checkpoint_path=checkpoint_path,
        selected_metric_name=selected_metric_name,
        selected_metric_value=selected_metric_value,
    )
    metadata_path = checkpoint_path.parent / "checkpoint_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    if is_best:
        best_payload = {
            "selected_metric_name": selected_metric_name,
            "selected_metric_value": selected_metric_value,
            "metrics": state.metrics,
            "metadata": metadata,
        }
        (checkpoint_path.parent / "best_metrics.json").write_text(
            json.dumps(best_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def save_agent_checkpoint(
    agent: Agent,
    cfg: DictConfig,
    path: str | Path,
    state: CheckpointState,
    *,
    selected_metric_name: str | None = None,
    selected_metric_value: float | None = None,
    is_best: bool = False,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_agent_checkpoint(agent, cfg, state), checkpoint_path)
    write_checkpoint_metadata(
        cfg,
        checkpoint_path,
        state,
        selected_metric_name=selected_metric_name,
        selected_metric_value=selected_metric_value,
        is_best=is_best,
    )
    logger.info("Saved checkpoint to %s", checkpoint_path)
    return checkpoint_path


def load_agent_checkpoint(
    agent: Agent,
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    strict: bool = True,
    load_optimizers: bool = False,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    agent.prior_net.load_state_dict(checkpoint["prior_net"], strict=strict)
    agent.recognition_net.load_state_dict(checkpoint["recognition_net"], strict=strict)
    agent.value_net_main.load_state_dict(checkpoint["value_net_main"], strict=strict)
    target_state = checkpoint.get("value_net_target", checkpoint["value_net_main"])
    agent.value_net_target.load_state_dict(target_state, strict=strict)
    if agent.generation_net is not None and checkpoint.get("generation_net") is not None:
        agent.generation_net.load_state_dict(checkpoint["generation_net"], strict=strict)

    if load_optimizers:
        if "optimizer_prior" in checkpoint:
            agent.optimizer_prior.load_state_dict(checkpoint["optimizer_prior"])
        if "optimizer_q_rl" in checkpoint:
            agent.optimizer_q_rl.load_state_dict(checkpoint["optimizer_q_rl"])

    logger.info("Loaded checkpoint %s at epoch=%s", path, checkpoint.get("epoch", "unknown"))
    return checkpoint


def metric_improved(
    current: float | None,
    best: float | None,
    *,
    mode: str,
) -> bool:
    if current is None:
        return False
    if best is None:
        return True
    if mode == "min":
        return current < best
    if mode == "max":
        return current > best
    raise ValueError(f"checkpointing.monitor_mode must be 'min' or 'max', got {mode!r}")
