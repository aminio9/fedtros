"""Checkpoint save/load utilities with schema versioning for FedTROS-PR."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from src.models.bundle import FedTROSModelBundle as Agent

logger = logging.getLogger(__name__)


class IncompatibleCheckpointError(ValueError):
    """Raised when an incompatible or legacy DQN/RL checkpoint is loaded."""


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
    """Build a Schema Version 2 checkpoint dictionary."""
    payload = {
        "schema_version": 2,
        "method": "FedTROS-PR",
        "teacher_type": "variational_classifier",
        "teacher_latent_dim": int(getattr(agent, "latent_dim", 64)),
        "epoch": int(state.epoch),
        "round": int(state.epoch),
        "global_step": int(state.global_step),
        "metrics": state.metrics,
        "best_metric": state.best_metric,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "student_model": agent.student_model.state_dict(),
        "teacher": agent.teacher.state_dict(),
        "teacher_to_student_aligner": agent.teacher_to_student_aligner.state_dict(),
        "optimizer_student": agent.optimizer_student.state_dict(),
        "optimizer_teacher": agent.optimizer_teacher.state_dict(),
    }
    if bool(cfg.checkpointing.include_rng_state):
        payload["rng_state"] = _rng_state()
    return payload


def save_agent_checkpoint(
    agent: Agent,
    cfg: DictConfig,
    path: str | Path,
    state: CheckpointState,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_agent_checkpoint(agent, cfg, state), checkpoint_path)
    logger.info("Saved Schema v2 checkpoint to %s", checkpoint_path)
    return checkpoint_path


def load_agent_checkpoint(
    agent: Agent,
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    strict: bool = True,
    load_optimizers: bool = False,
) -> dict[str, Any]:
    """Load a Schema Version 2 checkpoint into the Agent.

    Raises IncompatibleCheckpointError if an old DQN/Q-network checkpoint is encountered.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Check for legacy DQN / Q-network keys
    legacy_keys = {f"{a}_{b}" for a, b in [("prior", "net"), ("recognition", "net"), ("value", "net_main"), ("value", "net_target"), ("generation", "net")]}
    if isinstance(checkpoint, dict) and any(k in checkpoint for k in legacy_keys) and "student_model" not in checkpoint:
        raise IncompatibleCheckpointError(
            f"Checkpoint at '{path}' is a legacy DQN/RL checkpoint (contains {legacy_keys & set(checkpoint.keys())}). "
            "It cannot be loaded into the FedTROS-PR Variational Classifier Teacher architecture (schema_version 2)."
        )

    if isinstance(checkpoint, dict) and "student_model" in checkpoint:
        agent.student_model.load_state_dict(checkpoint["student_model"], strict=strict)
        if hasattr(agent, "student_anchor_model") and agent.student_anchor_model is not None:
            agent.student_anchor_model.load_state_dict(checkpoint["student_model"], strict=False)
            agent.student_anchor_model.eval()

        if "teacher" in checkpoint and hasattr(agent, "teacher") and agent.teacher is not None:
            agent.teacher.load_state_dict(checkpoint["teacher"], strict=strict)

        if "teacher_to_student_aligner" in checkpoint and hasattr(agent, "teacher_to_student_aligner"):
            agent.teacher_to_student_aligner.load_state_dict(checkpoint["teacher_to_student_aligner"], strict=strict)

        if load_optimizers:
            if "optimizer_student" in checkpoint:
                agent.optimizer_student.load_state_dict(checkpoint["optimizer_student"])
            if "optimizer_teacher" in checkpoint:
                agent.optimizer_teacher.load_state_dict(checkpoint["optimizer_teacher"])

        logger.info("Loaded Schema v2 checkpoint %s at round/epoch=%s", path, checkpoint.get("round", checkpoint.get("epoch", "unknown")))
        return checkpoint

    raise IncompatibleCheckpointError(f"Checkpoint at '{path}' is missing required 'student_model' dictionary.")


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
