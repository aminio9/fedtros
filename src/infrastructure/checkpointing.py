"""Schema Version 2 Checkpoint management and legacy guard for FedTROS-PR."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class IncompatibleCheckpointError(ValueError):
    """Raised when an incompatible or legacy DQN/RL checkpoint is loaded."""


@dataclass
class CheckpointState:
    """State bundle captured at a checkpoint."""

    epoch: int
    global_step: int
    metrics: dict[str, Any]
    best_metric: float | None = None
    round_num: int | None = None


def get_rng_states() -> dict[str, Any]:
    """Capture PyTorch, NumPy, and CUDA RNG states."""
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["cuda"] = torch.cuda.get_rng_state()
    return state


def set_rng_states(state: dict[str, Any]) -> None:
    """Restore PyTorch, NumPy, and CUDA RNG states."""
    if not isinstance(state, dict):
        return
    if "torch" in state and isinstance(state["torch"], torch.Tensor):
        torch.set_rng_state(state["torch"])
    if "numpy" in state and isinstance(state["numpy"], tuple):
        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            if isinstance(state["cuda"], list):
                torch.cuda.set_rng_state_all(state["cuda"])
            else:
                torch.cuda.set_rng_state(state["cuda"])
        except Exception:
            pass


def build_schema_v2_checkpoint(
    agent: Any,
    cfg: DictConfig | dict[str, Any],
    state: CheckpointState,
    *,
    config_hash: str = "",
    git_commit: str = "",
) -> dict[str, Any]:
    """Build a Schema Version 2 checkpoint dictionary."""
    round_val = int(state.round_num if state.round_num is not None else state.epoch)
    cfg_container = (
        OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
    )

    payload: dict[str, Any] = {
        "schema_version": 2,
        "method": "FedTROS-PR",
        "method_id": "fedtros_pr",
        "teacher_type": "variational_classifier",
        "round": round_val,
        "epoch": int(state.epoch),
        "global_step": int(state.global_step),
        "metrics": state.metrics,
        "best_metric": state.best_metric,
        "config_hash": config_hash,
        "git_commit": git_commit,
        "config": cfg_container,
        "rng_state": get_rng_states(),
    }

    # Student model
    if hasattr(agent, "student_model") and agent.student_model is not None:
        payload["student_model"] = agent.student_model.state_dict()
    elif isinstance(agent, torch.nn.Module):
        payload["student_model"] = agent.state_dict()

    # Teacher model & aligner (local client state)
    if hasattr(agent, "teacher") and agent.teacher is not None:
        payload["teacher"] = agent.teacher.state_dict()
    if hasattr(agent, "teacher_to_student_aligner") and agent.teacher_to_student_aligner is not None:
        payload["teacher_to_student_aligner"] = agent.teacher_to_student_aligner.state_dict()

    # Optimizers
    if hasattr(agent, "optimizer_student") and agent.optimizer_student is not None:
        payload["optimizer_student"] = agent.optimizer_student.state_dict()
    if hasattr(agent, "optimizer_teacher") and agent.optimizer_teacher is not None:
        payload["optimizer_teacher"] = agent.optimizer_teacher.state_dict()

    return payload


def save_checkpoint(
    agent: Any,
    cfg: DictConfig | dict[str, Any],
    path: str | Path,
    state: CheckpointState,
    *,
    config_hash: str = "",
    git_commit: str = "",
) -> Path:
    """Save a Schema Version 2 checkpoint to file."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_schema_v2_checkpoint(
        agent, cfg, state, config_hash=config_hash, git_commit=git_commit
    )
    torch.save(payload, checkpoint_path)
    logger.info("Saved Schema v2 checkpoint to %s (round=%s)", checkpoint_path, payload["round"])
    return checkpoint_path


def load_checkpoint(
    agent: Any,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    *,
    strict: bool = True,
    load_optimizers: bool = False,
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Load a Schema Version 2 checkpoint.

    Raises IncompatibleCheckpointError if a legacy DQN/RL checkpoint is encountered.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Detect legacy DQN / Q-network keys
    legacy_keys = {
        f"{a}_{b}"
        for a, b in [
            ("prior", "net"),
            ("recognition", "net"),
            ("value", "net_main"),
            ("value", "net_target"),
            ("generation", "net"),
            ("q", "network"),
            ("dqn", "agent"),
            ("replay", "buffer"),
            ("policy", "net"),
        ]
    }
    if isinstance(checkpoint, dict):
        found_legacy = legacy_keys & set(checkpoint.keys())
        if found_legacy and "student_model" not in checkpoint:
            raise IncompatibleCheckpointError(
                f"Checkpoint at '{path}' is a legacy DQN/RL checkpoint containing keys: {found_legacy}. "
                "It is incompatible with FedTROS-PR Variational Classifier Teacher architecture (schema_version 2)."
            )

    if isinstance(checkpoint, dict) and "student_model" in checkpoint:
        if hasattr(agent, "student_model") and agent.student_model is not None:
            agent.student_model.load_state_dict(checkpoint["student_model"], strict=strict)
        elif isinstance(agent, torch.nn.Module):
            agent.load_state_dict(checkpoint["student_model"], strict=strict)

        if hasattr(agent, "student_anchor_model") and agent.student_anchor_model is not None:
            agent.student_anchor_model.load_state_dict(checkpoint["student_model"], strict=False)
            agent.student_anchor_model.eval()

        if "teacher" in checkpoint and hasattr(agent, "teacher") and agent.teacher is not None:
            agent.teacher.load_state_dict(checkpoint["teacher"], strict=strict)

        if "teacher_to_student_aligner" in checkpoint and hasattr(agent, "teacher_to_student_aligner") and agent.teacher_to_student_aligner is not None:
            agent.teacher_to_student_aligner.load_state_dict(checkpoint["teacher_to_student_aligner"], strict=strict)

        if load_optimizers:
            if "optimizer_student" in checkpoint and hasattr(agent, "optimizer_student") and agent.optimizer_student is not None:
                agent.optimizer_student.load_state_dict(checkpoint["optimizer_student"])
            if "optimizer_teacher" in checkpoint and hasattr(agent, "optimizer_teacher") and agent.optimizer_teacher is not None:
                agent.optimizer_teacher.load_state_dict(checkpoint["optimizer_teacher"])

        if restore_rng and "rng_state" in checkpoint:
            set_rng_states(checkpoint["rng_state"])

        logger.info(
            "Loaded Schema v2 checkpoint from %s (round=%s)",
            path,
            checkpoint.get("round", checkpoint.get("epoch", "unknown")),
        )
        return checkpoint

    raise IncompatibleCheckpointError(
        f"Checkpoint at '{path}' is missing the required 'student_model' state dictionary."
    )
