from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.agents.agent import Agent

logger = logging.getLogger(__name__)


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


def save_agent_checkpoint(
    agent: Agent,
    cfg: DictConfig,
    path: str | Path,
    state: CheckpointState,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_agent_checkpoint(agent, cfg, state), checkpoint_path)
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
