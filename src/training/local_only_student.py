"""Local-only Student baseline training routine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
import copy

import torch
from omegaconf import DictConfig

if TYPE_CHECKING:
    from src.models.bundle import FedTROSModelBundle as Agent
from src.training.local_training import run_local_training_round

logger = logging.getLogger(__name__)


def run_local_only_training(
    agent: Agent,
    features: torch.Tensor,
    labels: torch.Tensor,
    cfg_training: DictConfig,
    device: torch.device,
    client_id: str | int = 0,
) -> dict[str, Any]:
    """Execute purely local training (no federation) for a client over its local dataset.
    
    This trains the student model completely locally and can be used as a no-federation baseline.

    Args:
        agent: Local Agent instance.
        features: Tensor of local features [N, D].
        labels: Tensor of local labels [N].
        cfg_training: Training configuration.
        device: Active torch device.
        client_id: Client identifier.

    Returns:
        Summary metrics dictionary.
    """
    epochs = int(getattr(cfg_training, "local_epochs", getattr(cfg_training, "epochs", 50)))
    active_logger = logging.getLogger(f"LocalOnlyTraining.{client_id}")
    active_logger.info("Starting local-only training for client %s for %d epochs", client_id, epochs)
    steps, metrics = run_local_training_round(
        agent=agent,
        features=features,
        labels=labels,
        cfg_training=cfg_training,
        device=device,
        round_num=1,
        client_id=client_id,
        is_fedtros=False,
        logger=active_logger,
    )
    metrics = dict(metrics)
    metrics["global_step"] = int(steps)
    metrics["client_id"] = client_id
    metrics["is_local_only_baseline"] = 1.0
    active_logger.info(
        "Completed local-only training | client=%s | steps=%d | final_loss=%.4f",
        client_id, steps, metrics.get("train_loss", 0.0)
    )
    return metrics
