"""Local training routine for federated clients."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from src.models.bundle import FedTROSModelBundle as Agent
from src.training.class_balance import class_balanced_cross_entropy, effective_number_class_weights

logger = logging.getLogger("LocalTraining")


def run_local_training_round(
    agent: Agent,
    features: torch.Tensor,
    labels: torch.Tensor,
    cfg_training: DictConfig,
    device: torch.device,
    *,
    proximal_mu: float = 0.0,
    round_num: int = 0,
    client_id: str | int = 0,
    is_fedtros: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[int, dict[str, Any]]:
    """Execute local training for a client over its local dataset.

    Args:
        agent: Local Agent instance.
        features: Tensor of local features [N, D].
        labels: Tensor of local labels [N].
        cfg_training: Training configuration.
        device: Active torch device.
        proximal_mu: FedProx proximal weight (0.0 for FedAvg / FedTROS).
        round_num: Current communication round index.
        client_id: Client identifier.
        is_fedtros: True if running FedTROS (VCT + Student), False for standard baselines.
        logger: Logger instance.

    Returns:
        total_steps: Number of optimizer steps executed.
        metrics: Summary metrics dictionary for the round.
    """
    active_logger = logger or logging.getLogger(f"LocalTraining.{client_id}")
    num_classes = agent.num_classes

    # Compute class weights and present class mask
    class_weights = effective_number_class_weights(
        labels.detach().cpu(),
        num_classes,
        beta=float(getattr(cfg_training, "class_balance_beta", 0.999)),
        min_weight=float(getattr(cfg_training, "class_weight_min", 0.2)),
        max_weight=float(getattr(cfg_training, "class_weight_max", 5.0)),
        normalize=True,
        device=device,
    )
    counts = torch.bincount(labels.detach().cpu().long().clamp_min(0), minlength=num_classes)[:num_classes]
    present_classes = (counts > 0).to(device)

    if is_fedtros:
        # FedTROS-PR: Private VCT teacher + Guided Student
        metrics = agent.train_fedtros_dataset(
            features=features,
            labels=labels,
            cfg_training=cfg_training,
            round_num=round_num,
            class_weights=class_weights,
            present_classes=present_classes,
            device=device,
            logger=active_logger,
        )
        total_steps = int(metrics.get("student_train_steps", 0))
    else:
        # Standard Baseline (FedAvg-Student / FedProx-Student)
        batch_size = int(getattr(cfg_training, "batch_size", 64))
        local_epochs = int(getattr(cfg_training, "local_epochs", 2))
        label_smoothing = float(getattr(cfg_training, "label_smoothing", 0.0))
        max_grad_norm = float(getattr(cfg_training, "grad_clip_norm", 1.0))

        dataset = TensorDataset(features.float().to(device), labels.long().to(device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        agent.student_model.train()

        total_loss = 0.0
        total_steps = 0
        student_start = time.perf_counter()

        for _ in range(max(1, local_epochs)):
            for bx, by in loader:
                agent.optimizer_student.zero_grad()
                _, logits = agent.student_model(bx)
                loss = class_balanced_cross_entropy(
                    logits, by, weights=class_weights, label_smoothing=label_smoothing
                )
                if proximal_mu > 0.0:
                    loss = loss + (0.5 * proximal_mu * agent._proximal_penalty())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.student_model.classifier_parameters(), max_norm=max_grad_norm)
                agent.optimizer_student.step()

                total_loss += float(loss.detach().item())
                total_steps += 1

        denom = max(1, total_steps)
        student_seconds = float(time.perf_counter() - student_start)
        metrics = {
            "train_loss": total_loss / denom,
            "train_steps": float(total_steps),
            "proximal_mu": float(proximal_mu),
            "is_standard_baseline": 1.0,
            "runtime/teacher_seconds": 0.0,
            "runtime/student_seconds": student_seconds,
        }
        active_logger.info(
            "Standard baseline local training | steps=%d loss=%.4f proximal_mu=%.4f",
            total_steps,
            metrics["train_loss"],
            proximal_mu,
        )

    return total_steps, metrics
