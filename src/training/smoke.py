"""Smoke test routine for fast pipeline verification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from src.models.bundle import FedTROSModelBundle as Agent
from src.models.models import ModelFactory
from src.training.local_training import run_local_training_round
from src.experiment.run_services import MetricsSink

logger = logging.getLogger(__name__)


def run_smoke_test(
    cfg: DictConfig,
    *,
    project_root: Path,
    device: torch.device,
    tracker: MetricsSink,
) -> dict[str, Any]:
    _ = project_root
    smoke_dir = tracker.run_dir / "smoke_data"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    num_samples = int(cfg.training.smoke.num_samples)
    feature_dim = int(cfg.model.feature_dim)
    num_classes = int(cfg.model.num_classes)
    features = torch.randn(num_samples, feature_dim, generator=generator)
    labels = torch.arange(num_samples, dtype=torch.long) % num_classes
    data_path = smoke_dir / "client_1_train.pt"
    torch.save({"features": features, "labels": labels}, data_path)

    smoke_training = cfg.training.copy()
    smoke_training.batch_size = int(cfg.training.smoke.batch_size)
    smoke_training.local_epochs = 1

    model_factory = ModelFactory(cfg.model)
    agent = Agent(model_factory, smoke_training, device=device)

    steps, metrics = run_local_training_round(
        agent=agent,
        features=features,
        labels=labels,
        cfg_training=smoke_training,
        device=device,
        round_num=1,
        is_fedtros=True,
        logger=logger,
    )
    result = {
        "smoke/steps": int(steps),
        "smoke/train_loss": float(metrics.get("avg_student_total_loss", metrics.get("train_loss", 0.0))),
        "smoke/teacher_loss": float(metrics.get("avg_teacher_loss", 0.0)),
        "smoke/student_accuracy": float(metrics.get("student_acc", 0.0)),
    }
    tracker.log_metrics(result, step=1)
    tracker.write_json("smoke_test_metrics.json", result)
    logger.info("Smoke test complete | %s", result)
    return result
