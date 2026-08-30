"""Centralized baseline training routine for FedTROS-PR."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from src.models.bundle import FedTROSModelBundle as Agent
from src.checkpointing.checkpoints import (
    CheckpointState,
    load_agent_checkpoint,
    metric_improved,
    save_agent_checkpoint,
)
from src.data.io import load_tensor_dataset
from src.evaluation.closed_set import evaluate_closed_set, load_class_names
from src.models.models import ModelFactory
from src.training.local_training import run_local_training_round
from src.experiment.run_services import MetricsSink
from src.utils.config import resolve_path

logger = logging.getLogger(__name__)


def _central_train_path(cfg: DictConfig, project_root: Path) -> Path:
    """Use the full known-train tensor for centralized baselines."""
    known_train_path = resolve_path(project_root, cfg.paths.known_train_data)
    if not known_train_path.exists():
        raise FileNotFoundError(
            f"Centralized known-train tensor not found: {known_train_path}. "
            "Run preprocessing first."
        )
    return known_train_path


def run_training(
    cfg: DictConfig,
    *,
    project_root: Path,
    device: torch.device,
    tracker: MetricsSink,
) -> dict[str, Any]:
    from src.models.bundle import FedTROSModelBundle as Agent

    train_data_path = _central_train_path(cfg, project_root)
    train_features, train_labels = load_tensor_dataset(train_data_path, map_location="cpu")

    model_factory = ModelFactory(cfg.model)
    agent = Agent(model_factory, cfg.training, device=device)
    if cfg.training.resume_from is not None:
        load_agent_checkpoint(
            agent,
            resolve_path(project_root, cfg.training.resume_from),
            device,
            strict=bool(cfg.checkpointing.strict_load),
            load_optimizers=True,
        )

    best_metric: float | None = None
    best_metrics: dict[str, Any] = {}
    total_steps = 0

    val_data_path = (
        resolve_path(project_root, cfg.dataset.preprocessing.output_dir) / "validation.pt"
    )
    class_names_path = resolve_path(project_root, cfg.paths.class_names)
    can_validate = val_data_path.exists() and class_names_path.exists()
    class_names = (
        load_class_names(class_names_path, int(cfg.model.num_classes)) if can_validate else None
    )

    method = str(OmegaConf.select(cfg, "experiment.method", default="Centralized-Student")).lower()
    is_fedtros = "fedtros" in method or method == "centralized-fedtros"

    for epoch in range(1, int(cfg.training.epochs) + 1):
        steps, train_metrics = run_local_training_round(
            agent=agent,
            features=train_features,
            labels=train_labels,
            cfg_training=cfg.training,
            device=device,
            round_num=epoch,
            is_fedtros=is_fedtros,
            logger=logger,
        )
        total_steps += int(steps)
        metrics: dict[str, Any] = {
            "epoch": epoch,
            "global_step": total_steps,
            "train/loss": float(train_metrics.get("avg_student_total_loss", train_metrics.get("train_loss", 0.0))),
            "train/task_loss": float(train_metrics.get("avg_student_task_loss", 0.0)),
            "train/accuracy": float(train_metrics.get("student_acc", 0.0)),
        }
        if is_fedtros:
            metrics.update({
                "train/teacher_loss": float(train_metrics.get("avg_teacher_loss", 0.0)),
                "train/teacher_kl_loss": float(train_metrics.get("avg_teacher_kl_loss", 0.0)),
                "train/kd_loss": float(train_metrics.get("avg_student_kd_loss", 0.0)),
                "train/align_loss": float(train_metrics.get("avg_student_align_loss", 0.0)),
            })

        if can_validate and epoch % int(cfg.training.validation_interval) == 0:
            val_features, val_labels = load_tensor_dataset(val_data_path, map_location="cpu")
            val_metrics = evaluate_closed_set(
                agent,
                val_features,
                val_labels,
                batch_size=int(cfg.training.batch_size),
                device=device,
                class_names=class_names or {},
                output_dir=tracker.run_dir,
                prefix="val",
                save_predictions=False,
            )
            metrics.update(val_metrics)

        tracker.log_metrics(metrics, step=epoch)

        state = CheckpointState(
            epoch=epoch,
            global_step=total_steps,
            metrics=metrics,
            best_metric=best_metric,
        )
        if (
            bool(cfg.checkpointing.save_latest)
            and epoch % int(cfg.training.checkpoint_interval) == 0
        ):
            save_agent_checkpoint(
                agent,
                cfg,
                resolve_path(project_root, cfg.checkpointing.latest_checkpoint_path),
                state,
            )

        monitor_value = metrics.get(str(cfg.checkpointing.monitor_metric))
        if monitor_value is None and str(cfg.checkpointing.monitor_metric) == "val/accuracy":
            monitor_value = metrics.get("train/accuracy")
        if metric_improved(
            float(monitor_value) if monitor_value is not None else None,
            best_metric,
            mode=str(cfg.checkpointing.monitor_mode),
        ):
            best_metric = float(monitor_value)
            best_metrics = dict(metrics)
            if bool(cfg.checkpointing.save_best):
                state.best_metric = best_metric
                save_agent_checkpoint(
                    agent,
                    cfg,
                    resolve_path(project_root, cfg.checkpointing.best_model_path),
                    state,
                )

    final_state = CheckpointState(
        epoch=int(cfg.training.epochs),
        global_step=total_steps,
        metrics=best_metrics or {"global_step": total_steps},
        best_metric=best_metric,
    )
    if bool(cfg.checkpointing.save_final):
        save_agent_checkpoint(
            agent,
            cfg,
            resolve_path(project_root, cfg.checkpointing.final_model_path),
            final_state,
        )

    summary = {
        "epochs": int(cfg.training.epochs),
        "global_step": total_steps,
        "best_metric": best_metric,
        "best_metrics": best_metrics,
    }
    tracker.write_json("training_summary.json", summary)
    logger.info("Centralized training complete | best_metric=%s | total_steps=%s", best_metric, total_steps)
    return summary
