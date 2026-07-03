from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from src.agents.agent import Agent
from src.agents.policy import EpsilonGreedyPolicy, EpsilonScheduler
from src.checkpointing.checkpoints import (
    CheckpointState,
    load_agent_checkpoint,
    metric_improved,
    save_agent_checkpoint,
    select_checkpoint_metric,
)
from src.data.io import load_tensor_dataset
from src.evaluation.closed_set import evaluate_closed_set, load_class_names
from src.models.cvae_dqn import OpenSetQChainModelFactory
from src.rl.environment import BlockchainIntrusionEnv
from src.rl.local_training import run_local_training_round
from src.rl.replay_buffer import ExperienceReplayBuffer
from src.tracking.local import LocalRunTracker
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
    tracker: LocalRunTracker,
) -> dict[str, Any]:
    train_data_path = _central_train_path(cfg, project_root)

    env = BlockchainIntrusionEnv(
        processed_data_path=str(train_data_path),
        steps_per_episode=int(cfg.training.steps_per_episode),
        device=device,
        move_data_to_device=bool(cfg.device.move_data_to_device),
        global_num_actions=int(cfg.model.num_actions),
        reward_correct=float(cfg.training.reward.correct),
        reward_incorrect=float(cfg.training.reward.incorrect),
        class_balanced_rewards=bool(cfg.training.reward.class_balanced),
        class_balance_power=float(cfg.training.reward.class_balance_power),
        imbalance_cfg=getattr(cfg.training, "imbalance", None),
    )
    agent = Agent(OpenSetQChainModelFactory(cfg.model), cfg.training, device=device)
    if cfg.training.resume_from is not None:
        load_agent_checkpoint(
            agent,
            resolve_path(project_root, cfg.training.resume_from),
            device,
            strict=bool(cfg.checkpointing.strict_load),
            load_optimizers=True,
        )

    buffer = ExperienceReplayBuffer(int(cfg.training.replay_buffer_size))
    policy = EpsilonGreedyPolicy(
        agent.prior_net,
        agent.value_net_main,
        int(cfg.model.num_actions),
        device,
    )
    epsilon_scheduler = EpsilonScheduler(cfg.training)

    best_metric: float | None = None
    best_metrics: dict[str, Any] = {}
    total_steps = 0

    val_data_path = (
        resolve_path(project_root, cfg.dataset.preprocessing.output_dir) / "validation.pt"
    )
    class_names_path = resolve_path(project_root, cfg.paths.class_names)
    can_validate = val_data_path.exists() and class_names_path.exists()
    class_names = (
        load_class_names(class_names_path, int(cfg.model.num_actions)) if can_validate else None
    )

    for epoch in range(1, int(cfg.training.epochs) + 1):
        steps, train_metrics = run_local_training_round(
            agent=agent,
            env=env,
            buffer=buffer,
            policy=policy,
            epsilon_scheduler=epsilon_scheduler,
            cfg_training=cfg.training,
            device=device,
        )
        total_steps += int(steps)
        metrics: dict[str, Any] = {
            "epoch": epoch,
            "global_step": total_steps,
            "train/loss": float(train_metrics.get("avg_td_loss", 0.0)),
            "train/total_loss": float(train_metrics.get("avg_total_loss", 0.0)),
            "train/accuracy": float(train_metrics.get("policy_accuracy", 0.0)),
            "train/reward": float(train_metrics.get("avg_reward_per_episode", 0.0)),
            "train/reward_mean": float(train_metrics.get("reward_mean", 0.0)),
            "train/reward_std": float(train_metrics.get("reward_std", 0.0)),
            "train/double_q_loss": float(train_metrics.get("avg_td_loss", 0.0)),
            "train/kl_loss": float(train_metrics.get("avg_kl_loss", 0.0)),
            "train/classification_loss": float(
                train_metrics.get("avg_classification_loss", 0.0)
            ),
            "train/prox_loss": float(train_metrics.get("avg_prox_loss", 0.0)),
            "train/gradient_norm_prior": float(
                train_metrics.get("gradient_norm_prior", 0.0)
            ),
            "train/gradient_norm_q": float(train_metrics.get("gradient_norm_q", 0.0)),
            "train/learning_rate_prior": float(
                train_metrics.get("learning_rate_prior", 0.0)
            ),
            "train/learning_rate_q_rl": float(
                train_metrics.get("learning_rate_q_rl", 0.0)
            ),
            "train/q_value_mean": float(train_metrics.get("q_value_mean", 0.0)),
            "train/q_value_std": float(train_metrics.get("q_value_std", 0.0)),
            "train/kl_std": float(train_metrics.get("kl_std", 0.0)),
            "train/epsilon": float(train_metrics.get("epsilon", 0.0)),
        }

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
            save_agent_checkpoint(
                agent,
                cfg,
                resolve_path(project_root, cfg.checkpointing.last_model_path),
                state,
            )

        selected_metric = select_checkpoint_metric(
            metrics,
            monitor_metric=str(cfg.checkpointing.monitor_metric),
        )
        monitor_value = selected_metric[1] if selected_metric is not None else None
        if metric_improved(
            float(monitor_value) if monitor_value is not None else None,
            best_metric,
            mode=str(cfg.checkpointing.monitor_mode),
        ):
            best_metric = float(monitor_value)
            best_metrics = dict(metrics)
            if bool(cfg.checkpointing.save_best):
                state.best_metric = best_metric
                metric_name = selected_metric[0] if selected_metric is not None else None
                save_agent_checkpoint(
                    agent,
                    cfg,
                    resolve_path(project_root, cfg.checkpointing.best_model_path),
                    state,
                    selected_metric_name=metric_name,
                    selected_metric_value=best_metric,
                    is_best=True,
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
    logger.info("Training complete | best_metric=%s | total_steps=%s", best_metric, total_steps)
    return summary
