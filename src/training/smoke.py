from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from src.agents.agent import Agent
from src.agents.policy import EpsilonGreedyPolicy, EpsilonScheduler
<<<<<<< HEAD
from src.models.models import OpenSetQChainModelFactory
=======
from src.models.cvae_dqn import OpenSetQChainModelFactory
>>>>>>> ea28efe (Initial commit with updated source code)
from src.rl.environment import BlockchainIntrusionEnv
from src.rl.local_training import run_local_training_round
from src.rl.replay_buffer import ExperienceReplayBuffer
from src.tracking.local import LocalRunTracker

logger = logging.getLogger(__name__)


def run_smoke_test(
    cfg: DictConfig,
    *,
    project_root: Path,
    device: torch.device,
    tracker: LocalRunTracker,
) -> dict[str, Any]:
    _ = project_root
    smoke_dir = tracker.run_dir / "smoke_data"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    num_samples = int(cfg.training.smoke.num_samples)
    state_dim = int(cfg.model.state_dim)
    num_actions = int(cfg.model.num_actions)
    features = torch.randn(num_samples, state_dim, generator=generator)
    labels = torch.arange(num_samples, dtype=torch.long) % num_actions
    data_path = smoke_dir / "client_1_train.pt"
    torch.save({"features": features, "labels": labels}, data_path)

    smoke_training = cfg.training.copy()
    smoke_training.local_episodes_per_round = int(cfg.training.smoke.epochs)
    smoke_training.steps_per_episode = int(cfg.training.smoke.steps_per_episode)
    smoke_training.min_buffer_size = int(cfg.training.smoke.min_buffer_size)
    smoke_training.batch_size = int(cfg.training.smoke.batch_size)
    smoke_training.replay_buffer_size = max(int(cfg.training.replay_buffer_size), num_samples)

    agent = Agent(OpenSetQChainModelFactory(cfg.model), smoke_training, device=device)
<<<<<<< HEAD
=======
    agent.set_local_class_counts(torch.bincount(labels, minlength=num_actions)[:num_actions].tolist())
>>>>>>> ea28efe (Initial commit with updated source code)
    env = BlockchainIntrusionEnv(
        str(data_path),
        int(smoke_training.steps_per_episode),
        device=device,
        global_num_actions=num_actions,
<<<<<<< HEAD
    )
    buffer = ExperienceReplayBuffer(int(smoke_training.replay_buffer_size))
    policy = EpsilonGreedyPolicy(agent.prior_net, agent.value_net_main, num_actions, device)
=======
        reward_correct=float(getattr(getattr(smoke_training, "reward", None), "correct", 1.0)),
        reward_incorrect=float(
            getattr(getattr(smoke_training, "reward", None), "incorrect", -1.0)
        ),
        class_balanced_rewards=bool(
            getattr(getattr(smoke_training, "reward", None), "class_balanced", False)
        ),
        class_balance_power=float(
            getattr(getattr(smoke_training, "reward", None), "class_balance_power", 1.0)
        ),
        imbalance_cfg=getattr(smoke_training, "imbalance", None),
    )
    buffer = ExperienceReplayBuffer(int(smoke_training.replay_buffer_size))
    policy = EpsilonGreedyPolicy(agent.prior_net, agent.value_net_main, num_actions, device)
    policy.set_allowed_actions(list(range(num_actions)))
>>>>>>> ea28efe (Initial commit with updated source code)
    scheduler = EpsilonScheduler(smoke_training)
    steps, metrics = run_local_training_round(
        agent=agent,
        env=env,
        buffer=buffer,
        policy=policy,
        epsilon_scheduler=scheduler,
        cfg_training=smoke_training,
        device=device,
    )
    result = {
        "smoke/steps": int(steps),
        "smoke/train_loss": float(metrics.get("avg_td_loss", 0.0)),
        "smoke/train_accuracy": float(metrics.get("policy_accuracy", 0.0)),
    }
    tracker.log_metrics(result, step=1)
    tracker.write_json("smoke_test_metrics.json", result)
    logger.info("Smoke test complete | %s", result)
    return result
