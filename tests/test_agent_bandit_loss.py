"""Regression tests for contextual-bandit Q supervision."""

import math

import torch
from omegaconf import OmegaConf

from src.agents.agent import Agent
from src.models.cvae_dqn import OpenSetQChainModelFactory


def _tiny_training_cfg():
    return OmegaConf.create(
        {
            "rl_mode": "contextual_bandit",
            "gamma": 0.0,
            "use_double_dqn": True,
            "lr_prior": 1e-3,
            "lr_q_rl": 1e-3,
            "prior_grad_clip_norm": 1.0,
            "q_grad_clip_norm": 1.0,
            "prior_kl_raw": False,
            "loss_weights": {
                "prior_kl": 0.5,
                "q_td": 0.25,
                "bandit_q": 1.0,
                "classification": 2.0,
                "generator_reconstruction": 1.0,
                "proximal": 1.0,
            },
            "reward": {"correct": 1.0, "incorrect": -1.0},
            "missing_class_gradient": {"enabled": True, "mask_value": -20.0},
            "classification_loss": {
                "name": "focal",
                "focal_gamma": 1.5,
                "use_class_weights": True,
            },
            "imbalance": {
                "enabled": True,
                "weight_mode": "effective_number",
                "effective_number_beta": 0.999,
                "min_weight": 0.3,
                "max_weight": 3.0,
                "normalize": "mean",
                "class_balanced_sampling": True,
                "weighted_reward": True,
                "weight_negative_reward": False,
            },
            "auxiliary_losses": {
                "supervised_contrastive_lambda": 0.02,
                "supervised_contrastive_temperature": 0.1,
                "center_loss_lambda": 0.01,
            },
            "kl": {"free_nats": 0.25, "warmup_steps": 1},
        }
    )


def _tiny_model_cfg():
    return OmegaConf.create(
        {
            "state_dim": 5,
            "latent_dim": 3,
            "num_actions": 4,
            "backbone": {
                "hidden_dim": 8,
                "depth": 1,
                "ensemble_size": 1,
                "dropout": 0.0,
                "expansion": 1,
                "q_hidden_dim": 8,
                "q_depth": 1,
                "q_ensemble_size": 1,
            },
            "generator": {"hidden_dim": 8, "depth": 1, "dropout": 0.0, "expansion": 1},
        }
    )


def test_agent_bandit_loss_metrics_are_finite():
    torch.manual_seed(7)
    agent = Agent(
        OpenSetQChainModelFactory(_tiny_model_cfg()),
        _tiny_training_cfg(),
        torch.device("cpu"),
    )
    agent.set_local_class_counts([4, 0, 4, 0])

    labels = torch.tensor([[0], [2], [0], [2]], dtype=torch.long)
    batch = (
        torch.randn(4, 5),
        labels.clone(),
        torch.ones(4, 1),
        torch.randn(4, 5),
        torch.ones(4, 1),
        labels,
    )

    metrics = agent.train_step(batch)

    assert "loss/bandit_q" in metrics
    assert "loss/bandit_q_weighted" in metrics
    assert math.isfinite(metrics["loss/bandit_q"])
    assert math.isfinite(metrics["loss/bandit_q_weighted"])
    assert math.isfinite(metrics["loss/total"])
    assert metrics["local_class_coverage_count"] == 2.0
