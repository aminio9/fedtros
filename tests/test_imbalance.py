import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.rl.environment import BlockchainIntrusionEnv
from src.utils.imbalance import compute_class_weights, sample_weights_from_class_weights


def test_inverse_frequency_weights_raise_minority_class():
    labels = torch.tensor([0] * 10 + [1], dtype=torch.long)

    weights = compute_class_weights(labels, num_classes=2, mode="inverse_frequency")

    assert torch.isfinite(weights).all()
    assert weights[1] > weights[0]


def test_sample_weights_follow_class_weights():
    labels = torch.tensor([0, 1, 1], dtype=torch.long)
    class_weights = torch.tensor([0.5, 2.0])

    sample_weights = sample_weights_from_class_weights(labels, class_weights)

    assert sample_weights.tolist() == [0.5, 2.0, 2.0]


def test_weighted_reward_changes_minority_reward_without_nan(tmp_path):
    path = tmp_path / "toy.pt"
    torch.save(
        {
            "features": torch.randn(4, 3),
            "labels": torch.tensor([0, 0, 0, 1], dtype=torch.long),
        },
        path,
    )
    cfg = OmegaConf.create(
        {
            "enabled": True,
            "manual_class_weights": [1.0, 3.0],
            "weighted_reward": True,
            "weight_negative_reward": False,
            "class_balanced_sampling": False,
            "log_weights": False,
        }
    )

    env = BlockchainIntrusionEnv(
        str(path),
        steps_per_episode=1,
        device=torch.device("cpu"),
        indices=np.array([3]),
        global_num_actions=2,
        imbalance_cfg=cfg,
    )
    env.reset(seed=1)
    _, reward, *_ = env.step(1)

    assert reward == 3.0
    assert np.isfinite(reward)


def test_class_balanced_sampling_builds_finite_probabilities(tmp_path):
    path = tmp_path / "toy.pt"
    torch.save(
        {
            "features": torch.randn(6, 3),
            "labels": torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.long),
        },
        path,
    )
    cfg = OmegaConf.create(
        {
            "enabled": True,
            "weight_mode": "inverse_frequency",
            "min_weight": 0.2,
            "max_weight": 5.0,
            "normalize": "mean",
            "weighted_reward": False,
            "class_balanced_sampling": True,
            "log_weights": False,
        }
    )

    env = BlockchainIntrusionEnv(
        str(path),
        steps_per_episode=2,
        device=torch.device("cpu"),
        global_num_actions=2,
        imbalance_cfg=cfg,
    )

    assert env._sampling_probabilities is not None
    assert np.isfinite(env._sampling_probabilities).all()
    assert env._sampling_probabilities.sum() == pytest.approx(1.0)
