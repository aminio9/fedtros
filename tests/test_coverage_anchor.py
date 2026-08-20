"""Tests for coverage-adaptive global anchor regularization in FedTROS-PR."""

import torch
from omegaconf import OmegaConf

from src.models.bundle import FedTROSModelBundle as Agent
from src.models.models import ModelFactory


def test_coverage_adaptive_anchor_scaling():
    num_classes = 10
    feature_dim = 20
    model_cfg = OmegaConf.create({"feature_dim": feature_dim, "latent_dim": 16, "num_classes": num_classes})
    train_cfg = OmegaConf.create(
        {
            "fedtros_global_anchor_weight": 2.0,
            "fedtros_global_anchor_min_weight": 0.0,
            "fedtros_global_anchor_coverage_power": 1.0,
            "teacher_beta_kl": 0.01,
            "lambda_kd_init": 0.20,
            "lambda_align_init": 0.08,
        }
    )

    device = torch.device("cpu")
    agent = Agent(ModelFactory(model_cfg), train_cfg, device=device)

    x = torch.randn(16, feature_dim)
    y = torch.tensor([0, 1] * 8)  # Only 2 out of 10 classes present -> coverage = 0.2

    present_classes = torch.zeros(num_classes, dtype=torch.bool)
    present_classes[0] = True
    present_classes[1] = True

    metrics = agent.train_student_step(
        x, y, present_classes=present_classes, round_num=1
    )

    # Coverage gap = 1 - 0.2 = 0.8 -> anchor_weight = 2.0 * 0.8 = 1.6
    assert metrics["student_anchor_weight"] > 0.0
    assert metrics["student_anchor_loss"] >= -1e-6


def test_full_coverage_zero_anchor():
    num_classes = 4
    feature_dim = 20
    model_cfg = OmegaConf.create({"feature_dim": feature_dim, "latent_dim": 16, "num_classes": num_classes})
    train_cfg = OmegaConf.create(
        {
            "fedtros_global_anchor_weight": 2.0,
            "fedtros_global_anchor_min_weight": 0.0,
            "fedtros_global_anchor_coverage_power": 1.0,
            "teacher_beta_kl": 0.01,
            "lambda_kd_init": 0.20,
            "lambda_align_init": 0.08,
        }
    )

    device = torch.device("cpu")
    agent = Agent(ModelFactory(model_cfg), train_cfg, device=device)

    x = torch.randn(16, feature_dim)
    y = torch.randint(0, num_classes, (16,))
    present_classes = torch.ones(num_classes, dtype=torch.bool)  # All classes present -> coverage = 1.0

    metrics = agent.train_student_step(
        x, y, present_classes=present_classes, round_num=1
    )

    # Coverage gap = 1 - 1.0 = 0.0 -> anchor_weight = 0.0
    assert abs(metrics["student_anchor_weight"]) < 1e-6
