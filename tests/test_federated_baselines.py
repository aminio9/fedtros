"""Tests for Student-based Federated Baselines (FedAvg-Student and FedProx-Student)."""

import torch
from omegaconf import OmegaConf

from src.models.bundle import FedTROSModelBundle as Agent
from src.models.models import ModelFactory
from src.training.local_training import run_local_training_round


def test_fedavg_student_training():
    features = torch.randn(20, 10)
    labels = torch.randint(0, 4, (20,))
    model_cfg = OmegaConf.create({"feature_dim": 10, "latent_dim": 8, "num_classes": 4})
    train_cfg = OmegaConf.create(
        {
            "student_lr": 1e-3,
            "local_epochs": 1,
            "batch_size": 10,
        }
    )
    device = torch.device("cpu")
    agent = Agent(ModelFactory(model_cfg), train_cfg, device=device)

    steps, metrics = run_local_training_round(
        agent=agent,
        features=features,
        labels=labels,
        cfg_training=train_cfg,
        device=device,
        proximal_mu=0.0,
        is_fedtros=False,
    )

    assert steps > 0
    assert "train_loss" in metrics
    assert metrics["is_standard_baseline"] == 1.0


def test_fedprox_student_proximal_penalty():
    features = torch.randn(20, 10)
    labels = torch.randint(0, 4, (20,))
    model_cfg = OmegaConf.create({"feature_dim": 10, "latent_dim": 8, "num_classes": 4})
    train_cfg = OmegaConf.create(
        {
            "student_lr": 1e-3,
            "local_epochs": 2,
            "batch_size": 10,
        }
    )
    device = torch.device("cpu")
    agent = Agent(ModelFactory(model_cfg), train_cfg, device=device)

    # Modify student weights to create drift from proximal reference
    with torch.no_grad():
        for p in agent.student_model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    penalty = agent._proximal_penalty()
    assert penalty.item() > 0.0

    steps, metrics = run_local_training_round(
        agent=agent,
        features=features,
        labels=labels,
        cfg_training=train_cfg,
        device=device,
        proximal_mu=0.05,
        is_fedtros=False,
    )

    assert steps > 0
    assert metrics["proximal_mu"] == 0.05
