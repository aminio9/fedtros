import numpy as np
import torch
from omegaconf import OmegaConf

from src.models.bundle import FedTROSModelBundle as Agent
from src.models.models import ModelFactory


def _model_cfg():
    return OmegaConf.create({"feature_dim": 5, "latent_dim": 8, "num_classes": 4})


def _training_cfg():
    return OmegaConf.create(
        {
            "teacher_lr": 1e-3,
            "student_lr": 1e-3,
            "teacher_beta_kl": 0.01,
            "teacher_epochs": 1,
            "student_epochs": 1,
            "batch_size": 4,
            "lambda_kd_init": 0.20,
            "lambda_align_init": 0.08,
            "fedtros_global_anchor_weight": 2.0,
            "fedtros_global_anchor_min_weight": 0.0,
            "fedtros_global_anchor_coverage_power": 1.0,
        }
    )


def test_fedtros_train_dataset_updates_both_models():
    factory = ModelFactory(_model_cfg())
    cfg = _training_cfg()
    agent = Agent(factory, cfg, torch.device("cpu"))

    teacher_before = [p.detach().clone() for p in agent.teacher.parameters()]
    student_before = [p.copy() for p in agent.get_student_parameters()]

    features = torch.randn(16, 5)
    labels = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)

    metrics = agent.train_fedtros_dataset(
        features=features,
        labels=labels,
        cfg_training=cfg,
        round_num=1,
        present_classes=torch.ones(4, dtype=torch.bool),
        device=torch.device("cpu"),
    )

    teacher_after = [p.detach().clone() for p in agent.teacher.parameters()]
    student_after = agent.get_student_parameters()

    assert metrics["teacher_train_steps"] > 0
    assert metrics["student_train_steps"] > 0
    assert metrics["avg_teacher_loss"] > 0.0
    assert metrics["avg_student_total_loss"] > 0.0

    # Both models updated locally
    assert any(not torch.allclose(a, b) for a, b in zip(teacher_before, teacher_after, strict=True))
    assert any(not np.allclose(a, b) for a, b in zip(student_before, student_after, strict=True))


def test_one_class_dataset_activates_anchor():
    factory = ModelFactory(_model_cfg())
    cfg = _training_cfg()
    agent = Agent(factory, cfg, torch.device("cpu"))

    features = torch.randn(12, 5)
    labels = torch.zeros(12, dtype=torch.long)  # 1 class out of 4 -> coverage = 0.25

    metrics = agent.train_fedtros_dataset(
        features=features,
        labels=labels,
        cfg_training=cfg,
        round_num=2,
        present_classes=torch.tensor([1, 0, 0, 0], dtype=torch.bool),
        device=torch.device("cpu"),
    )

    assert metrics["avg_student_anchor_loss"] >= 0.0
    assert metrics["student_anchor_weight"] > 0.0

