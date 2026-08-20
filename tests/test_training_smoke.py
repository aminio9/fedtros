import torch
from omegaconf import OmegaConf

from src.models.bundle import FedTROSModelBundle as Agent
from src.models.models import ModelFactory
from src.training.local_training import run_local_training_round


def test_minimal_local_training_smoke(tmp_path):
    features = torch.randn(8, 5)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    model_cfg = OmegaConf.create({"feature_dim": 5, "latent_dim": 8, "num_classes": 2})
    training_cfg = OmegaConf.create(
        {
            "seed": 7,
            "batch_size": 4,
            "local_epochs": 1,
            "teacher_lr": 1e-3,
            "student_lr": 1e-3,
            "teacher_beta_kl": 0.01,
            "lambda_kd_init": 0.20,
            "lambda_align_init": 0.08,
            "fedtros_global_anchor_weight": 2.0,
        }
    )

    device = torch.device("cpu")
    agent = Agent(ModelFactory(model_cfg), training_cfg, device=device)

    steps, metrics = run_local_training_round(
        agent=agent,
        features=features,
        labels=labels,
        cfg_training=training_cfg,
        device=device,
        round_num=1,
        is_fedtros=True,
    )

    assert steps > 0
    assert "avg_teacher_loss" in metrics
    assert "avg_student_total_loss" in metrics
    assert "agreement" in metrics
