import numpy as np
import torch
from omegaconf import OmegaConf

from src.models.bundle import FedTROSModelBundle
from src.models.models import FedTROSModelFactory


def _model_cfg():
    return OmegaConf.create({"feature_dim": 5, "latent_dim": 3, "num_classes": 4})


def _training_cfg():
    return OmegaConf.create(
        {
            "teacher_lr": 1e-3,
            "student_lr": 1e-3,
            "teacher_beta_kl": 0.01,
            "lambda_kd_init": 0.20,
            "lambda_align_init": 0.08,
            "fedtros_global_anchor_weight": 2.0,
        }
    )


def test_student_proximal_penalty_after_local_drift():
    factory = FedTROSModelFactory(_model_cfg())
    bundle = FedTROSModelBundle(factory, _training_cfg(), torch.device("cpu"))
    bundle.set_federated_parameters(bundle.get_federated_parameters())

    with torch.no_grad():
        for p in bundle.student_model.parameters():
            p.add_(0.5)

    prox_penalty = bundle._proximal_penalty()
    assert prox_penalty.item() > 0.0
