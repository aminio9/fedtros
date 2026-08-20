import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
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
            "lambda_kd_init": 0.20,
            "lambda_align_init": 0.08,
            "fedtros_global_anchor_weight": 2.0,
        }
    )


def test_model_forward_shapes():
    factory = ModelFactory(_model_cfg())
    teacher = factory.create_teacher()
    student = factory.create_student()
    states = torch.randn(2, 5)

    t_logits, t_mu, t_logvar, t_h = teacher(states, sample=True)
    s_feat, s_logits = student(states)

    assert t_logits.shape == (2, 4)
    assert t_mu.shape == (2, 8)
    assert t_logvar.shape == (2, 8)
    assert s_logits.shape == (2, 4)


def test_flower_parameter_roundtrip_sets_agent_weights():
    factory = ModelFactory(_model_cfg())
    agent = Agent(factory, _training_cfg(), torch.device("cpu"))
    original = agent.get_federated_parameters()

    flower_params = ndarrays_to_parameters(original)
    roundtrip = parameters_to_ndarrays(flower_params)
    agent.set_federated_parameters(roundtrip)
    after = agent.get_federated_parameters()

    assert len(original) == len(after)
    assert all(np.array_equal(a, b) for a, b in zip(roundtrip, after, strict=True))
