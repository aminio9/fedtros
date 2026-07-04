import numpy as np
import torch
from omegaconf import OmegaConf

from src.agents.agent import Agent
from src.models.models import OpenSetQChainModelFactory


def _model_cfg():
    return OmegaConf.create({"state_dim": 5, "latent_dim": 3, "num_actions": 4})


def _training_cfg():
    return OmegaConf.create(
        {
            "gamma": 0.7,
            "use_double_dqn": True,
            "lr_prior": 1e-3,
            "lr_q_rl": 1e-3,
            "prior_grad_clip_norm": 1.0,
            "prior_kl_raw": False,
        }
    )


def test_train_step_reports_fedprox_penalty_after_local_drift():
    factory = OpenSetQChainModelFactory(_model_cfg())
    agent = Agent(factory, _training_cfg(), torch.device("cpu"))
    agent.set_federated_parameters(agent.get_federated_parameters())

    with torch.no_grad():
        next(agent.value_net_main.parameters()).add_(0.5)

    batch_size = 4
    batch = (
        torch.randn(batch_size, 5),
        torch.zeros(batch_size, 1, dtype=torch.long),
        torch.ones(batch_size, 1),
        torch.randn(batch_size, 5),
        torch.zeros(batch_size, 1),
        torch.zeros(batch_size, 1, dtype=torch.long),
    )

    td_loss, kl_loss, prox_loss, avg_q = agent.train_step(batch, proximal_mu=0.1)

    assert td_loss >= 0.0
    assert kl_loss >= 0.0
    assert prox_loss > 0.0
    assert np.isfinite(avg_q)


def test_train_step_records_auxiliary_ce_loss():
    factory = OpenSetQChainModelFactory(_model_cfg())
    agent = Agent(factory, _training_cfg(), torch.device("cpu"))

    batch_size = 4
    batch = (
        torch.randn(batch_size, 5),
        torch.zeros(batch_size, 1, dtype=torch.long),
        torch.ones(batch_size, 1),
        torch.randn(batch_size, 5),
        torch.zeros(batch_size, 1),
        torch.tensor([[0], [1], [2], [3]], dtype=torch.long),
    )

    td_loss, kl_loss, prox_loss, avg_q = agent.train_step(
        batch,
        aux_ce_weight=0.1,
        aux_ce_label_smoothing=0.01,
        class_weights=torch.ones(4),
    )

    assert td_loss >= 0.0
    assert kl_loss >= 0.0
    assert prox_loss >= 0.0
    assert agent.last_aux_ce_loss > 0.0
    assert np.isfinite(avg_q)
