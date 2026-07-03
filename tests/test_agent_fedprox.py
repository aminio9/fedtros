import numpy as np
import torch
from omegaconf import OmegaConf

from src.agents.agent import Agent
<<<<<<< HEAD
from src.models.models import OpenSetQChainModelFactory


def _model_cfg():
    return OmegaConf.create({"state_dim": 5, "latent_dim": 3, "num_actions": 4})
=======
from src.models.cvae_dqn import OpenSetQChainModelFactory


def _model_cfg():
    return OmegaConf.create({"state_dim": 5, "latent_dim": 3, "num_actions": 4, "backbone": {"hidden_dim": 8, "depth": 1, "ensemble_size": 1, "dropout": 0.0, "expansion": 1, "q_hidden_dim": 8, "q_depth": 1, "q_ensemble_size": 1}, "generator": {"hidden_dim": 8, "depth": 1, "dropout": 0.0, "expansion": 1}})
>>>>>>> ea28efe (Initial commit with updated source code)


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

<<<<<<< HEAD
    td_loss, kl_loss, prox_loss, avg_q = agent.train_step(batch, proximal_mu=0.1)

    assert td_loss >= 0.0
    assert kl_loss >= 0.0
    assert prox_loss > 0.0
    assert np.isfinite(avg_q)
=======
    metrics = agent.train_step(batch, proximal_mu=0.1)

    assert metrics["loss/q_td"] >= 0.0
    assert metrics["loss/prior_kl"] >= 0.0
    assert metrics["loss/proximal"] > 0.0
    assert np.isfinite(metrics["q/pred_mean"])
>>>>>>> ea28efe (Initial commit with updated source code)
