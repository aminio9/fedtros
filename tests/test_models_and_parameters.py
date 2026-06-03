import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
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


def test_model_forward_shapes():
    factory = OpenSetQChainModelFactory(_model_cfg())
    value_network = factory.create_value_network()
    generation_network = factory.create_generation_network()
    states = torch.randn(2, 5)
    actions = torch.tensor([0, 3])

    mu, logvar = value_network.encoder.prior_forward(states)
    q_values = value_network.decoder.main_q(mu, states)
    recon = generation_network(mu, actions)

    assert mu.shape == (2, 3)
    assert logvar.shape == (2, 3)
    assert q_values.shape == (2, 4)
    assert recon.shape == (2, 5)


def test_prior_and_recognition_use_gated_tabular_encoder():
    factory = OpenSetQChainModelFactory(_model_cfg())
    value_network = factory.create_value_network()

    prior_encoder = value_network.encoder.prior.encoder
    recognition_encoder = value_network.encoder.recognition.encoder

    assert prior_encoder.tokenizer.feature_dim == 5
    assert recognition_encoder.tokenizer.feature_dim == 9
    assert len(prior_encoder.blocks) == 4
    assert len(recognition_encoder.blocks) == 4
    assert prior_encoder.pool.__class__.__name__ == "AttentionPooling"
    assert recognition_encoder.pool.__class__.__name__ == "AttentionPooling"


def test_flower_parameter_roundtrip_sets_agent_weights():
    factory = OpenSetQChainModelFactory(_model_cfg())
    agent = Agent(factory, _training_cfg(), torch.device("cpu"))
    original = agent.get_federated_parameters()

    flower_params = ndarrays_to_parameters(original)
    roundtrip = parameters_to_ndarrays(flower_params)
    agent.set_federated_parameters(roundtrip)
    after = agent.get_federated_parameters()

    assert len(original) == len(after)
    assert all(np.array_equal(a, b) for a, b in zip(roundtrip, after, strict=True))
