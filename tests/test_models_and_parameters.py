import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from omegaconf import OmegaConf

from src.agents.agent import Agent
<<<<<<< HEAD
from src.models.models import OpenSetQChainModelFactory


def _model_cfg():
    return OmegaConf.create({"state_dim": 5, "latent_dim": 3, "num_actions": 4})
=======
from src.models.cvae_dqn import FastTabMBackbone, GenerationNetwork, OpenSetQChainModelFactory


def _model_cfg():
    return OmegaConf.create(
        {
            "state_dim": 5,
            "latent_dim": 3,
            "num_actions": 4,
            "backbone": {
                "hidden_dim": 16,
                "depth": 1,
                "ensemble_size": 2,
                "dropout": 0.0,
                "expansion": 2,
                "q_hidden_dim": 16,
                "q_depth": 1,
                "q_ensemble_size": 2,
            },
            "generator": {"hidden_dim": 12, "depth": 1, "dropout": 0.0},
        }
    )
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


<<<<<<< HEAD
=======
def test_fast_tabm_backbone_shape_contract():
    backbone = FastTabMBackbone(
        input_dim=5,
        hidden_dim=8,
        depth=1,
        ensemble_size=2,
        dropout=0.0,
    )

    out = backbone(torch.randn(3, 5))

    assert out.shape == (3, 8)


def test_generation_network_shape_contract():
    generator = GenerationNetwork(
        z_dim=3,
        num_actions=4,
        s_dim=5,
        generator_cfg=OmegaConf.create({"hidden_dim": 10, "depth": 1, "dropout": 0.0}),
    )

    recon = generator(torch.randn(6, 3), torch.tensor([0, 1, 2, 3, 0, 1]))

    assert recon.shape == (6, 5)


>>>>>>> ea28efe (Initial commit with updated source code)
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
