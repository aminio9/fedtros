import torch
from omegaconf import OmegaConf

from src.federated.server import get_effective_num_rounds
from src.federated.server_models import AsyncCritic, CentralizedAggregator


def test_async_critic_outputs_nonnegative_utility():
    critic = AsyncCritic(hidden_dim=8, latent_dim=3, scalar_dim=4)
    utility = critic(torch.randn(2, 3), torch.rand(2, 4))

    assert utility.shape == (2, 1)
    assert torch.all(utility >= 0)


def test_centralized_aggregator_outputs_system_utility():
    mixer = CentralizedAggregator(num_agents=3, state_dim=15, hidden_dim=8)
    q_total = mixer(torch.rand(1, 3), torch.rand(1, 15))

    assert q_total.shape == (1, 1)


def test_fmrlla_uses_two_flower_rounds_per_logical_round():
    cfg = OmegaConf.create(
        {
            "server": {"num_rounds": 5},
            "strategy": {"name": "fmrl_la"},
        }
    )

    assert get_effective_num_rounds(cfg) == 10
