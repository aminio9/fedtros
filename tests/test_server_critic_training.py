import torch
from omegaconf import OmegaConf

from src.federated.server import AUDIT_SCALAR_KEYS, FMRLAdaptiveVectorAlignedAggregationStrategy
from src.federated.server_models import AsyncCritic, CentralizedAggregator


def test_server_critic_training_updates_critic_parameters():
    strategy = object.__new__(FMRLAdaptiveVectorAlignedAggregationStrategy)
    strategy.device = torch.device("cpu")
    strategy.latent_dim = 3
    strategy.scalar_dim = len(AUDIT_SCALAR_KEYS)
    strategy.max_agents = 2
    strategy.client_feature_dim = strategy.latent_dim + strategy.scalar_dim
    strategy.state_dim = strategy.client_feature_dim * strategy.max_agents
    strategy.utility_temperature = 1.0
    strategy._AsyncCriticClass = AsyncCritic
    strategy.critics = {}
    strategy.cfg = OmegaConf.create(
        {"strategy": {"critic_hidden_dim": 8}, "model": {"num_actions": 5}}
    )
    strategy.aggregator = CentralizedAggregator(
        num_agents=strategy.max_agents,
        state_dim=strategy.state_dim,
        hidden_dim=8,
        hyper_hidden_dim=8,
    ).to(strategy.device)
    strategy.optimizer = torch.optim.Adam(strategy.aggregator.parameters(), lr=1e-2)
    strategy.stage1_data_cache = {}
    strategy.selection_records = []
    strategy.client_order = ["1", "2"]
    strategy.last_team_reward_target = 0.25
    strategy.validation_team_reward_ema = 0.5
    strategy.last_support_reward = 0.4

    for idx, cid in enumerate(strategy.client_order):
        h = torch.randn(1, strategy.latent_dim)
        scalars = torch.rand(1, strategy.scalar_dim)
        feature = torch.cat([h, scalars], dim=1)
        strategy.stage1_data_cache[cid] = {"h": h, "scalars": scalars, "feature": feature}
        strategy.selection_records.append(
            {"cid": cid, "selected": idx == 0, "audit_score": 0.35 + 0.1 * idx}
        )

    critic = strategy._get_critic("1")
    before = [param.detach().clone() for param in critic.parameters()]

    metrics = strategy._train_server_models(0.65)

    after = list(critic.parameters())
    assert "fmrl_ava_critic_loss" in metrics
    assert metrics["fmrl_ava_critic_loss"] >= 0.0
    assert any(not torch.allclose(old, new) for old, new in zip(before, after, strict=True))
