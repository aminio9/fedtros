import torch

from src.federated.selection_utils import (
    centered_utility,
    combine_utility_score,
    critic_utility_score,
    select_utility_records,
)
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


def test_fmrlla_utility_is_fedavg_neutral_for_iid_like_scores():
    utilities = [
        centered_utility(
            score=0.55,
            round_mean_score=0.55,
            utility_strength=0.75,
            min_utility=0.25,
            max_utility=2.0,
            utility_threshold=0.1,
        )
        for _ in range(3)
    ]

    assert utilities == [1.0, 1.0, 1.0]


def test_fmrlla_downweights_or_drops_relative_low_quality_clients():
    low = centered_utility(
        score=0.05,
        round_mean_score=0.50,
        utility_strength=0.75,
        min_utility=0.25,
        max_utility=2.0,
        utility_threshold=0.1,
    )
    high = centered_utility(
        score=0.80,
        round_mean_score=0.50,
        utility_strength=0.75,
        min_utility=0.25,
        max_utility=2.0,
        utility_threshold=0.1,
    )

    assert low == 0.0
    assert high > 1.0


def test_fmrlla_combines_audit_score_with_bounded_critic_residual():
    critic_score = critic_utility_score(1.0, utility_temperature=1.0)
    combined = combine_utility_score(
        audit_score=0.8,
        critic_score=critic_score,
        critic_blend=0.15,
    )

    assert 0.0 <= critic_score <= 1.0
    assert 0.5 < combined < 0.8


def test_fmrlla_selects_only_nonzero_utility_clients():
    selection_records = [
        {"cid": "1", "utility": 0.25, "selected": False},
        {"cid": "2", "utility": 0.00, "selected": False},
        {"cid": "3", "utility": 0.10, "selected": False},
        {"cid": "4", "utility": 0.00, "selected": False},
    ]

    selected = select_utility_records(
        selection_records,
        server_round=2,
        min_selected_clients=1,
        max_selected_fraction=1.0,
        warmup_rounds=0,
    )

    assert [record["cid"] for record in selected] == ["1", "3"]
    assert [record["selected"] for record in selection_records] == [
        True,
        False,
        True,
        False,
    ]
