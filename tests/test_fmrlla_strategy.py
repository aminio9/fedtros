import torch

from src.federated.selection_utils import gate_utility, select_utility_records
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


def test_fmrlla_gates_low_utilities_and_selects_only_nonzero_clients():
    assert (
        gate_utility(
            0.05,
            utility_temperature=1.0,
            max_utility=10.0,
            utility_threshold=0.1,
        )
        == 0.0
    )
    assert (
        gate_utility(
            0.20,
            utility_temperature=1.0,
            max_utility=10.0,
            utility_threshold=0.1,
        )
        == 0.20
    )

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
