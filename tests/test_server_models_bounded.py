import torch

from src.federated.server_models import CentralizedAggregator


def test_centralized_aggregator_output_is_bounded():
    model = CentralizedAggregator(num_agents=3, state_dim=12, hidden_dim=8, hyper_hidden_dim=8)
    utilities = torch.randn(4, 3) * 10
    global_state = torch.randn(4, 12) * 10

    out = model(utilities, global_state)

    assert out.shape == (4, 1)
    assert torch.isfinite(out).all()
    assert bool((out >= 0.0).all())
    assert bool((out <= 1.0).all())
