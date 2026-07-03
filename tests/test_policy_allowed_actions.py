import random

import numpy as np
import torch

from src.agents.policy import EpsilonGreedyPolicy


class _Prior(torch.nn.Module):
    def forward(self, state):
        return torch.zeros(state.shape[0], 2, device=state.device), torch.zeros(
            state.shape[0], 2, device=state.device
        )


class _Q(torch.nn.Module):
    def forward(self, z, state):
        _ = z, state
        return torch.tensor([[10.0, 1.0, 8.0, 7.0]], device=state.device)


def test_epsilon_random_selection_only_uses_allowed_actions():
    random.seed(123)
    policy = EpsilonGreedyPolicy(_Prior(), _Q(), num_actions=4, device=torch.device("cpu"))
    policy.set_allowed_actions([1, 3])

    actions = {policy.select_action(np.zeros(3, dtype=np.float32), epsilon=1.0) for _ in range(100)}

    assert actions <= {1, 3}
    assert actions == {1, 3}


def test_greedy_selection_masks_disallowed_actions():
    policy = EpsilonGreedyPolicy(_Prior(), _Q(), num_actions=4, device=torch.device("cpu"))
    policy.set_allowed_actions([1, 3])

    action = policy.select_action(np.zeros(3, dtype=np.float32), epsilon=0.0)

    assert action == 3
