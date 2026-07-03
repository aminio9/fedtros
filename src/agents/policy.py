import logging
import random

import numpy as np
import torch
from omegaconf import DictConfig

<<<<<<< HEAD
from src.models.models import MainQNetwork, PriorNetwork
=======
from src.models.cvae_dqn import MainQNetwork, PriorNetwork
>>>>>>> ea28efe (Initial commit with updated source code)
from src.utils.utils import reparameterization_trick

logger = logging.getLogger("Policy")


class EpsilonScheduler:
    """Manages the decay of epsilon over time (episodes)."""

    def __init__(self, cfg: DictConfig):
        self.epsilon = cfg.epsilon_start
        self.min_epsilon = cfg.epsilon_end
        self.decay_rate = cfg.epsilon_decay_rate
        logger.debug(
            f"EpsilonScheduler init: start={self.epsilon}, "
            f"end={self.min_epsilon}, rate={self.decay_rate}"
        )

    def get_epsilon(self) -> float:
        """Get the current epsilon value."""
        return self.epsilon

    def step(self):
        """Decays epsilon one step (call per episode)."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)


class EpsilonGreedyPolicy:
    """Implements the standard epsilon-greedy action selection strategy."""

    def __init__(
        self,
        prior_net: PriorNetwork,
        q_net: MainQNetwork,
        num_actions: int,
        device: torch.device,
    ):
        self.prior_net = prior_net
        self.q_net = q_net
        self.num_actions = num_actions
        self.device = device
<<<<<<< HEAD
=======
        self.allowed_actions: list[int] | None = None

    def set_allowed_actions(self, actions: list[int]) -> None:
        clean = sorted({int(action) for action in actions if 0 <= int(action) < self.num_actions})
        self.allowed_actions = clean if clean else None
>>>>>>> ea28efe (Initial commit with updated source code)

    @torch.no_grad()
    def select_action(self, state_np: np.ndarray, epsilon: float) -> int:
        """
        Select an action using the epsilon-greedy policy.
        Uses the PriorNetwork to get z, as in paper's logic.
        """
        # 1. With prob epsilon, explore
        if random.random() < epsilon:
<<<<<<< HEAD
=======
            if self.allowed_actions:
                return int(random.choice(self.allowed_actions))
>>>>>>> ea28efe (Initial commit with updated source code)
            return random.randint(0, self.num_actions - 1)

        # 2. Otherwise, exploit (greedy action)
        state = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)

<<<<<<< HEAD
        # Get Q-values
        self.prior_net.eval()  # Set to eval mode for inference
        self.q_net.eval()  # Set to eval mode for inference

        mu_p, log_var_p = self.prior_net(state)
        z_sample = reparameterization_trick(mu_p, log_var_p)
        q_values = self.q_net(z_sample, state)

        return int(q_values.argmax(dim=1).item())
=======
        prior_was_training = self.prior_net.training
        q_was_training = self.q_net.training

        try:
            self.prior_net.eval()
            self.q_net.eval()

            mu_p, log_var_p = self.prior_net(state)
            z_sample = reparameterization_trick(mu_p, log_var_p)
            q_values = self.q_net(z_sample, state)
            if self.allowed_actions:
                mask = torch.full_like(q_values, float("-inf"))
                mask[:, self.allowed_actions] = 0.0
                q_values = q_values + mask
            return int(q_values.argmax(dim=1).item())
        finally:
            self.prior_net.train(prior_was_training)
            self.q_net.train(q_was_training)
>>>>>>> ea28efe (Initial commit with updated source code)
