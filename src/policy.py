import numpy as np
import torch
import random
import logging
from omegaconf import DictConfig

try:
    from .models import PriorNetwork, MainQNetwork
    from .utils import reparameterization_trick
except ImportError:
    from models import PriorNetwork, MainQNetwork
    from utils import reparameterization_trick

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
    def __init__(self, prior_net: PriorNetwork, q_net: MainQNetwork, num_actions: int, device: torch.device):
        self.prior_net = prior_net
        self.q_net = q_net
        self.num_actions = num_actions
        self.device = device

    @torch.no_grad()
    def select_action(self, state_np: np.ndarray, epsilon: float) -> int:
        """
        Select an action using the epsilon-greedy policy.
        Uses the PriorNetwork to get z, as in paper's logic.
        """
        # 1. With prob epsilon, explore
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        # 2. Otherwise, exploit (greedy action)
        state = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get Q-values
        self.prior_net.eval() # Set to eval mode for inference
        self.q_net.eval()     # Set to eval mode for inference
        
        mu_p, log_var_p = self.prior_net(state)
        z_sample = reparameterization_trick(mu_p, log_var_p)
        q_values = self.q_net(z_sample, state)
        
        greedy_action = int(q_values.argmax(dim=1).item())
        
        return greedy_action
