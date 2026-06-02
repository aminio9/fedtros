import torch
import torch.nn as nn
import torch.nn.functional as F


class AsyncCritic(nn.Module):
    """
    Per-client utility estimator for FMRL-LA.

    The paper defines one asynchronous critic per agent:
        w_i^k = C_i(h_i^k, r_i^k, history_r_i^k)

    This project has richer local diagnostics than the paper's driving hidden
    state, so the critic consumes the latent state plus scalar diagnostics derived
    from reward, local accuracy/F1, TD stability, novelty, and data coverage.
    The output is a utility score; the server clips low utilities to zero before
    communication selection so that zero means "do not upload" in that round.
    """

    def __init__(self, hidden_dim: int, latent_dim: int, scalar_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.scalar_dim = int(scalar_dim)
        self.input_dim = self.latent_dim + self.scalar_dim

        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_state: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
        if scalar_features.dim() == 1:
            scalar_features = scalar_features.unsqueeze(0)

        x = torch.cat([hidden_state, scalar_features], dim=-1)
        return F.softplus(self.net(x))


class CentralizedAggregator(nn.Module):
    """
    QMIX-style monotonic mixer for learnable aggregation.

    The mixer maps per-client utilities to a scalar system utility conditioned on
    a padded global client-state vector. Hypernetworks generate nonnegative
    mixing weights, preserving monotonicity in each client utility while allowing
    the server to model non-i.i.d. interactions between clients.
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        hidden_dim: int = 64,
        hyper_hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_agents = int(num_agents)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)

        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, self.num_agents * self.hidden_dim),
        )
        self.hyper_b1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, self.hidden_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, utilities: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        if utilities.dim() == 1:
            utilities = utilities.unsqueeze(0)
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)

        batch_size = utilities.shape[0]
        utilities = utilities[:, : self.num_agents]
        if utilities.shape[1] < self.num_agents:
            utilities = F.pad(utilities, (0, self.num_agents - utilities.shape[1]))

        global_state = global_state[:, : self.state_dim]
        if global_state.shape[1] < self.state_dim:
            global_state = F.pad(global_state, (0, self.state_dim - global_state.shape[1]))

        w1 = torch.abs(self.hyper_w1(global_state)).view(
            batch_size, self.num_agents, self.hidden_dim
        )
        b1 = self.hyper_b1(global_state).view(batch_size, 1, self.hidden_dim)
        hidden = F.elu(torch.bmm(utilities.unsqueeze(1), w1) + b1)

        w2 = torch.abs(self.hyper_w2(global_state)).view(batch_size, self.hidden_dim, 1)
        b2 = self.hyper_b2(global_state).view(batch_size, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.view(batch_size, 1)
