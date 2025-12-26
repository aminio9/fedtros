import torch
import torch.nn as nn
import torch.nn.functional as F


class AsyncCritic(nn.Module):
    """
    Asynchronous Critic C_i(h_i, r, r_hist) -> w_i
    Improved to handle negative rewards.
    """

    def __init__(self, hidden_dim: int, latent_dim: int):
        super(AsyncCritic, self).__init__()
        # Input: Hidden State (latent_dim) + Recent Reward (1) + History Reward (1)
        self.input_dim = latent_dim + 2

        # Use LayerNorm to stabilize inputs (crucial for RL)
        self.ln1 = nn.LayerNorm(self.input_dim)

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, h_i, r_curr, r_hist):
        # Ensure inputs are at least 2D [Batch, Dim]
        if h_i.dim() == 1:
            h_i = h_i.unsqueeze(0)
        if r_curr.dim() == 1:
            r_curr = r_curr.unsqueeze(0)
        if r_hist.dim() == 1:
            r_hist = r_hist.unsqueeze(0)

        # Concatenate
        x = torch.cat([h_i, r_curr, r_hist], dim=1)

        # Normalize inputs so -60 doesn't break the network
        x = self.ln1(x)

        # Use LeakyReLU: allows negative gradients to flow
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)

        # Softplus ensures positive weight w_i >= 0
        w_i = F.softplus(self.fc3(x))
        return w_i


class CentralizedAggregator(nn.Module):
    """
    Learns to mix local utilities/weights into a global system utility Q_tot.
    """

    def __init__(self, num_agents: int, state_dim: int):
        super(CentralizedAggregator, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim

        # Hypernetwork to generate weights for mixing
        self.hyper_w = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(state_dim, num_agents),
        )

        # Value estimator (Bias term)
        self.val = nn.Sequential(
            nn.Linear(state_dim, 64), nn.LeakyReLU(0.01), nn.Linear(64, 1)
        )

    def forward(self, w_i_list, global_state):
        # Generate mixing weights from global state
        mix_weights = torch.abs(self.hyper_w(global_state))

        # Reshape for dot product
        w_i_list = w_i_list.view(-1, 1, self.num_agents)
        mix_weights = mix_weights.view(-1, self.num_agents, 1)

        # Mix: q_tot (Positive Component)
        q_tot = torch.bmm(w_i_list, mix_weights)
        q_tot = q_tot.view(-1, 1)

        # Bias: v (Can be negative to handle negative rewards)
        v = self.val(global_state)

        return q_tot + v
