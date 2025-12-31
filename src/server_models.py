import torch
import torch.nn as nn
import torch.nn.functional as F


class AsyncCritic(nn.Module):
    """
    Asynchronous Critic C_i(h_i, r, r_hist, td, recon) -> w_i
    
    Inputs:
        h_i:   Latent State (Context)
        r:     Recent Reward
        r_hist: Historical Reward
        td:    TD Error (Learning Potential)
        recon: Utility Signal (Reconstruction Loss OR Batch F1 Score)
    """

    def __init__(self, hidden_dim: int, latent_dim: int):
        super(AsyncCritic, self).__init__()
        # Input: Latent(z) + Reward(1) + HistReward(1) + TD_Error(1) + Recon/F1(1)
        # Total extra inputs = 4
        self.input_dim = latent_dim + 4

        # LayerNorm is vital for normalizing mixed signals (e.g., small F1 vs large Reward)
        self.ln1 = nn.LayerNorm(self.input_dim)

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, h_i, r_curr, r_hist, td_err, recon_signal):
        # 1. Unsqueeze scalars if they are 1D [Batch] -> [Batch, 1]
        if h_i.dim() == 1: h_i = h_i.unsqueeze(0)
        if r_curr.dim() == 1: r_curr = r_curr.unsqueeze(0)
        if r_hist.dim() == 1: r_hist = r_hist.unsqueeze(0)
        if td_err.dim() == 1: td_err = td_err.unsqueeze(0)
        if recon_signal.dim() == 1: recon_signal = recon_signal.unsqueeze(0)

        # 2. Concatenate all 5 signals
        x = torch.cat([h_i, r_curr, r_hist, td_err, recon_signal], dim=1)

        # 3. Network Body
        x = self.ln1(x)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)

        # 4. Output (BOUNDED)
        # CHANGED: Softplus -> Sigmoid
        # This forces weights to be between 0.0 and 1.0.
        # This prevents the "Utility Explosion" (values going to 12.0+) seen in logs.
        w_i = torch.sigmoid(self.fc3(x))
        
        return w_i


class CentralizedAggregator(nn.Module):
    """
    Learns to mix local utilities/weights into a global system utility Q_tot.
    """

    def __init__(self, num_agents: int, state_dim: int):
        super(CentralizedAggregator, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim

        # Hypernetwork to generate mixing weights from global state
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
        # Generate mixing weights
        mix_weights = torch.abs(self.hyper_w(global_state))

        # Reshape for dot product: [Batch, 1, Agents] x [Batch, Agents, 1]
        w_i_list = w_i_list.view(-1, 1, self.num_agents)
        mix_weights = mix_weights.view(-1, self.num_agents, 1)

        # Mix: q_tot
        q_tot = torch.bmm(w_i_list, mix_weights)
        q_tot = q_tot.view(-1, 1)

        # Bias: v
        v = self.val(global_state)

        return q_tot + v