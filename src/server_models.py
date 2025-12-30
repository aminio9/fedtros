import torch
import torch.nn as nn
import torch.nn.functional as F

class AsyncCritic(nn.Module):
    """
    Asynchronous Critic C_i(h_i, r_curr, r_hist, recon, td_err) -> w_i
    """

    def __init__(self, hidden_dim: int, latent_dim: int):
        super(AsyncCritic, self).__init__()
        
        # Input Dimensions Breakdown:
        # 1. Latent Vector (h_i)      = latent_dim
        # 2. Recent Reward (r_curr)   = 1
        # 3. History Reward (r_hist)  = 1
        # 4. Recon Loss (Novelty)     = 1 
        # 5. TD Error (RL Surprise)   = 1
        self.input_dim = latent_dim + 4  # Total = 32 + 4 = 36

        self.ln1 = nn.LayerNorm(self.input_dim)
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, h_i, r_curr, r_hist, recon_loss, td_error):
        # 1. Unsqueeze scalars if 1D
        if h_i.dim() == 1: h_i = h_i.unsqueeze(0)
        if r_curr.dim() == 1: r_curr = r_curr.unsqueeze(0)
        if r_hist.dim() == 1: r_hist = r_hist.unsqueeze(0)
        if recon_loss.dim() == 1: recon_loss = recon_loss.unsqueeze(0)
        if td_error.dim() == 1: td_error = td_error.unsqueeze(0)

        # 2. Concatenate [State | Performance | Context | Novelty | Learning]
        x = torch.cat([h_i, r_curr, r_hist, recon_loss, td_error], dim=1)

        x = self.ln1(x)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        w_i = F.softplus(self.fc3(x))
        
        return w_i
    
class CentralizedAggregator(nn.Module):
    """
    Learns to mix local utilities/weights into a global system utility Q_tot.
    """

    def __init__(self, num_agents: int, state_dim: int):
        super(CentralizedAggregator, self).__init__()
        self.num_agents = num_agents
        
        # NOTE: state_dim here refers to the dimension of the GLOBAL state.
        # In this architecture, that is usually the size of one client's latent vector
        # (assuming the server averages them to get a global view).
        self.state_dim = state_dim

        # Hypernetwork to generate weights for mixing
        self.hyper_w = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(state_dim, num_agents),
        )

        # Value estimator (Bias term)
        self.val = nn.Sequential(
            nn.Linear(state_dim, 64), 
            nn.LeakyReLU(0.01), 
            nn.Linear(64, 1)
        )

    def forward(self, w_i_list, global_state):
        """
        w_i_list: Tensor of shape [Batch, num_agents] containing the weights from AsyncCritics
        global_state: Tensor [Batch, state_dim] (aggregated from clients)
        """
        # Generate mixing weights from global state
        # We use absolute value to ensure mixing weights are positive
        mix_weights = torch.abs(self.hyper_w(global_state))

        # Reshape for dot product
        # w_i_list: [Batch, 1, Agents]
        w_i_list = w_i_list.view(-1, 1, self.num_agents)
        # mix_weights: [Batch, Agents, 1]
        mix_weights = mix_weights.view(-1, self.num_agents, 1)

        # Mix: q_tot (The weighted sum of client utilities)
        q_tot = torch.bmm(w_i_list, mix_weights)
        q_tot = q_tot.view(-1, 1)

        # Bias: v (System-level value expectation)
        v = self.val(global_state)

        return q_tot + v