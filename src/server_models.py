import torch
import torch.nn as nn
import torch.nn.functional as F

class AsyncCritic(nn.Module):
    """
    Asynchronous Critic with RESIDUAL LOGIC INJECTION.
    
    Paper Concept: "Residual Control". 
    We force the network to respect the 'competence' signal (F1/Recon) 
    by adding it directly to the learned output.
    """
    def __init__(self, hidden_dim: int, latent_dim: int):
        super(AsyncCritic, self).__init__()
        # Inputs: Latent(z) + Reward(1) + HistReward(1) + TD(1) + F1(1)
        self.input_dim = latent_dim + 4

        # LayerNorm is CRITICAL for stability
        self.ln1 = nn.LayerNorm(self.input_dim)

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Dropout prevents overfitting to "Lazy Agents"
        self.dropout = nn.Dropout(0.1)

    def forward(self, h_i, r_curr, r_hist, td_err, recon_signal):
        # 1. Unsqueeze if necessary
        if h_i.dim() == 1: h_i = h_i.unsqueeze(0)
        if r_curr.dim() == 1: r_curr = r_curr.unsqueeze(0)
        if r_hist.dim() == 1: r_hist = r_hist.unsqueeze(0)
        if td_err.dim() == 1: td_err = td_err.unsqueeze(0)
        if recon_signal.dim() == 1: recon_signal = recon_signal.unsqueeze(0)

        # 2. Concatenate
        x = torch.cat([h_i, r_curr, r_hist, td_err, recon_signal], dim=1)

        # 3. Neural Processing (The "Gut Feeling")
        x = self.ln1(x)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.01)
        
        # Learned "Correction" (Bounded 0-1)
        learned_val = torch.sigmoid(self.fc3(x))

        # 4. RESIDUAL LOGIC INJECTION (The "Better Idea")
        # w = 50% Learned Opinion + 50% Hard Fact (F1 Score)
        # This prevents "Lazy Agents" (Low F1) from ever dominating,
        # even if they have high Rewards.
        w_final = 0.5 * learned_val + 0.5 * recon_signal
        
        return w_final

class CentralizedAggregator(nn.Module):
    """
    Standard Aggregator with Softmax for relative importance.
    """
    def __init__(self, num_agents: int, state_dim: int):
        super(CentralizedAggregator, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim

        self.hyper_w = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(state_dim // 2, num_agents),
        )

        self.val = nn.Sequential(
            nn.Linear(state_dim, 64), nn.LeakyReLU(0.01), nn.Linear(64, 1)
        )

    def forward(self, w_i_list, global_state):
        # Softmax ensures weights sum to 1 naturally
        # This prevents "Utility Explosion" (12.0 -> 100.0)
        mix_weights = F.softmax(self.hyper_w(global_state), dim=1)

        w_i_list = w_i_list.view(-1, 1, self.num_agents)
        mix_weights = mix_weights.view(-1, self.num_agents, 1)

        q_tot = torch.bmm(w_i_list, mix_weights)
        q_tot = q_tot.view(-1, 1)
        v = self.val(global_state)
        
        return q_tot + v