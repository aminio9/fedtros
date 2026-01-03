import torch
import torch.nn as nn
import torch.nn.functional as F

class AsyncCritic(nn.Module):
    """
    NOVELTY-FIRST CRITIC: Prioritizes KL Divergence.
    
    Structure:
    Score = Neural_Net(All_Inputs) + (Importance_Scalar * KL_Divergence)
    
    This 'Skip Connection' guarantees that if a client has high KL (Novelty),
    their score is heavily boosted immediately, regardless of the neural network's opinion.
    """
    def __init__(self, hidden_dim: int, latent_dim: int):
        super(AsyncCritic, self).__init__()
        # Input Dim: Latent + Reward + Hist + TD + F1 + KL
        self.input_dim = latent_dim + 5 

        self.norm = nn.LayerNorm(self.input_dim)
        
        # 1. The "Brain" (Learns complex tradeoffs between reward and stability)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        # 2. The "Bias" (Enforces KL Importance)
        # We initialize this to 2.0 so KL is immediately 2x more important than other factors.
        # This parameter is LEARNABLE, so the server can tune it over time.
        self.kl_importance = nn.Parameter(torch.tensor(2.0))

    def forward(self, h_i, r_curr, r_hist, td_err, f1_score, kl_div):
        # 1. Unsqueeze logic
        if h_i.dim() == 1: h_i = h_i.unsqueeze(0)
        if r_curr.dim() == 1: r_curr = r_curr.unsqueeze(1)
        if r_hist.dim() == 1: r_hist = r_hist.unsqueeze(1)
        if td_err.dim() == 1: td_err = td_err.unsqueeze(1)
        if f1_score.dim() == 1: f1_score = f1_score.unsqueeze(1)
        if kl_div.dim() == 1: kl_div = kl_div.unsqueeze(1)

        # 2. Neural Path (Standard Analysis)
        features = torch.cat([
            h_i, 
            r_curr * 0.01, 
            r_hist * 0.01, 
            td_err, 
            f1_score,
            kl_div * 0.1 # Scaled down for the NN input so it doesn't overwhelm gradients
        ], dim=1)
        
        norm_features = self.norm(features)
        neural_score = self.net(norm_features)
        
        # 3. NOVELTY BOOST (The "Skip Connection")
        # We add raw KL (multiplied by importance) directly to the logits.
        # Even if neural_score is low, a high KL will force total_score high.
        novelty_boost = kl_div * self.kl_importance
        
        total_score = neural_score + novelty_boost
        
        return total_score


class CentralizedAggregator(nn.Module):
    """
    Identity-Aware Aggregator.
    
    Because 'w_i' now heavily encodes KL Divergence, this Aggregator will
    naturally learn to attend to high-KL clients via the 'w_embed' projection.
    """
    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 64):
        super(CentralizedAggregator, self).__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        
        self.state_proj = nn.Linear(state_dim, embed_dim)
        self.weight_proj = nn.Linear(1, embed_dim)
        
        # Learn specific profiles for each client ID
        self.agent_pos_embed = nn.Parameter(torch.randn(1, num_agents, embed_dim))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, w_i_list, global_state):
        # 1. Embed 'w_i' 
        # Since w_i is boosted by KL, high-KL clients get distinct embeddings here.
        w_embed = self.weight_proj(w_i_list.unsqueeze(-1))
        
        # 2. Add Identity (Who sent this update?)
        w_embed = w_embed + self.agent_pos_embed[:, :w_embed.shape[1], :]

        # 3. Contextualize
        client_context = self.transformer_encoder(w_embed)

        # 4. Global Attention
        q = self.state_proj(global_state).unsqueeze(1)
        attn_out, _ = self.cross_attention(query=q, key=client_context, value=client_context)
        
        val_pred = self.head(attn_out.squeeze(1))
        
        return val_pred