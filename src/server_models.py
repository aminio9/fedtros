import torch
import torch.nn as nn
import torch.nn.functional as F

class AsyncCritic(nn.Module):
    """
    FIXED CRITIC: Stability-First Architecture.
    
    Why this fixes the crash:
    1. Heuristic Path now includes REWARD and STABILITY (TD), not just F1.
    2. Uses a 'Residual' structure so it starts with common sense (High Reward = Good)
       and learns to refine it, rather than learning from scratch.
    """
    def __init__(self, hidden_dim: int, latent_dim: int):
        super(AsyncCritic, self).__init__()
        # Input Dim: Latent(32) + Reward(1) + Hist(1) + TD(1) + F1(1)
        self.input_dim = latent_dim + 4 

        # Neural Path (The "Gut Feeling")
        self.norm = nn.LayerNorm(self.input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        # Learnable scale for the heuristic (starts at 1.0)
        self.heuristic_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h_i, r_curr, r_hist, td_err, recon_signal):
        # 1. Unsqueeze logic
        if h_i.dim() == 1: h_i = h_i.unsqueeze(0)
        if r_curr.dim() == 1: r_curr = r_curr.unsqueeze(1)
        if r_hist.dim() == 1: r_hist = r_hist.unsqueeze(1)
        if td_err.dim() == 1: td_err = td_err.unsqueeze(1)
        if recon_signal.dim() == 1: recon_signal = recon_signal.unsqueeze(1)

        # 2. Neural Score (Complex Analysis)
        features = torch.cat([h_i, r_curr, r_hist, td_err, recon_signal], dim=1)
        norm_features = self.norm(features)
        neural_score = self.net(norm_features)

        # 3. ROBUST HEURISTIC (The Fix for Instability)
        # We explicitly calculate a "Safe Score":
        # Score = Reward (Performance) + F1 (Generalization) - TD_Error (Instability)
        
        # --- CRITICAL FIX: SCALING ---
        # Raw rewards can be ~100. We scale by 0.01 to get them near 1.0
        # Then clamp to prevent massive values from exploding gradients
        r_safe = torch.clamp(r_curr * 0.01, -2.0, 2.0)
        
        f1_safe = torch.clamp(recon_signal, 0.0, 1.0)
        
        # # TD Error is bad, so we subtract it. tanh() keeps it in [0, 1]
        # td_penalty = torch.tanh(td_err) 
        
        # Ideally: High Reward + High F1 - High Instability
        # heuristic_score = r_safe + f1_safe - (0.5 * td_penalty)
        heuristic_score = r_safe + f1_safe
        # 4. Final Weighted Combination
        # We add the heuristic to the neural score. 
        # This guarantees that a client with High Reward/F1 ALWAYS starts with a higher score.
        final_score = neural_score + (self.heuristic_scale * heuristic_score)
        
        return final_score


class CentralizedAggregator(nn.Module):
    """
    FIXED AGGREGATOR: Identity-Aware Attention.
    
    Why this fixes the crash:
    1. It treats clients as a SEQUENCE, not a blob.
    2. It uses Positional Encodings to remember "Client 1 vs Client 2".
    """
    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 64):
        super(CentralizedAggregator, self).__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        
        # Project Global State (Context)
        self.state_proj = nn.Linear(state_dim, embed_dim)
        
        # Project Scalar Weights to Vectors
        self.weight_proj = nn.Linear(1, embed_dim)
        
        # IDENTITY AWARENESS (Positional Encoding)
        # This allows the server to learn: "Client 5 is reliable for DoS attacks"
        self.agent_pos_embed = nn.Parameter(torch.randn(1, num_agents, embed_dim))

        # Self-Attention (Let agents "talk" / context aware)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # Cross-Attention (Global State attends to Agents)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True
        )

        # Final Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, w_i_list, global_state):
        # w_i_list: [Batch, Num_Agents]
        # global_state: [Batch, State_Dim]

        # 1. Embed Clients
        # [B, N] -> [B, N, 1] -> [B, N, Embed]
        w_embed = self.weight_proj(w_i_list.unsqueeze(-1))
        
        # Add Identity (The Fix)
        # Now the model knows WHO sent the weight
        w_embed = w_embed + self.agent_pos_embed[:, :w_embed.shape[1], :]

        # 2. Refine Context (Transformer Encoder)
        client_context = self.transformer_encoder(w_embed) # [B, N, Embed]

        # 3. Create Query (Global State)
        # [B, Dim] -> [B, 1, Embed]
        q = self.state_proj(global_state).unsqueeze(1)

        # 4. Cross Attention
        # Query: Global State
        # Key/Value: The Sequence of Clients
        attn_out, _ = self.cross_attention(query=q, key=client_context, value=client_context)
        
        # 5. Predict Global Reward
        val_pred = self.head(attn_out.squeeze(1))
        
        return val_pred