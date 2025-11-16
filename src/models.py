import torch
import torch.nn as nn
from typing import Tuple
import logging
from omegaconf import DictConfig

try:
    from .utils import to_one_hot, LOGVAR_MIN, LOGVAR_MAX
except ImportError:
    from utils import to_one_hot, LOGVAR_MIN, LOGVAR_MAX

logger = logging.getLogger("Models")


class ResidualBlock(nn.Module):
    """Simple fully-connected residual block with LayerNorm."""
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SelfAttentionBlock(nn.Module):
    """Project features into tokens and mix them via multi-head self-attention."""

    def __init__(self, input_dim: int, num_tokens: int = 4, token_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.proj = nn.Linear(input_dim, num_tokens * token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(num_tokens * token_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        tokens = self.proj(x).view(batch_size, self.num_tokens, self.token_dim)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = attn_out.reshape(batch_size, -1)
        attn_out = self.out(attn_out)
        return x + attn_out


class PriorNetwork(nn.Module):
    """Encodes s -> (μ_p, log_var_p) with residual connections. GLOBAL component."""
    def __init__(self, s_dim: int, z_dim: int):
        super().__init__()
        self.act = nn.GELU()
        self.fc_in = nn.Linear(s_dim, 256)
        self.res1 = ResidualBlock(256)
        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.res2 = ResidualBlock(128)
        self.mu_head = nn.Linear(128, z_dim)
        self.logvar_head = nn.Linear(128, z_dim)
        logger.debug("PriorNetwork initialized (Residual)")

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.act(self.fc_in(s))
        x = self.res1(x)
        x = self.act(self.norm2(self.fc2(x)))
        x = self.res2(x)
        mu = self.mu_head(x)
        logvar = self.logvar_head(x).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar


class RecognitionNetwork(nn.Module):
    """Encodes (s, a) -> (μ_q, log_var_q) with residual connections. LOCAL component."""
    def __init__(self, s_dim: int, num_actions: int, z_dim: int):
        super().__init__()
        self.num_actions = num_actions
        in_dim = s_dim + num_actions
        self.act = nn.GELU()
        self.fc_in = nn.Linear(in_dim, 256)
        self.res1 = ResidualBlock(256)
        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.res2 = ResidualBlock(128)
        self.mu_head = nn.Linear(128, z_dim)
        self.logvar_head = nn.Linear(128, z_dim)
        logger.debug("RecognitionNetwork initialized (Residual)")
        
    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if a.dim() == 1 or (a.dim() == 2 and a.shape[1] == 1):
            a_onehot = to_one_hot(a, self.num_actions)
        else:
            a_onehot = a
        x = torch.cat([s, a_onehot], dim=-1)
        x = self.act(self.fc_in(x))
        x = self.res1(x)
        x = self.act(self.norm2(self.fc2(x)))
        x = self.res2(x)
        mu = self.mu_head(x)
        logvar = self.logvar_head(x).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar


class Encoder(nn.Module):
    """Container for the two encoder networks."""
    def __init__(self, s_dim: int, num_actions: int, z_dim: int):
        super().__init__()
        self.prior = PriorNetwork(s_dim, z_dim)
        self.recognition = RecognitionNetwork(s_dim, num_actions, z_dim)

    def prior_forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.prior(s)

    def recognition_forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.recognition(s, a)


class MainQNetwork(nn.Module):
    """Decodes (z, s) -> Q-values with a dueling value/advantage head."""
    def __init__(self, s_dim: int, z_dim: int, num_actions: int):
        super().__init__()
        in_dim = s_dim + z_dim
        self.num_actions = num_actions
        self.act = nn.GELU()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.norm = nn.LayerNorm(128)
        self.attn_block = SelfAttentionBlock(128, num_tokens=4, token_dim=32, num_heads=4)

        self.value_fc1 = nn.Linear(128, 64)
        self.value_fc2 = nn.Linear(64, 1)

        self.advantage_fc1 = nn.Linear(128, 64)
        self.advantage_fc2 = nn.Linear(64, num_actions)
        logger.debug("MainQNetwork initialized (Dueling)")

    def forward(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, s], dim=-1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.norm(x)
        x = self.attn_block(x)

        value = self.act(self.value_fc1(x))
        value = self.value_fc2(value)

        advantage = self.act(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class TargetQNetwork(MainQNetwork):
    """A copy of MainQNetwork for stable TD targets. This is the LOCAL component."""
    pass


class Decoder(nn.Module):
    """Container for the main and target Q-networks."""
    def __init__(self, s_dim: int, z_dim: int, num_actions: int):
        super().__init__()
        self.main_q = MainQNetwork(s_dim, z_dim, num_actions)
        self.target_q = TargetQNetwork(s_dim, z_dim, num_actions)

    def forward(self, z: torch.Tensor, s: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        net = self.target_q if use_target else self.main_q
        return net(z, s)


class ValueNetwork(nn.Module):
    """The complete CVAE-DQN value network (Encoder + Decoder)."""
    def __init__(self, s_dim: int, z_dim: int, num_actions: int):
        super().__init__()
        self.encoder = Encoder(s_dim, num_actions, z_dim)
        self.decoder = Decoder(s_dim, z_dim, num_actions)

    def forward(self, z: torch.Tensor, s: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        return self.decoder(z, s, use_target=use_target)


class GenerationNetwork(nn.Module):
    """The decoder for the Unknown Attack Recognition module (Part 2 of paper)."""
    def __init__(self, z_dim: int, num_actions: int, s_dim: int):
        super().__init__()
        self.num_actions = num_actions
        in_dim = z_dim + num_actions
        # Symmetrical (reversed) Balanced Architecture: 32 -> 64 -> 128 -> 256
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.out = nn.Linear(256, s_dim)
        self.act = nn.ReLU()
        self.final_act = nn.Sigmoid() # Assumes data is scaled 0-1
        logger.debug("GenerationNetwork initialized (Balanced)")

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if a.dim() == 1 or (a.dim() == 2 and a.shape[1] == 1):
            a_onehot = to_one_hot(a, self.num_actions)
        else:
            a_onehot = a
        x = torch.cat([z, a_onehot], dim=-1)
        x = self.act(self.fc1(x))   
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        s_recon = self.final_act(self.out(x))
        return s_recon


class OpenSetQChainModelFactory:
    """Factory to create model components based on config."""
    def __init__(self, cfg: DictConfig):
        self.s_dim = cfg.state_dim
        self.z_dim = cfg.latent_dim
        self.num_actions = cfg.num_actions
        logger.info(
            "Initializing models with s_dim=%s, z_dim=%s, num_actions=%s",
            self.s_dim,
            self.z_dim,
            self.num_actions,
        )

    def create_value_network(self) -> ValueNetwork:
        return ValueNetwork(self.s_dim, self.z_dim, self.num_actions)

    def create_generation_network(self) -> GenerationNetwork:
        return GenerationNetwork(self.z_dim, self.num_actions, self.s_dim)
