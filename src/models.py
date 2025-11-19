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
    """Wider fully-connected residual block."""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2),  # Changed to LeakyReLU for better gradient flow
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PriorNetwork(nn.Module):
    """Encodes s -> (μ_p, log_var_p). GLOBAL component."""

    def __init__(self, s_dim: int, z_dim: int):
        super().__init__()
        # INCREASED SIZE: 256 -> 512 to capture more patterns
        self.fc_in = nn.Linear(s_dim, 512)
        self.res1 = ResidualBlock(512)
        self.res2 = ResidualBlock(512)  # Added extra depth

        self.fc2 = nn.Linear(512, 256)
        self.norm2 = nn.LayerNorm(256)

        self.mu_head = nn.Linear(256, z_dim)
        self.logvar_head = nn.Linear(256, z_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.act(self.fc_in(s))
        x = self.res1(x)
        x = self.res2(x)
        x = self.act(self.norm2(self.fc2(x)))
        mu = self.mu_head(x)
        logvar = self.logvar_head(x).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar


class RecognitionNetwork(nn.Module):
    """Encodes (s, a) -> (μ_q, log_var_q). LOCAL component."""

    def __init__(self, s_dim: int, num_actions: int, z_dim: int):
        super().__init__()
        self.num_actions = num_actions
        in_dim = s_dim + num_actions

        # INCREASED SIZE: 256 -> 512
        self.fc_in = nn.Linear(in_dim, 512)
        self.res1 = ResidualBlock(512)
        self.res2 = ResidualBlock(512)

        self.fc2 = nn.Linear(512, 256)
        self.norm2 = nn.LayerNorm(256)

        self.mu_head = nn.Linear(256, z_dim)
        self.logvar_head = nn.Linear(256, z_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if a.dim() == 1 or (a.dim() == 2 and a.shape[1] == 1):
            a_onehot = to_one_hot(a, self.num_actions)
        else:
            a_onehot = a
        x = torch.cat([s, a_onehot], dim=-1)
        x = self.act(self.fc_in(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.act(self.norm2(self.fc2(x)))
        mu = self.mu_head(x)
        logvar = self.logvar_head(x).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar


class Encoder(nn.Module):
    def __init__(self, s_dim: int, num_actions: int, z_dim: int):
        super().__init__()
        self.prior = PriorNetwork(s_dim, z_dim)
        self.recognition = RecognitionNetwork(s_dim, num_actions, z_dim)

    def prior_forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.prior(s)

    def recognition_forward(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.recognition(s, a)


class MainQNetwork(nn.Module):
    """Decodes (z, s) -> Q-values with a wider dueling head."""

    def __init__(self, s_dim: int, z_dim: int, num_actions: int):
        super().__init__()
        in_dim = s_dim + z_dim
        self.num_actions = num_actions
        self.act = nn.LeakyReLU(0.2)

        # INCREASED SIZE: 512 neurons for better decision boundaries
        self.fc1 = nn.Linear(in_dim, 512)
        self.res1 = ResidualBlock(512)

        self.fc2 = nn.Linear(512, 256)
        self.norm = nn.LayerNorm(256)

        # Value Head
        self.value_fc1 = nn.Linear(256, 128)
        self.value_fc2 = nn.Linear(128, 1)

        # Advantage Head
        self.advantage_fc1 = nn.Linear(256, 128)
        self.advantage_fc2 = nn.Linear(128, num_actions)

    def forward(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, s], dim=-1)
        x = self.act(self.fc1(x))
        x = self.res1(x)
        x = self.norm(self.act(self.fc2(x)))

        value = self.act(self.value_fc1(x))
        value = self.value_fc2(value)

        advantage = self.act(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)

        # Dueling DQN Logic
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class TargetQNetwork(MainQNetwork):
    pass


class Decoder(nn.Module):
    def __init__(self, s_dim: int, z_dim: int, num_actions: int):
        super().__init__()
        self.main_q = MainQNetwork(s_dim, z_dim, num_actions)
        self.target_q = TargetQNetwork(s_dim, z_dim, num_actions)

    def forward(
        self, z: torch.Tensor, s: torch.Tensor, use_target: bool = False
    ) -> torch.Tensor:
        net = self.target_q if use_target else self.main_q
        return net(z, s)


class ValueNetwork(nn.Module):
    def __init__(self, s_dim: int, z_dim: int, num_actions: int):
        super().__init__()
        self.encoder = Encoder(s_dim, num_actions, z_dim)
        self.decoder = Decoder(s_dim, z_dim, num_actions)

    def forward(
        self, z: torch.Tensor, s: torch.Tensor, use_target: bool = False
    ) -> torch.Tensor:
        return self.decoder(z, s, use_target=use_target)


# Kept GenerationNetwork simpler as it is for part 2 (Intrusion Recog)
class GenerationNetwork(nn.Module):
    def __init__(self, z_dim: int, num_actions: int, s_dim: int):
        super().__init__()
        self.num_actions = num_actions
        in_dim = z_dim + num_actions
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.out = nn.Linear(256, s_dim)
        self.act = nn.ReLU()
        self.final_act = nn.Sigmoid()

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if a.dim() == 1 or (a.dim() == 2 and a.shape[1] == 1):
            a_onehot = to_one_hot(a, self.num_actions)
        else:
            a_onehot = a
        x = torch.cat([z, a_onehot], dim=-1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        s_recon = self.final_act(self.out(x))
        return s_recon


class OpenSetQChainModelFactory:
    """Simple factory to build value/generation networks using model config."""

    def __init__(self, model_cfg: DictConfig):
        self.state_dim = int(model_cfg.state_dim)
        self.latent_dim = int(model_cfg.latent_dim)
        self.num_actions = int(model_cfg.num_actions)

    def create_value_network(self) -> ValueNetwork:
        return ValueNetwork(
            s_dim=self.state_dim,
            z_dim=self.latent_dim,
            num_actions=self.num_actions,
        )

    def create_generation_network(self) -> GenerationNetwork:
        return GenerationNetwork(
            z_dim=self.latent_dim,
            num_actions=self.num_actions,
            s_dim=self.state_dim,
        )
