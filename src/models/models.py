import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.utils.utils import LOGVAR_MAX, LOGVAR_MIN, to_one_hot

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


class TabularTransformerEncoder(nn.Module):
    """Transformer encoder over scalar tabular feature tokens."""

    def __init__(
        self,
        feature_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        pooling: str,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.pooling = str(pooling)
        self.token_projection = nn.Linear(1, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, feature_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = d_model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError(f"Expected [batch, features], got {tuple(features.shape)}")
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Transformer feature_dim mismatch: expected {self.feature_dim}, got {features.shape[1]}"
            )
        x = self.token_projection(features.unsqueeze(-1)) + self.position_embedding
        encoded = self.encoder(x)
        if self.pooling == "cls":
            return encoded[:, 0, :]
        if self.pooling == "max":
            return encoded.max(dim=1).values
        return encoded.mean(dim=1)


class PriorNetwork(nn.Module):
    """Encodes s -> (mu_p, log_var_p). GLOBAL component."""

    def __init__(self, s_dim: int, z_dim: int, transformer_cfg: DictConfig | None = None):
        super().__init__()
        self.transformer_enabled = bool(
            transformer_cfg is not None and getattr(transformer_cfg, "enabled", False)
        )
        if self.transformer_enabled:
            self.transformer = TabularTransformerEncoder(
                feature_dim=s_dim,
                d_model=int(transformer_cfg.d_model),
                nhead=int(transformer_cfg.nhead),
                num_layers=int(transformer_cfg.num_layers),
                dim_feedforward=int(transformer_cfg.dim_feedforward),
                dropout=float(transformer_cfg.dropout),
                pooling=str(transformer_cfg.pooling),
            )
            encoder_dim = self.transformer.output_dim
        else:
            self.transformer = None
            encoder_dim = 512
            self.fc_in = nn.Linear(s_dim, encoder_dim)
            self.res1 = ResidualBlock(encoder_dim)
            self.res2 = ResidualBlock(encoder_dim)

        self.fc2 = nn.Linear(encoder_dim, 256)
        self.norm2 = nn.LayerNorm(256)

        self.mu_head = nn.Linear(256, z_dim)
        self.logvar_head = nn.Linear(256, z_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transformer is not None:
            x = self.transformer(s)
        else:
            x = self.act(self.fc_in(s))
            x = self.res1(x)
            x = self.res2(x)
        x = self.act(self.norm2(self.fc2(x)))
        mu = self.mu_head(x)
        logvar = self.logvar_head(x).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar


class RecognitionNetwork(nn.Module):
    """Encodes (s, a) -> (mu_q, log_var_q). LOCAL component."""

    def __init__(
        self,
        s_dim: int,
        num_actions: int,
        z_dim: int,
        transformer_cfg: DictConfig | None = None,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.transformer_enabled = bool(
            transformer_cfg is not None and getattr(transformer_cfg, "enabled", False)
        )
        if self.transformer_enabled:
            self.transformer = TabularTransformerEncoder(
                feature_dim=s_dim,
                d_model=int(transformer_cfg.d_model),
                nhead=int(transformer_cfg.nhead),
                num_layers=int(transformer_cfg.num_layers),
                dim_feedforward=int(transformer_cfg.dim_feedforward),
                dropout=float(transformer_cfg.dropout),
                pooling=str(transformer_cfg.pooling),
            )
            self.action_projection = nn.Linear(num_actions, self.transformer.output_dim)
            encoder_dim = self.transformer.output_dim
        else:
            self.transformer = None
            self.action_projection = None
            in_dim = s_dim + num_actions
            encoder_dim = 512
            self.fc_in = nn.Linear(in_dim, encoder_dim)
            self.res1 = ResidualBlock(encoder_dim)
            self.res2 = ResidualBlock(encoder_dim)

        self.fc2 = nn.Linear(encoder_dim, 256)
        self.norm2 = nn.LayerNorm(256)

        self.mu_head = nn.Linear(256, z_dim)
        self.logvar_head = nn.Linear(256, z_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if a.dim() == 1 or (a.dim() == 2 and a.shape[1] == 1):
            a_onehot = to_one_hot(a, self.num_actions)
        else:
            a_onehot = a
        if self.transformer is not None and self.action_projection is not None:
            x = self.transformer(s) + self.action_projection(a_onehot)
        else:
            x = torch.cat([s, a_onehot], dim=-1)
            x = self.act(self.fc_in(x))
            x = self.res1(x)
            x = self.res2(x)
        x = self.act(self.norm2(self.fc2(x)))
        mu = self.mu_head(x)
        logvar = self.logvar_head(x).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar


class Encoder(nn.Module):
    def __init__(
        self,
        s_dim: int,
        num_actions: int,
        z_dim: int,
        transformer_cfg: DictConfig | None = None,
    ):
        super().__init__()
        self.prior = PriorNetwork(s_dim, z_dim, transformer_cfg)
        self.recognition = RecognitionNetwork(s_dim, num_actions, z_dim, transformer_cfg)

    def prior_forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.prior(s)

    def recognition_forward(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


class TargetQNetwork(MainQNetwork):
    pass


class Decoder(nn.Module):
    def __init__(self, s_dim: int, z_dim: int, num_actions: int):
        super().__init__()
        self.main_q = MainQNetwork(s_dim, z_dim, num_actions)
        self.target_q = TargetQNetwork(s_dim, z_dim, num_actions)

    def forward(self, z: torch.Tensor, s: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        net = self.target_q if use_target else self.main_q
        return net(z, s)


class ValueNetwork(nn.Module):
    def __init__(
        self,
        s_dim: int,
        z_dim: int,
        num_actions: int,
        transformer_cfg: DictConfig | None = None,
    ):
        super().__init__()
        self.encoder = Encoder(s_dim, num_actions, z_dim, transformer_cfg)
        self.decoder = Decoder(s_dim, z_dim, num_actions)

    def forward(self, z: torch.Tensor, s: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        return self.decoder(z, s, use_target=use_target)


class GenerationNetwork(nn.Module):
    def __init__(self, z_dim: int, num_actions: int, s_dim: int):
        super().__init__()
        self.num_actions = num_actions

        # 1. Combine z and action
        in_dim = z_dim + num_actions

        # 2. REDUCED CAPACITY ARCHITECTURE
        # Instead of expanding (64->128->256), we keep it small.
        # This acts as an information bottleneck, forcing the model
        # to rely on the class structure (action) to reconstruct valid features.
        self.fc1 = nn.Linear(in_dim, 64)

        # We jump straight to the output or use one small hidden layer.
        # Removing layers prevents "memorization" of unknown patterns.
        self.out = nn.Linear(64, s_dim)

        # LeakyReLU is often better for anomaly detection gradients than ReLU
        self.act = nn.LeakyReLU(0.2)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # Handle One-Hot encoding
        if a.dim() == 1 or (a.dim() == 2 and a.shape[1] == 1):
            a_onehot = to_one_hot(a, self.num_actions)
        else:
            a_onehot = a

        x = torch.cat([z, a_onehot], dim=-1)

        x = self.act(self.fc1(x))

        # 3. NO SIGMOID ACTIVATION
        # We remove self.final_act = nn.Sigmoid()
        # Why? If an unknown sample has values like -2.5 or +5.0 (Standard Scaled),
        # Sigmoid cannot reproduce them, or it clips the error.
        # A linear output allows the model to try (and fail) to reconstruct,
        # often resulting in larger, more detectable MSE gradients.
        return self.out(x)


class OpenSetQChainModelFactory:
    """Simple factory to build value/generation networks using model config."""

    def __init__(self, model_cfg: DictConfig):
        self.state_dim = int(model_cfg.state_dim)
        self.latent_dim = int(model_cfg.latent_dim)
        self.num_actions = int(model_cfg.num_actions)
        self.transformer_cfg = getattr(model_cfg, "transformer", None)

    def create_value_network(self) -> ValueNetwork:
        return ValueNetwork(
            s_dim=self.state_dim,
            z_dim=self.latent_dim,
            num_actions=self.num_actions,
            transformer_cfg=getattr(self, "transformer_cfg", None),
        )

    def create_generation_network(self) -> GenerationNetwork:
        return GenerationNetwork(
            z_dim=self.latent_dim,
            num_actions=self.num_actions,
            s_dim=self.state_dim,
        )
