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


def _cfg_value(cfg: DictConfig | None, key: str, default: object) -> object:
    if cfg is None:
        return default
    return getattr(cfg, key, default)


class FTFeatureTokenizer(nn.Module):
    """FT-Transformer-style numerical tokenizer for scalar tabular features."""

    def __init__(
        self,
        feature_dim: int,
        d_model: int,
        dropout: float,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.weight = nn.Parameter(torch.empty(self.feature_dim, d_model))
        self.bias = nn.Parameter(torch.empty(self.feature_dim, d_model))
        self.feature_gate = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError(f"Expected [batch, features], got {tuple(features.shape)}")
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Tokenizer feature_dim mismatch: expected {self.feature_dim}, got {features.shape[1]}"
            )
        tokens = features.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        gate = self.feature_gate(features).unsqueeze(-1)
        return self.dropout(tokens * gate)


class GatedTransformerBlock(nn.Module):
    """Pre-norm transformer block with data-dependent gates on residual updates."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ff_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.attn_norm(tokens)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        tokens = tokens + self.dropout(self.attn_gate(x) * attn_out)

        x = self.ff_norm(tokens)
        ff_out = self.ff(x)
        return tokens + self.dropout(self.ff_gate(x) * ff_out)


class AttentionPooling(nn.Module):
    """Learned attention pooling over feature tokens."""

    def __init__(self, d_model: int):
        super().__init__()
        self.scorer = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.scorer(tokens), dim=1)
        return (weights * tokens).sum(dim=1)


class GatedTabularTransformerEncoder(nn.Module):
    """Tokenizer -> gated transformer blocks -> attention pooling -> embedding h."""

    def __init__(
        self,
        feature_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.tokenizer = FTFeatureTokenizer(
            feature_dim=feature_dim,
            d_model=d_model,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                GatedTransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.pool = AttentionPooling(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.output_dim = d_model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(features)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(self.pool(tokens))


def _build_gated_tabular_encoder(
    feature_dim: int,
    transformer_cfg: DictConfig | None,
) -> GatedTabularTransformerEncoder:
    d_model = int(_cfg_value(transformer_cfg, "d_model", 128))
    nhead = int(_cfg_value(transformer_cfg, "nhead", 4))
    num_layers = int(_cfg_value(transformer_cfg, "num_layers", 4))
    dim_feedforward = int(_cfg_value(transformer_cfg, "dim_feedforward", 2 * d_model))
    dropout = float(_cfg_value(transformer_cfg, "dropout", 0.1))
    return GatedTabularTransformerEncoder(
        feature_dim=feature_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )


class PriorNetwork(nn.Module):
    """Encodes s -> (mu_p, log_var_p). GLOBAL component."""

    def __init__(self, s_dim: int, z_dim: int, transformer_cfg: DictConfig | None = None):
        super().__init__()
        self.encoder = _build_gated_tabular_encoder(s_dim, transformer_cfg)
        self.shared_embedding = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
        )

        self.mu_head = nn.Linear(256, z_dim)
        self.logvar_head = nn.Linear(256, z_dim)

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared_embedding(self.encoder(s))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(LOGVAR_MIN, LOGVAR_MAX)
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
        self.encoder = _build_gated_tabular_encoder(s_dim + num_actions, transformer_cfg)
        self.shared_embedding = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
        )

        self.mu_head = nn.Linear(256, z_dim)
        self.logvar_head = nn.Linear(256, z_dim)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if a.dim() == 1 or (a.dim() == 2 and a.shape[1] == 1):
            a_onehot = to_one_hot(a, self.num_actions)
        else:
            a_onehot = a
        features = torch.cat([s, a_onehot.to(dtype=s.dtype)], dim=-1)
        h = self.shared_embedding(self.encoder(features))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(LOGVAR_MIN, LOGVAR_MAX)
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
