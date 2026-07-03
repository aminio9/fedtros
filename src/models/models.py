import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.utils.utils import LOGVAR_MAX, LOGVAR_MIN, to_one_hot

logger = logging.getLogger("Models")


def _cfg_value(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default) if cfg is not None else default


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(int(dim)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.scale


class SwiGLU(nn.Module):
    """SwiGLU feed-forward projection that preserves the input dimension."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(int(dim), int(hidden_dim) * 2)
        self.out = nn.Linear(int(hidden_dim), int(dim))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.fc(x).chunk(2, dim=-1)
        return self.out(self.dropout(F.silu(gate) * value))


class GatedResidualBlock(nn.Module):
    """Compact gated residual block used by the generator."""

    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.05):
        super().__init__()
        hidden_dim = int(dim) * int(expansion)
        self.norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim, dropout)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.ffn(self.norm(x)))


class BatchEnsembleLinear(nn.Module):
    """
    Parameter-efficient ensemble linear layer.

    Input shape: [batch, ensemble, in_dim]
    Output shape: [batch, ensemble, out_dim]
    """

    def __init__(self, in_dim: int, out_dim: int, ensemble_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(int(out_dim), int(in_dim)))
        self.bias = nn.Parameter(torch.zeros(int(ensemble_size), int(out_dim)))
        self.r = nn.Parameter(torch.empty(int(ensemble_size), int(in_dim)))
        self.s = nn.Parameter(torch.empty(int(ensemble_size), int(out_dim)))

        nn.init.kaiming_uniform_(self.weight, a=0.2)
        nn.init.normal_(self.r, mean=1.0, std=0.02)
        nn.init.normal_(self.s, mean=1.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.r.unsqueeze(0)
        y = torch.einsum("bei,oi->beo", x, self.weight)
        return y * self.s.unsqueeze(0) + self.bias.unsqueeze(0)


class EnsembleGatedBlock(nn.Module):
    """BatchEnsemble gated residual block for tabular hidden states."""

    def __init__(
        self,
        dim: int,
        ensemble_size: int,
        expansion: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        hidden_dim = int(dim) * int(expansion)
        self.norm = RMSNorm(dim)
        self.fc1 = BatchEnsembleLinear(dim, hidden_dim * 2, ensemble_size)
        self.fc2 = BatchEnsembleLinear(hidden_dim, dim, ensemble_size)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        gate, value = self.fc1(x).chunk(2, dim=-1)
        x = self.dropout(F.silu(gate) * value)
        x = self.fc2(x)
        return residual + self.dropout(x)


class FastTabMBackbone(nn.Module):
    """Fast MLP-style tabular backbone with a cheap implicit ensemble."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        ensemble_size: int = 4,
        dropout: float = 0.05,
        expansion: int = 2,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.ensemble_size = int(ensemble_size)
        self.output_dim = self.hidden_dim

        self.in_norm = RMSNorm(self.input_dim)
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                EnsembleGatedBlock(
                    dim=self.hidden_dim,
                    ensemble_size=self.ensemble_size,
                    expansion=int(expansion),
                    dropout=float(dropout),
                )
                for _ in range(int(depth))
            ]
        )
        self.out_norm = RMSNorm(self.hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError(f"Expected [batch, features], got {tuple(features.shape)}")
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"FastTabM feature_dim mismatch: expected {self.input_dim}, got {features.shape[1]}"
            )
        x = self.input_proj(self.in_norm(features))
        x = x.unsqueeze(1).expand(-1, self.ensemble_size, -1).contiguous()
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x.mean(dim=1))


def _backbone_kwargs(
    cfg: DictConfig | None,
    *,
    q_backbone: bool = False,
) -> dict[str, int | float]:
    hidden_key = "q_hidden_dim" if q_backbone else "hidden_dim"
    depth_key = "q_depth" if q_backbone else "depth"
    ensemble_key = "q_ensemble_size" if q_backbone else "ensemble_size"
    return {
        "hidden_dim": int(_cfg_value(cfg, hidden_key, _cfg_value(cfg, "hidden_dim", 256))),
        "depth": int(_cfg_value(cfg, depth_key, _cfg_value(cfg, "depth", 3))),
        "ensemble_size": int(
            _cfg_value(cfg, ensemble_key, _cfg_value(cfg, "ensemble_size", 4))
        ),
        "dropout": float(_cfg_value(cfg, "dropout", 0.05)),
        "expansion": int(_cfg_value(cfg, "expansion", 2)),
    }


class PriorNetwork(nn.Module):
    """Encodes s -> (mu_p, log_var_p)."""

    def __init__(self, s_dim: int, z_dim: int, backbone_cfg: DictConfig | None = None):
        super().__init__()
        kwargs = _backbone_kwargs(backbone_cfg)
        self.backbone = FastTabMBackbone(input_dim=s_dim, **kwargs)
        hidden_dim = int(kwargs["hidden_dim"])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            RMSNorm(hidden_dim),
        )
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.head(self.backbone(s))
        mu = self.mu_head(x)
        logvar = self.logvar_head(x).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar


class RecognitionNetwork(nn.Module):
    """Encodes (s, a) -> (mu_q, log_var_q)."""

    def __init__(
        self,
        s_dim: int,
        num_actions: int,
        z_dim: int,
        backbone_cfg: DictConfig | None = None,
    ):
        super().__init__()
        self.num_actions = int(num_actions)
        kwargs = _backbone_kwargs(backbone_cfg)
        self.backbone = FastTabMBackbone(input_dim=s_dim + self.num_actions, **kwargs)
        hidden_dim = int(kwargs["hidden_dim"])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            RMSNorm(hidden_dim),
        )
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if a.dim() == 1 or (a.dim() == 2 and a.shape[1] == 1):
            a_onehot = to_one_hot(a, self.num_actions)
        else:
            a_onehot = a.float()
        x = torch.cat([s, a_onehot], dim=-1)
        x = self.head(self.backbone(x))
        mu = self.mu_head(x)
        logvar = self.logvar_head(x).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar


class Encoder(nn.Module):
    def __init__(
        self,
        s_dim: int,
        num_actions: int,
        z_dim: int,
        backbone_cfg: DictConfig | None = None,
    ):
        super().__init__()
        self.prior = PriorNetwork(s_dim, z_dim, backbone_cfg)
        self.recognition = RecognitionNetwork(s_dim, num_actions, z_dim, backbone_cfg)

    def prior_forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.prior(s)

    def recognition_forward(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.recognition(s, a)


class MainQNetwork(nn.Module):
    """Decodes (z, s) -> Q-values with a dueling head."""

    def __init__(
        self,
        s_dim: int,
        z_dim: int,
        num_actions: int,
        backbone_cfg: DictConfig | None = None,
    ):
        super().__init__()
        self.num_actions = int(num_actions)
        kwargs = _backbone_kwargs(backbone_cfg, q_backbone=True)
        self.backbone = FastTabMBackbone(input_dim=s_dim + z_dim, **kwargs)
        hidden_dim = int(kwargs["hidden_dim"])
        head_dim = max(hidden_dim // 2, 1)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.SiLU(),
            RMSNorm(head_dim),
            nn.Linear(head_dim, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.SiLU(),
            RMSNorm(head_dim),
            nn.Linear(head_dim, self.num_actions),
        )

    def forward(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = self.backbone(torch.cat([z, s], dim=-1))
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class TargetQNetwork(MainQNetwork):
    pass


class Decoder(nn.Module):
    def __init__(
        self,
        s_dim: int,
        z_dim: int,
        num_actions: int,
        backbone_cfg: DictConfig | None = None,
    ):
        super().__init__()
        self.main_q = MainQNetwork(s_dim, z_dim, num_actions, backbone_cfg)
        self.target_q = TargetQNetwork(s_dim, z_dim, num_actions, backbone_cfg)

    def forward(self, z: torch.Tensor, s: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        net = self.target_q if use_target else self.main_q
        return net(z, s)


class ValueNetwork(nn.Module):
    def __init__(
        self,
        s_dim: int,
        z_dim: int,
        num_actions: int,
        backbone_cfg: DictConfig | None = None,
    ):
        super().__init__()
        self.encoder = Encoder(s_dim, num_actions, z_dim, backbone_cfg)
        self.decoder = Decoder(s_dim, z_dim, num_actions, backbone_cfg)

    def forward(self, z: torch.Tensor, s: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        return self.decoder(z, s, use_target=use_target)


class GenerationNetwork(nn.Module):
    """Class-conditioned feature generator p(s|z,a)."""

    def __init__(
        self,
        z_dim: int,
        num_actions: int,
        s_dim: int,
        generator_cfg: DictConfig | None = None,
    ):
        super().__init__()
        self.num_actions = int(num_actions)
        hidden_dim = int(_cfg_value(generator_cfg, "hidden_dim", 128))
        depth = int(_cfg_value(generator_cfg, "depth", 2))
        dropout = float(_cfg_value(generator_cfg, "dropout", 0.05))
        expansion = int(_cfg_value(generator_cfg, "expansion", 2))

        self.input = nn.Linear(z_dim + self.num_actions, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                GatedResidualBlock(hidden_dim, expansion=expansion, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.out_norm = RMSNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, s_dim)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if a.dim() == 1 or (a.dim() == 2 and a.shape[1] == 1):
            a_onehot = to_one_hot(a, self.num_actions)
        else:
            a_onehot = a.float()

        x = self.input(torch.cat([z, a_onehot], dim=-1))
        for block in self.blocks:
            x = block(x)
        return self.out(self.out_norm(x))


class OpenSetQChainModelFactory:
    """Factory to build the FastTabM CVAE-DQN value and generation networks."""

    def __init__(self, model_cfg: DictConfig):
        self.state_dim = int(model_cfg.state_dim)
        self.latent_dim = int(model_cfg.latent_dim)
        self.num_actions = int(model_cfg.num_actions)
        self.backbone_cfg = getattr(model_cfg, "backbone", None)
        self.generator_cfg = getattr(model_cfg, "generator", None)

    def create_value_network(self) -> ValueNetwork:
        return ValueNetwork(
            s_dim=self.state_dim,
            z_dim=self.latent_dim,
            num_actions=self.num_actions,
            backbone_cfg=self.backbone_cfg,
        )

    def create_generation_network(self) -> GenerationNetwork:
        return GenerationNetwork(
            z_dim=self.latent_dim,
            num_actions=self.num_actions,
            s_dim=self.state_dim,
            generator_cfg=self.generator_cfg,
        )
