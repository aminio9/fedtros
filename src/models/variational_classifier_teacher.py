"""Private Variational Classifier Teacher (VCT) for FedTROS-PR.

The private teacher adopts a supervised Variational Information Bottleneck (VIB)
formulation (following Alemi et al. and recent supervised VIB classification research).
It encodes client-specific input features into a regularized latent distribution
q_phi(z|x) = N(mu, diag(sigma^2)), trained with a class-balanced predictive loss plus
a KL divergence penalty against a standard Normal prior N(0, I):

    L_T = L_CBCE^T(logits, y) + beta_T * D_KL[q_phi(z|x) || N(0, I)]

During teacher training, latent sampling is stochastic (sample=True).
During evaluation, knowledge distillation (KD), and feature alignment, teacher
supervision is completely deterministic (sample=False / distill_forward), using mu_T(x)
directly to eliminate stochastic noise from repeated teacher supervision.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.common import ResidualBlock, TabularTransformerEncoder
from src.utils.utils import LOGVAR_MAX, LOGVAR_MIN

logger = logging.getLogger("VCT")


def kl_standard_normal(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    reduce: str = "mean",
) -> torch.Tensor:
    """Compute KL divergence D_KL[N(mu, sigma^2) || N(0, I)].

    L_KL = 0.5 * sum_k (mu_k^2 + exp(logvar_k) - 1 - logvar_k)

    Args:
        mu: Mean tensor of shape [B, D].
        logvar: Log-variance tensor of shape [B, D].
        reduce: 'mean' (default, averages over batch), 'sum', or 'none'.

    Returns:
        Scalar or per-sample KL divergence tensor.
    """
    logvar_clamped = logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
    var = torch.exp(logvar_clamped)
    # Element-wise KL divergence
    kl_per_dim = 0.5 * (mu.pow(2) + var - 1.0 - logvar_clamped)
    kl_per_sample = kl_per_dim.sum(dim=1)  # [B]

    if reduce == "none":
        return kl_per_sample
    if reduce == "sum":
        return kl_per_sample.sum()
    return kl_per_sample.mean()


class VariationalClassifierTeacher(nn.Module):
    """Private Variational Classifier Teacher with a stochastic latent bottleneck."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        transformer_cfg: DictConfig | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.latent_dim = int(latent_dim)
        self.act = nn.LeakyReLU(0.2)

        self.transformer_enabled = bool(
            transformer_cfg is not None and getattr(transformer_cfg, "enabled", False)
        )
        if self.transformer_enabled:
            self.transformer = TabularTransformerEncoder(
                feature_dim=self.input_dim,
                d_model=int(transformer_cfg.d_model),
                nhead=int(transformer_cfg.nhead),
                num_layers=int(transformer_cfg.num_layers),
                dim_feedforward=int(transformer_cfg.dim_feedforward),
                dropout=float(transformer_cfg.dropout),
                pooling=str(transformer_cfg.pooling),
            )
            backbone_out = self.transformer.output_dim
            self.fc_in = None
            self.res1 = None
            self.res2 = None
        else:
            self.transformer = None
            hidden_list = [int(v) for v in hidden_dims] or [512, 256]
            first_dim = hidden_list[0]
            self.fc_in = nn.Linear(self.input_dim, first_dim)
            self.res1 = ResidualBlock(first_dim)
            self.res2 = ResidualBlock(first_dim)
            backbone_out = first_dim

        second_dim = (
            int(hidden_dims[1])
            if (hidden_dims is not None and len(hidden_dims) > 1)
            else 256
        )
        self.fc2 = nn.Linear(backbone_out, second_dim)
        self.norm2 = nn.LayerNorm(second_dim)
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()

        self.mu_head = nn.Linear(second_dim, self.latent_dim)
        self.logvar_head = nn.Linear(second_dim, self.latent_dim)

        self.classifier = nn.Linear(self.latent_dim, self.num_classes)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode inputs to backbone representation and latent Gaussian parameters.

        Args:
            x: Input feature batch [B, input_dim].

        Returns:
            (h, mu, logvar): Backbone representation h [B, second_dim],
                             mean mu [B, latent_dim],
                             log-variance logvar [B, latent_dim].
        """
        if self.transformer is not None:
            h_raw = self.transformer(x)
        else:
            h_raw = self.act(self.fc_in(x))
            h_raw = self.res1(h_raw)
            h_raw = self.res2(h_raw)

        h = self.dropout(self.act(self.norm2(self.fc2(h_raw))))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return h, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Stochastic reparameterization trick: z = mu + sigma * eps."""
        logvar_clamped = logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
        std = torch.exp(0.5 * logvar_clamped)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    def forward(
        self, x: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the variational classifier teacher.

        Args:
            x: Input feature tensor [B, input_dim].
            sample: If True, sample z stochastically. If False, use deterministic mean mu.

        Returns:
            (logits, mu, logvar, h)
        """
        h, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        logits = self.classifier(z)
        return logits, mu, logvar, h

    def distill_forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministic inference / distillation forward pass.

        Computes logits = W_T * mu_T + b_T without stochastic sampling noise.

        Args:
            x: Input feature tensor [B, input_dim].

        Returns:
            (logits, mu, h)
        """
        h, mu, _logvar = self.encode(x)
        logits = self.classifier(mu)
        return logits, mu, h

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
        """Compute standard normal KL divergence."""
        return kl_standard_normal(mu, logvar, reduce=reduce)
