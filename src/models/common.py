"""Common neural network building blocks for tabular models."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Wider fully-connected residual block."""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2),
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
