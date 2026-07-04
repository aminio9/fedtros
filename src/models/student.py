"""Lightweight student models for dynamic-KD federated IDS methods."""

from __future__ import annotations

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU()
    if name in {"leaky_relu", "lrelu"}:
        return nn.LeakyReLU(negative_slope=0.1)
    if name == "tanh":
        return nn.Tanh()
    return nn.ReLU()


class StudentIDSModel(nn.Module):
    """Compact classifier shared by DKD-FedOS clients.

    The CVAE-DQN teacher stays local/personalized.  This student is the only
    model transmitted to the server, following Sentinel-style dual-model FL.
    v5 makes the student configurable enough for harsh non-IID traffic while
    still keeping it much smaller than the teacher/generator stack.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | tuple[int, ...],
        *,
        activation: str = "relu",
        dropout: float = 0.0,
        norm: str = "none",
    ):
        super().__init__()
        hidden = [int(v) for v in hidden_dims]
        if not hidden:
            hidden = [64, 32, 16]
        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        norm_name = str(norm).lower()
        dropout_p = max(float(dropout), 0.0)
        for dim in hidden:
            layers.append(nn.Linear(prev_dim, dim))
            if norm_name in {"layernorm", "layer_norm", "ln"}:
                layers.append(nn.LayerNorm(dim))
            elif norm_name in {"batchnorm", "batch_norm", "bn"}:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(_activation(activation))
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            prev_dim = dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, int(num_classes))
        self.feature_dim = prev_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.head(features)
        return features, logits
