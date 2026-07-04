"""Lightweight student models for dynamic-KD federated IDS methods."""

from __future__ import annotations

import torch
import torch.nn as nn


class StudentIDSModel(nn.Module):
    """Compact MLP classifier shared by DKD-FedOS clients.

    The CVAE-DQN teacher stays local/personalized.  This student is the only
    model transmitted to the server, following Sentinel-style dual-model FL.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list[int] | tuple[int, ...]):
        super().__init__()
        hidden = [int(v) for v in hidden_dims]
        if not hidden:
            hidden = [64, 32, 16]
        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        for dim in hidden:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, int(num_classes))
        self.feature_dim = prev_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.head(features)
        return features, logits
