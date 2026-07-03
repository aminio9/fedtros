from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ModelOutput = dict[str, torch.Tensor | dict[str, Any] | None]


class CVAEQChainModelAdapter(nn.Module):
    """Read-only prediction adapter for the existing CVAE-DQN model stack."""

    model_name = "cvae_dqn_adapter"

    def __init__(
        self,
        *,
        prior_net: nn.Module,
        value_net_main: nn.Module,
        recognition_net: nn.Module | None = None,
        generation_net: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.prior_net = prior_net
        self.value_net_main = value_net_main
        self.recognition_net = recognition_net
        self.generation_net = generation_net

    def forward(self, x: torch.Tensor) -> ModelOutput:
        mu, logvar = self.prior_net(x)
        logits = self.value_net_main(mu, x)
        return {
            "logits": logits,
            "features": mu,
            "reconstruction": None,
            "mu": mu,
            "logvar": logvar,
            "q_values": logits,
            "aux": {},
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.prior_net(x)
        return mu

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["logits"]  # type: ignore[return-value]

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.predict_logits(x), dim=-1)

    def get_open_set_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def model_summary(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "parameters": int(self.parameter_count()),
            "training": bool(self.training),
        }


def validate_tabular_output(output: Mapping[str, Any], *, batch_size: int, num_classes: int) -> None:
    """Validate the shared model-output dictionary shape contract."""
    logits = output.get("logits")
    features = output.get("features")
    if not torch.is_tensor(logits):
        raise ValueError("Model output must contain tensor key 'logits'.")
    if not torch.is_tensor(features):
        raise ValueError("Model output must contain tensor key 'features'.")
    if tuple(logits.shape) != (int(batch_size), int(num_classes)):
        raise ValueError(
            f"logits shape must be {(batch_size, num_classes)}, got {tuple(logits.shape)}."
        )
    if features.ndim != 2 or features.shape[0] != int(batch_size):
        raise ValueError("features must have shape [batch, feature_dim].")
