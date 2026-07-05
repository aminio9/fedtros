"""Lightweight student models for dynamic-KD federated IDS methods."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Compact globally shared student for DKD-FedOS.

    The CVAE-DQN teacher remains local/personalized.  This student is the only
    model family transmitted to the server.  For closed-set experiments it acts
    exactly as the previous classifier-only student.  For open-set experiments it
    can additionally expose a small class-conditioned reconstruction head used
    by the Yang-style reconstruction/EVT pipeline.
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
        reconstruction_enabled: bool = False,
        decoder_hidden_dims: list[int] | tuple[int, ...] | None = None,
        decoder_dropout: float | None = None,
        decoder_class_condition: bool = True,
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
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.feature_dim = prev_dim

        self.reconstruction_enabled = bool(reconstruction_enabled)
        self.decoder_class_condition = bool(decoder_class_condition)
        self.decoder: nn.Module | None = None
        if self.reconstruction_enabled:
            self.decoder = self._build_decoder(
                activation=activation,
                dropout=float(decoder_dropout if decoder_dropout is not None else dropout_p),
                norm=norm_name,
                hidden_dims=decoder_hidden_dims,
            )

    def _build_decoder(
        self,
        *,
        activation: str,
        dropout: float,
        norm: str,
        hidden_dims: list[int] | tuple[int, ...] | None,
    ) -> nn.Sequential:
        decoder_hidden = [int(v) for v in (hidden_dims or [128, 256])]
        decoder_input_dim = int(self.feature_dim) + (int(self.num_classes) if self.decoder_class_condition else 0)
        layers: list[nn.Module] = []
        prev_dim = decoder_input_dim
        dropout_p = max(float(dropout), 0.0)
        for dim in decoder_hidden:
            layers.append(nn.Linear(prev_dim, dim))
            if norm in {"layernorm", "layer_norm", "ln"}:
                layers.append(nn.LayerNorm(dim))
            elif norm in {"batchnorm", "batch_norm", "bn"}:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(_activation(activation))
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, int(self.input_dim)))
        return nn.Sequential(*layers)

    def _condition_to_one_hot(
        self,
        class_condition: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if class_condition.ndim == 1 or (class_condition.ndim == 2 and class_condition.shape[-1] == 1):
            labels = class_condition.view(-1).long().clamp(0, self.num_classes - 1)
            return F.one_hot(labels, num_classes=self.num_classes).to(device=device, dtype=dtype)
        if class_condition.ndim == 2 and class_condition.shape[1] == self.num_classes:
            return class_condition.to(device=device, dtype=dtype)
        raise ValueError(
            "class_condition must be labels shaped [B] / [B,1] or one-hot shaped "
            f"[B,{self.num_classes}], got {tuple(class_condition.shape)}."
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.head(features)
        return features, logits

    def reconstruct(
        self,
        features: torch.Tensor,
        class_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct inputs from student features for open-set EVT training.

        The decoder is intentionally optional.  Closed-set experiments keep it
        disabled so the student parameterization and behavior match the previous
        v6 classifier-only implementation.
        """
        if not self.reconstruction_enabled or self.decoder is None:
            raise RuntimeError("Student reconstruction head is disabled for this run.")
        decoder_input = features
        if self.decoder_class_condition:
            if class_condition is None:
                raise ValueError("class_condition is required when decoder_class_condition=True.")
            one_hot = self._condition_to_one_hot(
                class_condition,
                batch_size=int(features.shape[0]),
                device=features.device,
                dtype=features.dtype,
            )
            if one_hot.shape[0] != features.shape[0]:
                raise ValueError(
                    "class_condition batch size must match features batch size: "
                    f"{one_hot.shape[0]} != {features.shape[0]}"
                )
            decoder_input = torch.cat([features, one_hot], dim=1)
        return self.decoder(decoder_input)
