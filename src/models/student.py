"""Compact federated student model for FedTROS-PR and architecture-matched baselines."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.utils import LOGVAR_MAX, LOGVAR_MIN, to_one_hot


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU()
    if name in {"leaky_relu", "lrelu"}:
        return nn.LeakyReLU(negative_slope=0.1)
    if name == "tanh":
        return nn.Tanh()
    return nn.ReLU()


def _mlp(
    input_dim: int,
    hidden_dims: list[int] | tuple[int, ...],
    *,
    activation: str = "relu",
    dropout: float = 0.0,
    norm: str = "none",
    output_dim: int | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = int(input_dim)
    norm_name = str(norm).lower()
    dropout_p = max(float(dropout), 0.0)
    for dim in [int(v) for v in hidden_dims]:
        layers.append(nn.Linear(prev_dim, dim))
        if norm_name in {"layernorm", "layer_norm", "ln"}:
            layers.append(nn.LayerNorm(dim))
        elif norm_name in {"batchnorm", "batch_norm", "bn"}:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(_activation(activation))
        if dropout_p > 0.0:
            layers.append(nn.Dropout(p=dropout_p))
        prev_dim = dim
    if output_dim is not None:
        layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


class StudentIDSModel(nn.Module):
    """Federated student classifier with an optional disentangled OSR generator.

    The closed-set path is the compact FedTROS-PR student classifier.  When
    ``osr_enabled`` is true, the same federated student also owns an open-set
    generative branch inspired by classification-reconstruction OSR.  The OSR
    branch reuses the teacher generator idea (recognition/generation with class
    condition), but it is an independent **student branch**, not the private RL
    teacher generator.

    The OSR branch reads classifier features with ``detach`` by default during
    training.  That is the important part: reconstruction loss does not drag the
    classifier representation around and wreck known-class accuracy, because
    apparently we would prefer not to repeat that disaster.
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
        osr_enabled: bool = False,
        osr_latent_dim: int = 8,
        osr_hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        osr_decoder_hidden_dims: list[int] | tuple[int, ...] = (64, 128),
        osr_dropout: float = 0.05,
        osr_norm: str = "layernorm",
        osr_activation: str = "gelu",
        osr_detach_features: bool = True,
        open_set_enabled: bool = False,
    ):
        super().__init__()
        hidden = [int(v) for v in hidden_dims]
        if not hidden:
            hidden = [64, 32, 16]
        self.backbone = _mlp(
            int(input_dim),
            hidden,
            activation=activation,
            dropout=dropout,
            norm=norm,
            output_dim=None,
        )
        self.head = nn.Linear(hidden[-1], int(num_classes))
        # PROSER/FedPD++-style placeholder detector.  This auxiliary head has
        # K+1 outputs: K known classes plus one learned open-set placeholder.
        # It is federated as part of the global student and is trained only from
        # known samples plus known-derived feature/input mixup placeholders.
        self.open_set_enabled = bool(open_set_enabled)
        self.open_set_head = nn.Linear(hidden[-1], int(num_classes) + 1) if self.open_set_enabled else None
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.feature_dim = int(hidden[-1])

        self.osr_enabled = bool(osr_enabled)
        self.osr_latent_dim = int(osr_latent_dim)
        self.osr_detach_features = bool(osr_detach_features)
        if self.osr_enabled:
            enc_in = self.feature_dim + self.num_classes
            enc_hidden = [int(v) for v in osr_hidden_dims] or [128, 64]
            dec_hidden = [int(v) for v in osr_decoder_hidden_dims] or [64, 128]
            self.osr_encoder = _mlp(
                enc_in,
                enc_hidden,
                activation=osr_activation,
                dropout=osr_dropout,
                norm=osr_norm,
                output_dim=None,
            )
            enc_out = enc_hidden[-1]
            self.osr_mu_head = nn.Linear(enc_out, self.osr_latent_dim)
            self.osr_logvar_head = nn.Linear(enc_out, self.osr_latent_dim)
            self.osr_decoder = _mlp(
                self.osr_latent_dim + self.num_classes,
                dec_hidden,
                activation=osr_activation,
                dropout=osr_dropout,
                norm=osr_norm,
                output_dim=self.input_dim,
            )
        else:
            self.osr_encoder = None
            self.osr_mu_head = None
            self.osr_logvar_head = None
            self.osr_decoder = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.head(features)
        return features, logits

    def osr_parameters(self):
        if not self.osr_enabled:
            return []
        params = []
        for module in (self.osr_encoder, self.osr_mu_head, self.osr_logvar_head, self.osr_decoder):
            if module is not None:
                params.extend(list(module.parameters()))
        return params

    def classifier_parameters(self):
        return list(self.backbone.parameters()) + list(self.head.parameters())

    def open_set_parameters(self):
        if not self.open_set_enabled or self.open_set_head is None:
            return []
        return list(self.open_set_head.parameters())

    def open_set_forward(
        self,
        x: torch.Tensor,
        *,
        detach_features: bool = True,
    ) -> torch.Tensor:
        if not self.open_set_enabled or self.open_set_head is None:
            raise RuntimeError("Student open-set placeholder head is disabled.")
        features = self.backbone(x)
        if detach_features:
            features = features.detach()
        return self.open_set_head(features)

    def open_set_logits_from_features(
        self,
        features: torch.Tensor,
        *,
        detach_features: bool = True,
    ) -> torch.Tensor:
        if not self.open_set_enabled or self.open_set_head is None:
            raise RuntimeError("Student open-set placeholder head is disabled.")
        if detach_features:
            features = features.detach()
        return self.open_set_head(features)

    def open_set_score(
        self,
        x: torch.Tensor,
        *,
        detach_features: bool = True,
        score_type: str = "margin",
    ) -> dict[str, torch.Tensor]:
        logits = self.open_set_forward(x, detach_features=detach_features)
        known_logits = logits[:, : self.num_classes]
        unknown_logit = logits[:, self.num_classes]
        probs = F.softmax(logits, dim=1)
        unknown_prob = probs[:, self.num_classes]
        margin = unknown_logit - known_logits.max(dim=1).values
        score = unknown_prob if str(score_type).lower() in {"prob", "probability", "softmax"} else margin
        return {
            "logits": logits,
            "unknown_prob": unknown_prob,
            "unknown_margin": margin,
            "score": score,
        }

    def _onehot(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long().view(-1).clamp(0, self.num_classes - 1)
        return to_one_hot(labels, self.num_classes).to(next(self.parameters()).device)

    def osr_forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        *,
        detach_features: bool | None = None,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        if not self.osr_enabled:
            raise RuntimeError("Student OSR branch is disabled. Set training.student_osr_enabled=true.")
        features, logits = self.forward(x)
        should_detach = self.osr_detach_features if detach_features is None else bool(detach_features)
        osr_features = features.detach() if should_detach else features
        y_onehot = self._onehot(labels)
        enc = torch.cat([osr_features, y_onehot], dim=1)
        hidden = self.osr_encoder(enc)
        mu = self.osr_mu_head(hidden)
        logvar = self.osr_logvar_head(hidden).clamp(LOGVAR_MIN, LOGVAR_MAX)
        if sample and self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + (torch.randn_like(std) * std)
        else:
            z = mu
        recon = self.osr_decoder(torch.cat([z, y_onehot], dim=1))
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z, "features": features, "logits": logits}

    def osr_score(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        *,
        nll_weight: float = 0.10,
        detach_features: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.osr_forward(x, labels, detach_features=detach_features, sample=False)
        recon_error = F.mse_loss(out["recon"], x, reduction="none").mean(dim=1)
        latent_nll = 0.5 * (out["mu"].pow(2) + torch.exp(out["logvar"]) - out["logvar"] - 1.0).sum(dim=1)
        score = recon_error + (float(nll_weight) * latent_nll)
        out.update({"recon_error": recon_error, "latent_nll": latent_nll, "score": score})
        return out

    @staticmethod
    def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        t = max(float(temperature), 1.0e-6)
        # Raw energy.  Some tabular IDS splits show unknowns on the low-energy
        # side, so Prototype-Rank evaluation can calibrate energy direction instead of
        # assuming that only high energy means unknown.
        return -t * torch.logsumexp(logits / t, dim=1)
