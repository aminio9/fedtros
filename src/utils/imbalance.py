from __future__ import annotations

from typing import Any

import torch


def _cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    return getattr(cfg, key, default) if cfg is not None else default


def compute_class_weights(
    labels: torch.Tensor,
    *,
    num_classes: int,
    mode: str = "inverse_frequency",
    beta: float = 0.999,
    min_weight: float = 0.2,
    max_weight: float = 5.0,
    normalize: str = "mean",
) -> torch.Tensor:
    """Compute finite class weights from integer labels."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    labels = labels.detach().cpu().long()
    valid = (labels >= 0) & (labels < int(num_classes))
    counts = torch.bincount(labels[valid], minlength=int(num_classes)).float()
    if counts.sum() <= 0:
        return torch.ones(int(num_classes), dtype=torch.float32)

    mode = str(mode).lower()
    if mode in {"none", "uniform"}:
        weights = torch.ones_like(counts)
    elif mode in {"inverse", "inverse_frequency"}:
        weights = counts.sum() / counts.clamp_min(1.0)
    elif mode in {"effective", "effective_number"}:
        beta = min(max(float(beta), 0.0), 0.999999)
        weights = (1.0 - beta) / (1.0 - torch.pow(torch.full_like(counts, beta), counts))
        weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
    else:
        raise ValueError(f"Unknown class-weight mode: {mode!r}")

    if str(normalize).lower() == "mean":
        positive = weights[weights > 0]
        if positive.numel() > 0:
            weights = weights / positive.mean().clamp_min(1e-8)
    weights = torch.clamp(weights, min=float(min_weight), max=float(max_weight))
    weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
    return weights.float()


def class_weights_from_config(
    labels: torch.Tensor,
    *,
    num_classes: int,
    cfg: Any,
) -> torch.Tensor:
    if cfg is None or not bool(_cfg_value(cfg, "enabled", False)):
        return torch.ones(int(num_classes), dtype=torch.float32)

    manual = _cfg_value(cfg, "manual_class_weights", None)
    if manual is not None:
        weights = torch.tensor([float(value) for value in manual], dtype=torch.float32)
        if weights.numel() != int(num_classes):
            raise ValueError("manual_class_weights length must match num_classes.")
        if not torch.isfinite(weights).all():
            raise ValueError("manual_class_weights must be finite.")
        return weights

    return compute_class_weights(
        labels,
        num_classes=int(num_classes),
        mode=str(_cfg_value(cfg, "weight_mode", "inverse_frequency")),
        beta=float(_cfg_value(cfg, "effective_number_beta", 0.999)),
        min_weight=float(_cfg_value(cfg, "min_weight", 0.2)),
        max_weight=float(_cfg_value(cfg, "max_weight", 5.0)),
        normalize=str(_cfg_value(cfg, "normalize", "mean")),
    )


def sample_weights_from_class_weights(
    labels: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    labels = labels.detach().cpu().long()
    sample_weights = torch.ones(labels.numel(), dtype=torch.float32)
    valid = (labels >= 0) & (labels < class_weights.numel())
    sample_weights[valid] = class_weights[labels[valid]].float()
    sample_weights[~torch.isfinite(sample_weights)] = 1.0
    return sample_weights.clamp_min(0.0)
