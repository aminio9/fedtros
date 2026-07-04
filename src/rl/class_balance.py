"""Class-balanced losses for imbalanced/non-IID traffic clients."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def effective_number_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    *,
    beta: float = 0.999,
    min_weight: float = 0.2,
    max_weight: float = 5.0,
    normalize: bool = True,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return effective-number class weights from local labels.

    Uses w_c = (1 - beta) / (1 - beta ** n_c), clamps extreme values, and
    mean-normalizes over present classes to preserve CE scale.
    """
    target_device = torch.device(device) if device is not None else labels.device
    labels_cpu = labels.detach().cpu().long().view(-1)
    num_classes = max(int(num_classes), 1)
    counts = torch.bincount(labels_cpu.clamp_min(0), minlength=num_classes).float()[:num_classes]
    weights = torch.ones(num_classes, dtype=torch.float32)
    present = counts > 0
    if not bool(present.any().item()):
        return weights.to(target_device)

    beta = float(beta)
    if beta <= 0.0 or beta >= 1.0:
        beta = 0.999
    effective_num = 1.0 - torch.pow(torch.full_like(counts, beta), counts.clamp_min(1.0))
    raw = (1.0 - beta) / effective_num.clamp_min(1e-12)
    raw[~present] = 1.0
    raw = raw.clamp(min=float(min_weight), max=float(max_weight))
    if normalize:
        mean_present = raw[present].mean().clamp_min(1e-8)
        raw = raw / mean_present
        raw = raw.clamp(min=float(min_weight), max=float(max_weight))
    return raw.to(target_device)


def class_balanced_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None,
    *,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    labels = labels.view(-1).long().to(logits.device).clamp(0, logits.shape[1] - 1)
    ce_weights = None
    if weights is not None and weights.numel() >= logits.shape[1]:
        ce_weights = weights.to(logits.device, dtype=logits.dtype)[: logits.shape[1]]
        ce_weights = ce_weights / ce_weights.mean().clamp_min(1e-8)
    return F.cross_entropy(
        logits,
        labels,
        weight=ce_weights,
        label_smoothing=max(float(label_smoothing), 0.0),
    )
