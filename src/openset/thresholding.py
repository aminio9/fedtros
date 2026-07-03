from __future__ import annotations

import numpy as np
import torch


def _to_numpy_1d(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(values):
        return values.detach().float().cpu().numpy().reshape(-1)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _labels_to_numpy(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(values):
        return values.detach().long().cpu().numpy().reshape(-1)
    return np.asarray(values, dtype=np.int64).reshape(-1)


def known_validation_mask(
    labels: np.ndarray | torch.Tensor | None = None,
    *,
    known_labels: list[int] | tuple[int, ...] | np.ndarray | None = None,
    unknown_label_id: int = -1,
    known_mask: np.ndarray | torch.Tensor | None = None,
    size: int | None = None,
) -> np.ndarray:
    """Return a boolean mask for samples eligible for validation thresholding."""
    if known_mask is not None:
        mask = (
            known_mask.detach().cpu().numpy().astype(bool).reshape(-1)
            if torch.is_tensor(known_mask)
            else np.asarray(known_mask, dtype=bool).reshape(-1)
        )
    elif labels is not None:
        label_array = _labels_to_numpy(labels)
        if known_labels is not None:
            mask = np.isin(label_array, np.asarray(known_labels, dtype=np.int64))
        else:
            mask = label_array != int(unknown_label_id)
    elif size is not None:
        mask = np.ones(int(size), dtype=bool)
    else:
        raise ValueError("labels, known_mask, or size is required.")

    if size is not None and mask.size != int(size):
        raise ValueError("validation mask length must match the score array length.")
    return mask


def select_validation_threshold(
    scores: np.ndarray | torch.Tensor,
    validation_labels: np.ndarray | torch.Tensor | None = None,
    *,
    known_labels: list[int] | tuple[int, ...] | np.ndarray | None = None,
    unknown_label_id: int = -1,
    known_mask: np.ndarray | torch.Tensor | None = None,
    target_known_fpr: float = 0.05,
    mode: str = "validation_known_fpr",
    fixed_threshold: float | None = None,
    score_direction: str = "higher_unknown",
) -> float:
    """Select an unknown threshold from validation data only.

    ``score_direction`` controls how predictions are made:
    ``higher_unknown`` rejects scores above the threshold; ``lower_unknown``
    rejects scores below the threshold.
    """
    score_array = _to_numpy_1d(scores)
    if score_array.size == 0:
        raise ValueError("Cannot select threshold from empty scores.")
    if not np.isfinite(score_array).all():
        raise ValueError("Threshold scores contain NaN or Inf.")

    mode_normalized = str(mode).lower()
    if mode_normalized == "fixed":
        if fixed_threshold is None:
            raise ValueError("fixed_threshold must be provided when mode='fixed'.")
        return float(fixed_threshold)
    if mode_normalized not in {"validation_known_fpr", "known_fpr", "validation"}:
        raise ValueError("threshold mode must be validation_known_fpr or fixed.")

    fpr = float(target_known_fpr)
    if not 0.0 <= fpr < 1.0:
        raise ValueError("target_known_fpr must be in [0, 1).")

    known = known_validation_mask(
        validation_labels,
        known_labels=known_labels,
        unknown_label_id=unknown_label_id,
        known_mask=known_mask,
        size=score_array.size,
    )
    known_scores = score_array[known]
    if known_scores.size == 0:
        raise ValueError("Validation threshold selection needs known validation samples.")

    direction = str(score_direction).lower()
    if direction == "higher_unknown":
        quantile = 1.0 - fpr
    elif direction == "lower_unknown":
        quantile = fpr
    else:
        raise ValueError("score_direction must be higher_unknown or lower_unknown.")
    return float(np.quantile(known_scores, quantile))


def predict_known_unknown(
    scores: np.ndarray | torch.Tensor,
    threshold: float,
    *,
    score_direction: str = "higher_unknown",
) -> np.ndarray:
    """Return 1 for unknown, 0 for known under the configured score direction."""
    score_array = _to_numpy_1d(scores)
    direction = str(score_direction).lower()
    if direction == "higher_unknown":
        return (score_array > float(threshold)).astype(np.int64)
    if direction == "lower_unknown":
        return (score_array < float(threshold)).astype(np.int64)
    raise ValueError("score_direction must be higher_unknown or lower_unknown.")
