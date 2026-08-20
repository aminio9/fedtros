"""Unified Prototype-Rank Rejection (FedTROS-PR) Open-Set Recognition."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.openset.prototype_bank import PrototypeBank, l2_normalize_np
from src.openset.rank_calibration import empirical_cdf_rank

logger = logging.getLogger("PrototypeRank")
UNKNOWN_LABEL_ID = -1
OPEN_SET_LABEL_ID = 99


def compute_prototype_rank_scores(
    features: np.ndarray,
    predicted_classes: np.ndarray,
    prototype_bank: PrototypeBank,
    reference_distances_by_class: dict[int, np.ndarray],
) -> np.ndarray:
    """Compute Prototype-Rank rejection suspiciousness scores in [0, 1].

    For each sample x with predicted class c:
      1. Compute distance d(x, P_c) to class prototypes.
      2. Compute empirical CDF rank R_c(d(x, P_c)) against calibrated knowns.

    Args:
        features: Feature matrix [N, D] (either normalized h_S or osr_mu).
        predicted_classes: Class predictions [N].
        prototype_bank: Fitted PrototypeBank.
        reference_distances_by_class: Sorted reference distance arrays for rank mapping.

    Returns:
        Array of rank scores in [0, 1]. Higher -> more suspicious / unknown.
    """
    n_samples = features.shape[0]
    ranks = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        c = int(predicted_classes[i])
        raw_score = prototype_bank.score(features[i : i + 1], c)[0]
        ref = reference_distances_by_class.get(c, np.array([]))
        if ref.size > 0:
            ranks[i] = empirical_cdf_rank(raw_score, ref)
        else:
            ranks[i] = raw_score

    return ranks


def evaluate_prototype_rank_rejection(
    y_true: np.ndarray,
    y_pred_closed: np.ndarray,
    unknown_scores: np.ndarray,
    threshold: float,
    *,
    class_names: dict[int, str] | list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate open-set classification and unknown detection metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_closed = np.asarray(y_pred_closed, dtype=np.int64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)

    is_unknown_true = (y_true < 0) | (y_true == OPEN_SET_LABEL_ID)
    is_known_true = ~is_unknown_true

    # Binary unknown detection metrics
    if np.any(is_unknown_true) and np.any(is_known_true):
        binary_labels = is_unknown_true.astype(np.int64)
        auroc = float(roc_auc_score(binary_labels, unknown_scores))
        auprc = float(average_precision_score(binary_labels, unknown_scores))
        fpr, tpr, thresholds_roc = roc_curve(binary_labels, unknown_scores)
        idx95 = int(np.argmin(np.abs(tpr - 0.95)))
        fpr95 = float(fpr[idx95])
    else:
        auroc = float("nan")
        auprc = float("nan")
        fpr95 = float("nan")

    # Open-set final prediction
    is_rejected = unknown_scores >= threshold
    y_pred_open = np.where(is_rejected, UNKNOWN_LABEL_ID, y_pred_closed)

    # Closed-set accuracy before and after rejection
    if np.any(is_known_true):
        acc_before = float(accuracy_score(y_true[is_known_true], y_pred_closed[is_known_true]))
        known_retained = is_known_true & ~is_rejected
        if np.any(known_retained):
            acc_after = float(accuracy_score(y_true[known_retained], y_pred_open[known_retained]))
        else:
            acc_after = 0.0
        known_false_unknown_rate = float(np.mean(is_rejected[is_known_true]))
    else:
        acc_before = float("nan")
        acc_after = float("nan")
        known_false_unknown_rate = float("nan")

    # Unknown recall
    if np.any(is_unknown_true):
        unknown_recall = float(np.mean(is_rejected[is_unknown_true]))
    else:
        unknown_recall = float("nan")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "fpr95": fpr95,
        "closed_set_acc_before": acc_before,
        "closed_set_acc_after": acc_after,
        "unknown_recall": unknown_recall,
        "known_false_unknown_rate": known_false_unknown_rate,
        "rejection_threshold": float(threshold),
        "y_pred_open": y_pred_open,
    }
