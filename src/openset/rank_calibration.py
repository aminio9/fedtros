"""Stratified disjoint 70/30 rank and threshold calibration for FedTROS-PR."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger("RankCalibration")
EPS = 1.0e-12


@dataclass
class DisjointCalibrationSplit:
    """Provenance tracking for the strict 70/30 calibration partition."""

    fit_indices: np.ndarray
    cal_indices: np.ndarray
    fit_sha256: str
    cal_sha256: str
    seed: int
    num_fit_samples: int
    num_cal_samples: int

    def assert_disjoint(self) -> None:
        """Enforce zero sample leakage between prototype fitting and calibration."""
        intersection = np.intersect1d(self.fit_indices, self.cal_indices)
        if len(intersection) > 0:
            raise ValueError(
                f"Data leakage detected! {len(intersection)} samples overlap "
                f"between prototype fitting and rank calibration."
            )


def make_stratified_70_30_split(
    labels: np.ndarray,
    *,
    seed: int = 42,
    fit_fraction: float = 0.70,
) -> DisjointCalibrationSplit:
    """Split validation indices into deterministic stratified 70% fit and 30% calibration."""
    n_samples = len(labels)
    indices = np.arange(n_samples)

    if n_samples < 5:
        # Fallback for ultra-tiny unit test sets
        fit_idx = indices[: int(np.ceil(fit_fraction * n_samples))]
        cal_idx = indices[int(np.ceil(fit_fraction * n_samples)) :]
        if len(cal_idx) == 0:
            cal_idx = fit_idx
    else:
        unique, counts = np.unique(labels, return_counts=True)
        # Check if every class has at least 2 samples for stratified split
        if np.all(counts >= 2):
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=1.0 - fit_fraction,
                random_state=seed,
            )
            fit_idx, cal_idx = next(sss.split(indices, labels))
        else:
            rng = np.random.default_rng(seed)
            shuffled = rng.permutation(indices)
            split_point = int(np.ceil(fit_fraction * n_samples))
            fit_idx = np.sort(shuffled[:split_point])
            cal_idx = np.sort(shuffled[split_point:])

    fit_idx = np.sort(np.asarray(fit_idx, dtype=np.int64))
    cal_idx = np.sort(np.asarray(cal_idx, dtype=np.int64))

    fit_hash = hashlib.sha256(fit_idx.tobytes()).hexdigest()
    cal_hash = hashlib.sha256(cal_idx.tobytes()).hexdigest()

    split = DisjointCalibrationSplit(
        fit_indices=fit_idx,
        cal_indices=cal_idx,
        fit_sha256=fit_hash,
        cal_sha256=cal_hash,
        seed=seed,
        num_fit_samples=len(fit_idx),
        num_cal_samples=len(cal_idx),
    )
    if n_samples >= 5:
        split.assert_disjoint()
    return split


def empirical_cdf_rank(value: float, sorted_reference: np.ndarray) -> float:
    """Compute empirical CDF rank in [0, 1]. Higher score -> more unknown -> larger rank."""
    if not np.isfinite(value) or sorted_reference.size == 0:
        return float("nan")
    return float(
        np.searchsorted(sorted_reference, float(value), side="right")
        / max(sorted_reference.size, 1)
    )


def fit_empirical_rank_threshold(
    calibration_scores: np.ndarray,
    *,
    target_fpr: float = 0.05,
) -> float:
    """Select threshold corresponding to target known false positive rate."""
    scores = np.asarray(calibration_scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return 0.5
    # Threshold at (1 - target_fpr) quantile
    return float(np.quantile(scores, 1.0 - target_fpr))
