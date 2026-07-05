"""Extreme Value Theory utilities for reconstruction-error open-set detection.

This module implements the Yang-2025 style EVT calibration used by the
student-decoder open-set pipeline: fit a Generalized Pareto Distribution (GPD)
on exceedances over a high reconstruction-error threshold.  The threshold can
be selected by a Mean Excess Function (MEF) stability heuristic, with a robust
quantile fallback for small validation sets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import genpareto

logger = logging.getLogger("EVT")


class EVTModel:
    """Class-wise GPD tail model for reconstruction-error rejection.

    Parameters
    ----------
    tail_size_percent:
        Fraction of the largest known reconstruction errors used as the tail
        when the quantile fallback is selected.
    threshold_method:
        ``"mef"`` selects the threshold using a Mean Excess Function linearity
        heuristic. ``"quantile"`` uses the fixed upper-tail fraction. ``"fixed"``
        expects ``fixed_threshold`` in :meth:`fit`.
    target_fpr:
        Desired known false-positive rate used to derive the final class-wise
        rejection threshold from the fitted GPD.
    """

    def __init__(
        self,
        tail_size_percent: float = 0.10,
        *,
        threshold_method: str = "mef",
        target_fpr: float = 0.05,
    ):
        self.tail_size_percent = min(max(float(tail_size_percent), 1e-6), 1.0)
        self.threshold_method = str(threshold_method or "mef").lower()
        self.target_fpr = min(max(float(target_fpr), 1e-9), 1.0)

        self.threshold_u: float | None = None
        self.decision_threshold: float | None = None
        self.gpd_params: tuple[float, float, float] | None = None
        self.tail_fraction: float = 0.0
        self.tail_size: int = 0
        self.num_errors: int = 0
        self.empirical_quantile_threshold: float | None = None
        self.threshold_selection: dict[str, Any] = {}

    @staticmethod
    def _clean_errors(reconstruction_errors: np.ndarray) -> np.ndarray:
        errors = np.asarray(reconstruction_errors, dtype=np.float64).reshape(-1)
        errors = errors[np.isfinite(errors)]
        if errors.size == 0:
            raise ValueError("reconstruction_errors must contain at least one finite value")
        return errors

    def _quantile_threshold(self, errors: np.ndarray) -> float:
        q = max(0.0, min(1.0, 1.0 - self.tail_size_percent))
        threshold = float(np.quantile(errors, q))
        if q == 0.0:
            threshold -= 1e-12
        return threshold

    @staticmethod
    def _mean_excess(errors: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs: list[float] = []
        ys: list[float] = []
        for u in thresholds:
            tail = errors[errors > u] - u
            if tail.size > 0:
                xs.append(float(u))
                ys.append(float(np.mean(tail)))
        return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)

    def _mef_threshold(
        self,
        errors: np.ndarray,
        *,
        min_tail_size: int,
        mef_min_quantile: float,
        mef_max_quantile: float,
        mef_num_candidates: int,
    ) -> float:
        """Select a high threshold by MEF linearity with robust fallbacks.

        In the Pickands-Balkema-de Haan/GPD setting, the mean excess plot tends
        to be approximately linear above a suitable high threshold.  We choose
        the lowest candidate threshold whose subsequent MEF segment is both
        sufficiently populated and highly linear.  This is intentionally
        conservative: lower suitable thresholds leave enough tail samples for
        stable MLE on network datasets.
        """
        n = int(errors.size)
        min_tail_size = max(3, min(int(min_tail_size), max(n - 1, 1)))
        q_low = max(0.0, min(float(mef_min_quantile), 0.98))
        q_high = max(q_low + 1e-6, min(float(mef_max_quantile), 0.995))
        max_q_for_tail = max(0.0, 1.0 - (min_tail_size / max(n, 1)))
        q_high = min(q_high, max_q_for_tail)
        if q_high <= q_low or n < min_tail_size + 2:
            self.threshold_selection = {
                "method": "quantile_fallback",
                "reason": "too_few_samples_for_mef",
            }
            return self._quantile_threshold(errors)

        quantiles = np.linspace(q_low, q_high, max(5, int(mef_num_candidates)))
        candidates = np.unique(np.quantile(errors, quantiles))
        xs, ys = self._mean_excess(errors, candidates)
        if xs.size < 5:
            self.threshold_selection = {
                "method": "quantile_fallback",
                "reason": "insufficient_mean_excess_points",
            }
            return self._quantile_threshold(errors)

        best: tuple[float, float, int, int] | None = None  # r2, threshold, tail_count, start_idx
        # Require at least four MEF points in the linear segment.
        for start_idx in range(0, max(1, xs.size - 3)):
            u = float(xs[start_idx])
            tail_count = int(np.sum(errors > u))
            if tail_count < min_tail_size:
                continue
            x_seg = xs[start_idx:]
            y_seg = ys[start_idx:]
            if x_seg.size < 4 or float(np.var(x_seg)) <= 0.0:
                continue
            try:
                slope, intercept = np.polyfit(x_seg, y_seg, deg=1)
                pred = slope * x_seg + intercept
                ss_res = float(np.sum((y_seg - pred) ** 2))
                ss_tot = float(np.sum((y_seg - np.mean(y_seg)) ** 2))
                r2 = 1.0 if ss_tot <= 1e-12 else 1.0 - (ss_res / ss_tot)
            except Exception:
                continue
            # Prefer the lowest threshold once the line is good enough; otherwise
            # take the best R2 with enough tail samples.
            if r2 >= 0.90:
                self.threshold_selection = {
                    "method": "mef",
                    "r2": float(r2),
                    "start_index": int(start_idx),
                    "tail_count": int(tail_count),
                    "preferred_lowest_good_segment": True,
                }
                return u
            if best is None or r2 > best[0]:
                best = (float(r2), u, tail_count, start_idx)

        if best is not None:
            self.threshold_selection = {
                "method": "mef_best_available",
                "r2": float(best[0]),
                "start_index": int(best[3]),
                "tail_count": int(best[2]),
            }
            return float(best[1])

        self.threshold_selection = {
            "method": "quantile_fallback",
            "reason": "no_valid_mef_segment",
        }
        return self._quantile_threshold(errors)

    def fit(
        self,
        reconstruction_errors: np.ndarray,
        fixed_threshold: float | None = None,
        *,
        target_fpr: float | None = None,
        min_tail_size: int = 20,
        threshold_method: str | None = None,
        mef_min_quantile: float = 0.60,
        mef_max_quantile: float = 0.95,
        mef_num_candidates: int = 40,
        logger: logging.Logger | None = None,
    ) -> None:
        """Fit the GPD tail model by MLE.

        The final ``decision_threshold`` is class-wise.  It is derived from the
        fitted conditional GPD while respecting the desired overall known-FPR.
        """
        active_logger = logger or logging.getLogger("EVT")
        errors = self._clean_errors(reconstruction_errors)
        self.num_errors = int(errors.size)
        if target_fpr is not None:
            self.target_fpr = min(max(float(target_fpr), 1e-9), 1.0)
        if threshold_method is not None:
            self.threshold_method = str(threshold_method).lower()

        self.empirical_quantile_threshold = float(np.quantile(errors, 1.0 - self.target_fpr))

        if fixed_threshold is not None or self.threshold_method == "fixed":
            if fixed_threshold is None:
                raise ValueError("fixed_threshold must be provided when threshold_method='fixed'.")
            self.threshold_u = float(fixed_threshold)
            self.threshold_selection = {"method": "fixed"}
        elif self.threshold_method == "quantile":
            self.threshold_u = self._quantile_threshold(errors)
            self.threshold_selection = {"method": "quantile", "tail_size_percent": self.tail_size_percent}
        else:
            self.threshold_u = self._mef_threshold(
                errors,
                min_tail_size=min_tail_size,
                mef_min_quantile=mef_min_quantile,
                mef_max_quantile=mef_max_quantile,
                mef_num_candidates=mef_num_candidates,
            )

        # If the selected threshold leaves too few exceedances, relax to a
        # quantile that guarantees enough samples for MLE where possible.
        tail = errors[errors > self.threshold_u] - self.threshold_u
        if tail.size < min_tail_size and errors.size > min_tail_size:
            q = max(0.0, 1.0 - (min_tail_size / float(errors.size)))
            relaxed_u = float(np.quantile(errors, q))
            if q == 0.0:
                relaxed_u -= 1e-12
            active_logger.warning(
                "EVT threshold relaxed for tail size | old_u=%.6g | new_u=%.6g | old_tail=%d | min_tail=%d",
                float(self.threshold_u),
                relaxed_u,
                int(tail.size),
                int(min_tail_size),
            )
            self.threshold_u = relaxed_u
            self.threshold_selection = {
                **dict(self.threshold_selection),
                "relaxed_for_min_tail": True,
                "relaxed_quantile": float(q),
            }
            tail = errors[errors > self.threshold_u] - self.threshold_u

        self.tail_size = int(tail.size)
        self.tail_fraction = float(self.tail_size / max(self.num_errors, 1))
        if self.tail_size <= 0:
            active_logger.error("Insufficient tail data. Using empirical threshold only.")
            self.gpd_params = (0.0, 0.0, 1.0)
            self.decision_threshold = float(self.empirical_quantile_threshold)
            return

        try:
            shape, loc, scale = genpareto.fit(tail, floc=0.0)
            scale = max(float(scale), 1e-12)
            self.gpd_params = (float(shape), float(loc), scale)
        except Exception as exc:
            active_logger.error("GPD MLE fitting failed: %s. Using empirical threshold only.", exc)
            self.gpd_params = (0.0, 0.0, 1.0)
            self.decision_threshold = float(self.empirical_quantile_threshold)
            return

        # Convert desired overall known-FPR to a conditional tail quantile.
        if self.target_fpr >= self.tail_fraction or self.tail_fraction <= 0.0:
            decision = float(self.empirical_quantile_threshold)
            decision_source = "empirical_quantile"
        else:
            conditional_cdf = 1.0 - (self.target_fpr / max(self.tail_fraction, 1e-12))
            conditional_cdf = min(max(conditional_cdf, 0.0), 1.0 - 1e-12)
            shape, loc, scale = self.gpd_params
            q_excess = float(genpareto.ppf(conditional_cdf, shape, loc=loc, scale=scale))
            if not np.isfinite(q_excess) or q_excess < 0.0:
                decision = float(self.empirical_quantile_threshold)
                decision_source = "empirical_quantile_fallback"
            else:
                decision = float(self.threshold_u + q_excess)
                decision_source = "gpd_quantile"

        # Never make the final rejection threshold lower than u.
        self.decision_threshold = max(float(decision), float(self.threshold_u))
        self.threshold_selection["decision_source"] = decision_source

    def predict_probability_unknown(self, reconstruction_error: float) -> float:
        """Return a monotonic tail score in [0, 1]."""
        if self.gpd_params is None or self.threshold_u is None:
            return 0.0
        reconstruction_error = float(reconstruction_error)
        if reconstruction_error <= self.threshold_u:
            return 0.0
        shape, loc, scale = self.gpd_params
        try:
            prob = genpareto.cdf(reconstruction_error - self.threshold_u, shape, loc=loc, scale=scale)
        except Exception:
            prob = 0.0
        return float(np.clip(prob, 0.0, 1.0))

    def is_unknown(self, reconstruction_error: float) -> bool:
        threshold = self.decision_threshold
        if threshold is None:
            threshold = self.threshold_u
        if threshold is None:
            return False
        return float(reconstruction_error) > float(threshold)

    def to_payload(self) -> dict[str, Any]:
        return {
            "threshold_u": self.threshold_u,
            "decision_threshold": self.decision_threshold,
            "gpd_params": self.gpd_params,
            "tail_size_percent": self.tail_size_percent,
            "threshold_method": self.threshold_method,
            "target_fpr": self.target_fpr,
            "tail_fraction": self.tail_fraction,
            "tail_size": self.tail_size,
            "num_errors": self.num_errors,
            "empirical_quantile_threshold": self.empirical_quantile_threshold,
            "threshold_selection": self.threshold_selection,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EVTModel":
        model = cls(
            payload.get("tail_size_percent", 0.10),
            threshold_method=payload.get("threshold_method", "mef"),
            target_fpr=payload.get("target_fpr", 0.05),
        )
        model.threshold_u = None if payload.get("threshold_u") is None else float(payload["threshold_u"])
        model.decision_threshold = (
            None if payload.get("decision_threshold") is None else float(payload["decision_threshold"])
        )
        params = payload.get("gpd_params")
        model.gpd_params = None if params is None else tuple(float(v) for v in params)
        model.tail_fraction = float(payload.get("tail_fraction", 0.0))
        model.tail_size = int(payload.get("tail_size", 0))
        model.num_errors = int(payload.get("num_errors", 0))
        empirical = payload.get("empirical_quantile_threshold")
        model.empirical_quantile_threshold = None if empirical is None else float(empirical)
        model.threshold_selection = dict(payload.get("threshold_selection", {}))
        return model


def save_evt_collection(
    evt_map: dict[int, EVTModel], filepath: Path | str, logger: logging.Logger | None = None
) -> None:
    active_logger = logger or logging.getLogger("EVT")
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {int(label): model.to_payload() for label, model in evt_map.items()}
    joblib.dump(payload, path)
    active_logger.info("Saved EVT collection with %s classes to %s", len(payload), path)


def load_evt_collection(filepath: Path | str) -> dict[int, EVTModel]:
    data = joblib.load(filepath)
    if not isinstance(data, dict):
        raise ValueError("EVT collection file is corrupted or not a dict.")
    return {int(k): EVTModel.from_payload(v) for k, v in data.items()}


def save_evt_meta(meta: dict, filepath: Path | str, logger: logging.Logger | None = None) -> None:
    active_logger = logger or logging.getLogger("EVT")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    active_logger.info("Saved EVT meta to %s", filepath)


def load_evt_meta(filepath: Path | str) -> dict:
    with open(filepath, encoding="utf-8") as fh:
        return json.load(fh)
