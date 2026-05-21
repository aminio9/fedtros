import json
import logging
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import genpareto

logger = logging.getLogger("EVT")


class EVTModel:
    """EVT wrapper that supports fixed thresholds for robust tail fitting."""

    def __init__(self, tail_size_percent: float):
        self.tail_size_percent = min(max(float(tail_size_percent), 1e-6), 1.0)
        self.threshold_u: float | None = None
        self.gpd_params: tuple[float, float, float] | None = None

    def fit(
        self,
        reconstruction_errors: np.ndarray,
        fixed_threshold: float | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Fits the GPD to the tail.
        If fixed_threshold is provided, it is used directly.
        """
        active_logger = logger or logging.getLogger("EVT")
        if reconstruction_errors.ndim != 1 or reconstruction_errors.size == 0:
            raise ValueError("reconstruction_errors must be a non-empty 1D array")

        if fixed_threshold is not None:
            self.threshold_u = float(fixed_threshold)
        else:
            sorted_errors = np.sort(reconstruction_errors)
            if sorted_errors.size == 1:
                self.threshold_u = float(sorted_errors[0]) - 1e-9
                idx = 0
            else:
                idx = int(sorted_errors.size * (1.0 - self.tail_size_percent))
                idx = min(max(idx, 0), sorted_errors.size - 2)
                self.threshold_u = float(sorted_errors[idx])

        # Extract tail based on the established threshold
        tail = reconstruction_errors[reconstruction_errors > self.threshold_u] - self.threshold_u

        # Handle edge cases (empty tail)
        if tail.size == 0:
            if fixed_threshold is None:
                # Heuristic fallback if we aren't using a fixed threshold
                sorted_errors = np.sort(reconstruction_errors)
                idx = int(sorted_errors.size * (1.0 - self.tail_size_percent * 0.5))
                self.threshold_u = float(sorted_errors[idx])
                tail = (
                    reconstruction_errors[reconstruction_errors > self.threshold_u]
                    - self.threshold_u
                )

            if tail.size == 0:
                active_logger.error("Insufficient tail data. fitting dummy GPD.")
                self.gpd_params = (0.0, 0.0, 1.0)
                return

        # Fit GPD
        try:
            zeta, loc, eta = genpareto.fit(tail, loc=0)
            self.gpd_params = (float(zeta), float(loc), float(eta))
        except Exception as e:
            active_logger.error(f"GPD fitting failed: {e}. Using dummy params.")
            self.gpd_params = (0.0, 0.0, 1.0)

    def predict_probability_unknown(self, reconstruction_error: float) -> float:
        if self.gpd_params is None or self.threshold_u is None:
            return 0.0

        if reconstruction_error <= self.threshold_u:
            return 0.0

        zeta, loc, eta = self.gpd_params
        # CDF of GPD gives probability of being in the tail
        prob = genpareto.cdf(reconstruction_error - self.threshold_u, zeta, loc=loc, scale=eta)
        return float(prob)

    def to_payload(self) -> dict:
        return {
            "threshold_u": self.threshold_u,
            "gpd_params": self.gpd_params,
            "tail_size_percent": self.tail_size_percent,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "EVTModel":
        model = cls(payload["tail_size_percent"])
        model.threshold_u = float(payload["threshold_u"])
        model.gpd_params = tuple(payload["gpd_params"])
        return model


def save_evt_collection(
    evt_map: dict[int, EVTModel], filepath: Path | str, logger: logging.Logger | None = None
) -> None:
    active_logger = logger or logging.getLogger("EVT")
    payload = {}
    for label, model in evt_map.items():
        payload[int(label)] = model.to_payload()
    joblib.dump(payload, filepath)
    active_logger.info("Saved EVT collection with %s classes to %s", len(payload), filepath)


def load_evt_collection(filepath: Path | str) -> dict[int, EVTModel]:
    data = joblib.load(filepath)
    if not isinstance(data, dict):
        raise ValueError("EVT collection file is corrupted or not a dict.")
    return {int(k): EVTModel.from_payload(v) for k, v in data.items()}


def save_evt_meta(
    meta: dict, filepath: Path | str, logger: logging.Logger | None = None
) -> None:
    active_logger = logger or logging.getLogger("EVT")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    active_logger.info("Saved EVT meta to %s", filepath)


def load_evt_meta(filepath: Path | str) -> dict:
    with open(filepath, encoding="utf-8") as fh:
        return json.load(fh)
