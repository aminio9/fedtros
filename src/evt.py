import json
import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from scipy.stats import genpareto

logger = logging.getLogger("EVT")


class EVTModel:
    """Simple EVT wrapper around SciPy's Generalized Pareto fit."""

    def __init__(self, tail_size_percent: float):
        if not (0.0 < tail_size_percent < 1.0):
            raise ValueError("tail_size_percent must be between 0 and 1")
        self.tail_size_percent = float(tail_size_percent)
        self.threshold_u: float | None = None
        self.gpd_params: tuple[float, float, float] | None = None

    def fit(self, reconstruction_errors: np.ndarray) -> None:
        if reconstruction_errors.ndim != 1 or reconstruction_errors.size == 0:
            raise ValueError("reconstruction_errors must be a non-empty 1D array")

        sorted_errors = np.sort(reconstruction_errors)
        idx = int(sorted_errors.size * (1.0 - self.tail_size_percent))
        idx = min(max(idx, 0), sorted_errors.size - 2)
        self.threshold_u = float(sorted_errors[idx])
        tail = reconstruction_errors[reconstruction_errors > self.threshold_u] - self.threshold_u

        if tail.size == 0:
            logger.warning(
                "No tail data above threshold %.6f; lowering threshold heuristically.",
                self.threshold_u,
            )
            idx = int(sorted_errors.size * (1.0 - self.tail_size_percent * 0.5))
            idx = min(max(idx, 0), sorted_errors.size - 2)
            self.threshold_u = float(sorted_errors[idx])
            tail = reconstruction_errors[reconstruction_errors > self.threshold_u] - self.threshold_u
            if tail.size == 0:
                raise RuntimeError("Insufficient tail data for EVT fitting")

        zeta, loc, eta = genpareto.fit(tail, loc=0)
        self.gpd_params = (float(zeta), float(loc), float(eta))
        logger.info(
            "EVT fitted: u=%.6f, params=(shape=%.6f, loc=%.6f, scale=%.6f)",
            self.threshold_u,
            zeta,
            loc,
            eta,
        )

    def predict_probability_unknown(self, reconstruction_error: float) -> float:
        if self.gpd_params is None or self.threshold_u is None:
            raise RuntimeError("EVT model must be fitted before predicting")
        if reconstruction_error <= self.threshold_u:
            return 0.0
        zeta, loc, eta = self.gpd_params
        prob = genpareto.cdf(reconstruction_error - self.threshold_u, zeta, loc=loc, scale=eta)
        return float(prob)

    def to_payload(self) -> Dict[str, float]:
        if self.threshold_u is None or self.gpd_params is None:
            raise RuntimeError("EVT model must be fitted before serialization")
        return {
            "threshold_u": self.threshold_u,
            "gpd_params": self.gpd_params,
            "tail_size_percent": self.tail_size_percent,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, float]) -> "EVTModel":
        model = cls(payload["tail_size_percent"])
        model.threshold_u = float(payload["threshold_u"])
        model.gpd_params = tuple(payload["gpd_params"])
        return model


def save_evt_collection(evt_map: Dict[int, EVTModel], filepath: Path | str) -> None:
    payload = {}
    for label, model in evt_map.items():
        payload[int(label)] = model.to_payload()
    joblib.dump(payload, filepath)
    logger.info("Saved EVT collection with %s classes to %s", len(payload), filepath)


def load_evt_collection(filepath: Path | str) -> Dict[int, EVTModel]:
    data = joblib.load(filepath)
    if not isinstance(data, dict):
        raise ValueError("EVT collection file is corrupted or not a dict.")
    models = {int(k): EVTModel.from_payload(v) for k, v in data.items()}
    return models


def save_evt_meta(meta: Dict, filepath: Path | str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Saved EVT meta to %s", filepath)


def load_evt_meta(filepath: Path | str) -> Dict:
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data
