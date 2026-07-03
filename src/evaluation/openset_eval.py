"""Backward-compatible imports for the renamed open-set evaluator.

Use :mod:`src.evaluation.open_set` for new code.
"""

from src.evaluation.open_set import (
    calibrate_evt_thresholds,
    evaluate_open_set,
    fit_evt_models,
)

__all__ = [
    "calibrate_evt_thresholds",
    "evaluate_open_set",
    "fit_evt_models",
]
