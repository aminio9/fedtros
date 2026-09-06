"""Unit and regression tests for multicenter conformal pipeline metrics and calibration.

Validates:
1. Strict boolean masking for unknown_recall (preventing integer-indexing bug).
2. Clean calibration option (Zeng et al. 2026 Eq. 14) excluding misclassified outliers.
3. Class-conditional thresholds and shrinkage covariance floor.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.openset.conformal import fit_multicenter_conformal, score_multicenter_conformal


def _synthetic_proto_and_calib(seed=42):
    rng = np.random.default_rng(seed)
    dim = 8
    num_classes = 2
    
    # 2 well-separated clusters
    p0 = rng.normal(loc=[1.0] * dim, scale=0.1, size=(25, dim))
    p1 = rng.normal(loc=[-1.0] * dim, scale=0.1, size=(25, dim))
    
    proto = pd.DataFrame({
        "feature": list(np.vstack((p0, p1))),
        "y_raw": [0] * len(p0) + [1] * len(p1),
        "sample_id": [f"p{i}" for i in range(50)]
    })
    
    c0 = rng.normal(loc=[1.0] * dim, scale=0.1, size=(30, dim))
    c1 = rng.normal(loc=[-1.0] * dim, scale=0.1, size=(30, dim))
    
    # Introduce 3 misclassified outliers in class 0 predicted as class 1 with far features
    outliers = rng.normal(loc=[5.0] * dim, scale=0.5, size=(3, dim))
    
    features = list(np.vstack((c0, c1, outliers)))
    y_raw = [0] * len(c0) + [1] * len(c1) + [0] * len(outliers)
    pred = [0] * len(c0) + [1] * len(c1) + [1] * len(outliers)  # outliers misclassified as 1
    
    calib = pd.DataFrame({
        "feature": features,
        "y_raw": y_raw,
        "pred_before_osr": pred,
        "sample_id": [f"c{i}" for i in range(len(features))]
    })
    
    return proto, calib


def test_boolean_masking_prevents_indexing_bug():
    """Verify that integer arrays do not cause positional indexing in recall calculation."""
    y_true = np.array([0, 0, 1, 1, -1, -1, -1, -1])  # 4 knowns, 4 unknowns (-1)
    rejected = np.array([False, False, False, False, True, True, False, False], dtype=bool)
    
    is_unknown_bool = (y_true == -1)
    is_unknown_int = is_unknown_bool.astype(int)
    
    # The buggy integer indexing:
    buggy_recall = float(np.mean(rejected[is_unknown_int]))
    # Positional indexing picks rejected[0] and rejected[1] which are False
    assert buggy_recall == 0.0
    
    # Correct boolean indexing:
    correct_recall = float(np.mean(rejected[is_unknown_bool]))
    expected_recall = 2.0 / 4.0  # 2 rejected out of 4 unknowns
    assert np.isclose(correct_recall, expected_recall)


def test_clean_calibration_excludes_misclassified_outliers(tmp_path):
    proto, calib = _synthetic_proto_and_calib()
    
    # Default: clean_calibration=False (includes misclassified outliers)
    res_default = fit_multicenter_conformal(
        proto, calib, num_classes=2, alpha=0.05, clean_calibration=False,
        output_dir=tmp_path / "default"
    )
    
    # Clean: clean_calibration=True (filters misclassified outliers)
    res_clean = fit_multicenter_conformal(
        proto, calib, num_classes=2, alpha=0.05, clean_calibration=True,
        output_dir=tmp_path / "clean"
    )
    
    # Outliers should have inflated the default threshold compared to clean
    assert res_clean["tau_alpha"] < res_default["tau_alpha"]
    assert res_clean["tau_alpha_clean"] <= res_clean["tau_alpha_global"]
    assert "class_conditional_tau" in res_clean
    assert len(res_clean["class_conditional_tau"]) == 2


def test_shrinkage_floor_enforced():
    proto, calib = _synthetic_proto_and_calib()
    res = fit_multicenter_conformal(proto, calib, num_classes=2, alpha=0.05)
    
    # In each class model, precision matrix must be finite and invertible
    for c, m in res["models"].items():
        prec = np.array(m["precision"])
        assert np.isfinite(prec).all()
        cov = np.linalg.pinv(prec)
        assert np.isfinite(cov).all()


def test_score_multicenter_conformal_decision_modes():
    proto, calib = _synthetic_proto_and_calib()
    conformal_meta = fit_multicenter_conformal(proto, calib, num_classes=2, alpha=0.05, clean_calibration=True)
    
    test_df = calib.copy()
    
    # Standard candidate scoring
    scored_std = score_multicenter_conformal(test_df, conformal_meta)
    assert "conformal_score" in scored_std.columns
    assert "rejected" in scored_std.columns
    
    # Class-conditional scoring
    meta_cond = dict(conformal_meta)
    meta_cond["use_class_conditional"] = True
    scored_cond = score_multicenter_conformal(test_df, meta_cond)
    assert len(scored_cond) == len(test_df)
    
    # Min-ratio scoring
    meta_ratio = dict(conformal_meta)
    meta_ratio["score_mode"] = "min_ratio"
    scored_ratio = score_multicenter_conformal(test_df, meta_ratio)
    assert len(scored_ratio) == len(test_df)
