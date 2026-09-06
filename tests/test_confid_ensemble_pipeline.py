"""Unit tests for ConFID-faithful dual-path ensemble scoring and student intermediate layer tapping."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.student import StudentIDSModel
from src.openset.conformal import (
    fit_multicenter_conformal,
    normalize_features,
    score_multicenter_conformal,
)


def test_student_layer_tapping():
    """Verify that StudentIDSModel extracts representations from designated layers."""
    model = StudentIDSModel(
        input_dim=34,
        num_classes=4,
        hidden_dims=[512, 256, 128],
        activation="gelu",
        dropout=0.05,
        norm="layernorm",
        osr_enabled=True,
    )
    model.eval()

    x = torch.randn(10, 34)
    with torch.no_grad():
        h_l1 = model.extract_intermediate_features(x, layer="l1")
        assert h_l1.shape == (10, 512), f"Expected (10, 512), got {h_l1.shape}"

        h_l2 = model.extract_intermediate_features(x, layer="l2")
        assert h_l2.shape == (10, 256), f"Expected (10, 256), got {h_l2.shape}"

        h_l3 = model.extract_intermediate_features(x, layer="l3")
        assert h_l3.shape == (10, 128), f"Expected (10, 128), got {h_l3.shape}"

        h_all = model.extract_intermediate_features(x, layer="all")
        assert h_all.shape == (10, 512 + 256 + 128), f"Expected (10, 896), got {h_all.shape}"

        # Standard forward
        feat, logits = model(x)
        assert feat.shape == (10, 128)
        assert logits.shape == (10, 4)

        # OSR branch output
        labels = torch.randint(0, 4, (10,))
        osr_out = model.osr_score(x, labels, detach_features=True)
        assert "recon_error" in osr_out
        assert osr_out["recon_error"].shape == (10,)
        assert not torch.isnan(osr_out["recon_error"]).any()


def test_confid_ensemble_calibration_and_scoring():
    """Verify ConFID dual-path ensemble calibration and inference scoring."""
    np.random.seed(42)
    num_classes = 4
    dim = 512
    n_proto_per_class = 50
    n_calib_per_class = 40

    # Synthetic prototypes for each class
    proto_rows = []
    class_centers = {c: np.random.randn(dim) * 2.0 for c in range(num_classes)}

    for c in range(num_classes):
        center = class_centers[c]
        for i in range(n_proto_per_class):
            feat = center + np.random.randn(dim) * 0.3
            proto_rows.append({
                "sample_id": f"p_{c}_{i}",
                "y_raw": c,
                "feature": feat,
            })
    df_proto = pd.DataFrame(proto_rows)

    # Synthetic calibration data
    calib_rows = []
    for c in range(num_classes):
        center = class_centers[c]
        for i in range(n_calib_per_class):
            feat = center + np.random.randn(dim) * 0.3
            # Known samples have small reconstruction error ~ N(0.05, 0.01)
            rec_err = float(np.clip(np.random.normal(0.05, 0.01), 0.01, 0.2))
            calib_rows.append({
                "sample_id": f"c_{c}_{i}",
                "y_raw": c,
                "pred_before_osr": c,
                "feature": feat,
                "recon_error": rec_err,
            })
    df_calib = pd.DataFrame(calib_rows)

    # 1. Fit ConFID ensemble
    alpha = 0.05
    meta = fit_multicenter_conformal(
        df_proto,
        df_calib,
        num_classes=num_classes,
        alpha=alpha,
        seed=42,
        score_mode="confid_ensemble",
        ensemble_recon_weight=0.5,
    )

    assert "tau_alpha" in meta
    assert np.isfinite(meta["tau_alpha"])
    assert "ensemble_stats" in meta
    assert meta["ensemble_stats"]["has_recon"] is True

    # 2. Score calibration samples to verify empirical false alarm rate <= alpha
    calib_scored = score_multicenter_conformal(df_calib, meta)
    assert "conformal_score" in calib_scored.columns
    assert "score_mah" in calib_scored.columns
    assert "score_rec" in calib_scored.columns
    assert "rejected" in calib_scored.columns

    kfr = float(calib_scored["rejected"].mean())
    # Split conformal calibration bounds empirical false alarm rate near alpha
    assert kfr <= alpha + 0.03, f"Empirical KFR {kfr:.4f} exceeded expected alpha {alpha:.4f}"

    # 3. Test queries: Known vs Unknown anomaly
    test_rows = []
    # Known query
    test_rows.append({
        "sample_id": "test_known",
        "y_raw": 0,
        "pred_before_osr": 0,
        "feature": class_centers[0] + np.random.randn(dim) * 0.2,
        "recon_error": 0.04,  # low reconstruction error
    })
    # Unknown anomaly (far away, large recon error)
    test_rows.append({
        "sample_id": "test_unknown",
        "y_raw": -1,
        "pred_before_osr": 0,  # misclassified as class 0
        "feature": np.random.randn(dim) * 10.0,  # far from any center
        "recon_error": 2.50,  # massive reconstruction error
    })
    df_test = pd.DataFrame(test_rows)
    test_scored = score_multicenter_conformal(df_test, meta)

    # Known should not be rejected, unknown should be rejected
    assert test_scored.loc[0, "rejected"] == False, "Known test sample should be accepted"
    assert test_scored.loc[1, "rejected"] == True, "Anomalous unknown test sample should be rejected"
