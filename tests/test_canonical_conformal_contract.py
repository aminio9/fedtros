"""Contract tests for the canonical adaptive multicenter conformal detector.

These tests intentionally use only known prototype/calibration records; no
unknown labels are available to model fitting or calibration.
"""

import math

import numpy as np
import pandas as pd

from src.openset.conformal import fit_multicenter_conformal, normalize_features


def _known_frames(seed=17):
    rng = np.random.default_rng(seed)
    # Two well-separated known classes in a low-dimensional hidden space.
    p0 = rng.normal(loc=(2.0, 0.0, 0.0), scale=0.15, size=(18, 3))
    p1 = rng.normal(loc=(-2.0, 0.0, 0.0), scale=0.15, size=(18, 3))
    # Keep m large enough that ceil((m+1)*.95) is an in-range order statistic.
    c0 = rng.normal(loc=(2.0, 0.0, 0.0), scale=0.15, size=(20, 3))
    c1 = rng.normal(loc=(-2.0, 0.0, 0.0), scale=0.15, size=(20, 3))
    proto = pd.DataFrame({"feature": list(np.vstack((p0, p1))),
                          "y_raw": [0] * len(p0) + [1] * len(p1),
                          "sample_id": [f"p{i}" for i in range(36)]})
    calib = pd.DataFrame({"feature": list(np.vstack((c0, c1))),
                          "y_raw": [0] * len(c0) + [1] * len(c1),
                          "pred_before_osr": [0] * len(c0) + [1] * len(c1),
                          "sample_id": [f"c{i}" for i in range(40)]})
    return proto, calib


def test_normalize_features_is_unit_length_and_finite():
    h = np.array([[3.0, 4.0], [0.0, 0.0], [1e-14, -2e-14]])
    z = normalize_features(h)
    norms = np.linalg.norm(z, axis=1)
    assert np.isfinite(z).all()
    assert np.isclose(norms[0], 1.0, atol=1e-7)
    # Zero/near-zero vectors must not produce NaN or inf.
    assert norms[1] == 0.0
    assert norms[2] < 1e-5


def test_conformal_order_statistic_and_scores_are_reproducible(tmp_path):
    proto, calib = _known_frames()
    a = fit_multicenter_conformal(proto, calib, num_classes=2, alpha=0.05,
                                   seed=123, output_dir=tmp_path / "a")
    b = fit_multicenter_conformal(proto, calib, num_classes=2, alpha=0.05,
                                   seed=123, output_dir=tmp_path / "b")

    assert a["m"] == len(calib)
    assert a["k_alpha"] == math.ceil((len(calib) + 1) * 0.95)
    assert a["tau_alpha"] == b["tau_alpha"]
    assert a["k_alpha"] == b["k_alpha"]
    for cls in a["models"]:
        assert np.allclose(a["models"][cls]["centers"], b["models"][cls]["centers"])
        precision = np.asarray(a["models"][cls]["precision"])
        assert np.isfinite(precision).all()

    rows = pd.read_csv(tmp_path / "a" / "osr" / "calibration_scores.csv")
    assert set(rows["sample_id"]) == set(calib["sample_id"])
    assert not set(rows["sample_id"]) & {"u0", "u1"}
    scores = np.sort(rows["nonconformity_score"].to_numpy(float))
    assert np.isclose(a["tau_alpha"], scores[a["k_alpha"] - 1])
    internal = pd.read_csv(tmp_path / "a" / "osr" / "prototype_internal_split.csv")
    assert set(internal["subset"]) == {"proto-fit", "proto-val"}
    assert len(internal) == len(proto)
    assert a["prototype_internal_split"]["fit_count"] + a["prototype_internal_split"]["val_count"] == len(proto)

    # Independent Mahalanobis recomputation from exported model state.
    for _, row in rows.iterrows():
        sample = calib.loc[calib["sample_id"] == row["sample_id"]].iloc[0]
        z = normalize_features(np.asarray(sample["feature"])[None, :])[0]
        cls = int(row["candidate_pred"])
        centers = np.asarray(a["models"][cls]["centers"], dtype=float)
        precision = np.asarray(a["models"][cls]["precision"], dtype=float)
        diff = z[None, :] - centers
        expected = np.min(np.einsum("ki,ij,kj->k", diff, precision, diff))
        assert np.isclose(float(row["nonconformity_score"]), expected, rtol=1e-6, atol=1e-8)


def test_canonical_artifacts_have_no_boundary_prototype_rows(tmp_path):
    proto, calib = _known_frames()
    fit_multicenter_conformal(proto, calib, num_classes=2, seed=9,
                               output_dir=tmp_path)
    assignments = pd.read_csv(tmp_path / "osr" / "prototype_assignments.csv")
    assert not {"negative_prototype", "boundary_prototype", "synthetic_boundary"} & set(assignments.columns)
    centers = np.load(tmp_path / "osr" / "prototype_centers.npz")
    assert all("negative" not in k.lower() and "boundary" not in k.lower() for k in centers.files)
