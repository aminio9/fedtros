from types import SimpleNamespace

import numpy as np
import torch

from src.models.student import StudentIDSModel
from src.openset.feature_evt import (
    evaluate_feature_evt_open_set,
    fit_student_feature_evt_models,
    mahalanobis_diag,
)


def _cfg(**overrides):
    base = dict(
        backend="student_feature_evt",
        score="mahalanobis_feature_distance",
        classwise=True,
        threshold_method="quantile",
        tail_size_percent=0.10,
        mef_min_quantile=0.70,
        mef_max_quantile=0.98,
        mef_num_candidates=20,
        min_errors_per_class=10,
        min_tail_size=5,
        target_known_fpr=0.05,
        fit_correct_only=False,
        unknown_label_id=-1,
        open_set_label_id=99,
        covariance="diagonal_shrinkage",
        covariance_eps=1e-4,
        error_scale_factor=1.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class ToyStudent(torch.nn.Module):
    num_classes = 2

    def eval(self):
        return self

    def forward(self, x):
        # Features are the first two coordinates; logits split by x coordinate.
        h = x[:, :2]
        logits = torch.stack([-x[:, 0], x[:, 0]], dim=1)
        return h, logits


def test_mahalanobis_diag_is_finite_with_tiny_variance():
    x = np.array([[1.0, 2.0]])
    center = np.array([1.0, 1.0])
    variance = np.array([0.0, 0.0])
    d = mahalanobis_diag(x, center, variance, eps=1e-4)
    assert np.isfinite(d).all()
    assert d.shape == (1,)


def test_feature_evt_rejects_far_unknowns(tmp_path):
    torch.manual_seed(7)
    c0 = torch.randn(80, 2) * 0.08 + torch.tensor([-2.0, 0.0])
    c1 = torch.randn(80, 2) * 0.08 + torch.tensor([2.0, 0.0])
    x_cal = torch.cat([c0, c1], dim=0)
    y_cal = torch.cat([torch.zeros(80), torch.ones(80)]).long()
    model = ToyStudent()
    cfg = _cfg()
    boundaries, meta, _df = fit_student_feature_evt_models(
        x_cal,
        y_cal,
        batch_size=32,
        student_model=model,
        evt_cfg=cfg,
        device=torch.device("cpu"),
    )
    unknown = torch.randn(40, 2) * 0.05 + torch.tensor([0.0, 4.0])
    x_test = torch.cat([c0[:20], c1[:20], unknown], dim=0)
    y_test = torch.cat([torch.zeros(20), torch.ones(20), torch.full((40,), -1)]).long()
    metrics = evaluate_feature_evt_open_set(
        x_test,
        y_test,
        batch_size=16,
        student_model=model,
        feature_boundaries=boundaries,
        evt_meta=meta,
        class_names={0: "A", 1: "B"},
        output_dir=tmp_path,
        device=torch.device("cpu"),
        evt_cfg=cfg,
    )
    assert metrics["openset_unknown_recall"] > 0.9
    assert metrics["openset_auroc"] > 0.9
    assert (tmp_path / "feature_evt_thresholds.json").exists()


def test_dkd_student_feature_evt_uses_classifier_features_only():
    model = StudentIDSModel(input_dim=4, num_classes=2, hidden_dims=[8, 4])
    x = torch.randn(12, 4)
    h, logits = model(x)
    assert h.shape[0] == 12
    assert logits.shape == (12, 2)
    assert not hasattr(model, "decoder")
