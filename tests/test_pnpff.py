import numpy as np
import pytest
import torch

from src.openset.pnpff import (
    PNPFFConfig,
    PNPFFModel,
    fit_pnpff_detector,
    stratified_fit_calibration_split,
)
from src.openset.digos_eval import _oscr_score


def test_pnpff_equation_one_distance():
    x = torch.tensor([[[1.0, 2.0]]])
    p = torch.tensor([[[3.0, 1.0]]])
    # (||x-p||^2 / m) - x.p = (5 / 2) - 5 = -2.5
    assert torch.allclose(PNPFFModel.distance(x, p), torch.tensor([[-2.5]]))


def test_oscr_is_perfect_for_separated_correct_known_samples():
    y_true = np.asarray([0, 1, 99, 99])
    candidate = np.asarray([0, 1, 0, 1])
    unknown_score = np.asarray([0.1, 0.2, 0.8, 0.9])
    assert _oscr_score(y_true, candidate, unknown_score, open_set_label_id=99) == pytest.approx(1.0)


def test_pnpff_equations_two_and_twenty_eight():
    cfg = PNPFFConfig(feature_dim=2, num_positive_prototypes=2, eta=0.5, omega=0.5)
    model = PNPFFModel(2, 2, cfg)
    with torch.no_grad():
        model.projection.weight.copy_(torch.eye(2))
        model.positive_prototypes.copy_(
            torch.tensor([[[1.0, 0.0], [1.2, 0.0]], [[0.0, 1.0], [0.0, 1.2]]])
        )
        model.negative_prototypes.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    out = model.probabilities(torch.tensor([[1.0, 0.0]]))
    _, all_positive, _ = model.distances(torch.tensor([[1.0, 0.0]]))
    expected_positive = torch.softmax(-all_positive, dim=1)
    expected_negative = torch.softmax(out["negative_distances"], dim=1)
    expected_fused = 0.5 * expected_positive + 0.5 * expected_negative
    assert torch.allclose(out["positive_probabilities"], expected_positive)
    assert torch.allclose(out["negative_probabilities"], expected_negative)
    assert torch.allclose(out["fused_scores"], expected_fused)
    assert out["predicted_class"].item() == 0


def test_pnpff_structure_losses_and_trainable_radius():
    cfg = PNPFFConfig(feature_dim=4, num_positive_prototypes=7, epochs=1, batch_size=4)
    model = PNPFFModel(4, 3, cfg)
    assert model.positive_prototypes.shape == (3, 7, 4)
    assert model.negative_prototypes.shape == (3, 4)
    assert model.radius.requires_grad
    losses = model.positive_loss(torch.randn(6, 4), torch.tensor([0, 1, 2, 0, 1, 2]))
    losses["total"].backward()
    assert model.positive_prototypes.grad is not None
    assert model.radius.grad is not None


def test_stratified_split_is_deterministic_and_known_only():
    labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
    first = stratified_fit_calibration_split(labels, fit_fraction=0.7, seed=9)
    second = stratified_fit_calibration_split(labels, fit_fraction=0.7, seed=9)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert set(labels[first[0]].tolist()) == {0, 1, 2}
    assert set(labels[first[1]].tolist()) == {0, 1, 2}
    with pytest.raises(ValueError, match="known classes only"):
        stratified_fit_calibration_split(torch.tensor([0, 0, 1, 1, -1]))
    with pytest.raises(ValueError, match="two samples per class"):
        stratified_fit_calibration_split(torch.tensor([0, 0, 1]))


def test_fit_detector_uses_fixed_and_known_fpr_thresholds():
    torch.manual_seed(2)
    features = torch.cat([torch.randn(12, 4) * 0.1 - 1.0, torch.randn(12, 4) * 0.1 + 1.0])
    labels = torch.tensor([0] * 12 + [1] * 12)
    fit_idx, cal_idx = stratified_fit_calibration_split(labels, seed=2)
    cfg = PNPFFConfig(feature_dim=4, epochs=2, batch_size=8, learning_rate=0.01, tau=0.5)
    detector = fit_pnpff_detector(
        features[fit_idx], labels[fit_idx], features[cal_idx], labels[cal_idx],
        num_classes=2, cfg=cfg, device=torch.device("cpu"),
    )
    assert detector.threshold == 0.5
    assert detector.model.positive_prototypes.shape == (2, 7, 4)
    scores = detector.predict_features(features)
    assert scores["unknown_score"].shape == (24,)
    assert np.all((scores["unknown_score"] >= 0.0) & (scores["unknown_score"] <= 1.0))

    quantile_cfg = PNPFFConfig(
        feature_dim=4, epochs=1, batch_size=8, learning_rate=0.01,
        threshold_mode="known_fpr", target_known_fpr=0.10,
    )
    quantile_detector = fit_pnpff_detector(
        features[fit_idx], labels[fit_idx], features[cal_idx], labels[cal_idx],
        num_classes=2, cfg=quantile_cfg, device=torch.device("cpu"),
    )
    assert 0.0 <= quantile_detector.threshold <= 1.0
