import numpy as np
import pytest
import torch

from src.openset.scorers import (
    EnergyScorer,
    MahalanobisDistanceScorer,
    MSPScorer,
    NoRejectionScorer,
    PrototypeDistanceScorer,
    build_open_set_scorer_from_config,
    predict_known_unknown,
    select_validation_threshold,
)


def test_msp_and_energy_scores_rank_uncertain_logits_as_more_unknown():
    logits = torch.tensor([[8.0, 0.0], [0.0, 0.0]])

    msp_scores = MSPScorer().score(logits=logits)
    energy_scores = EnergyScorer().score(logits=logits)

    assert msp_scores[1] > msp_scores[0]
    assert energy_scores[1] > energy_scores[0]


def test_prototype_distance_scores_far_sample_higher():
    features = torch.tensor([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    labels = torch.tensor([0, 0, 1, 1])
    scorer = PrototypeDistanceScorer().fit(features, labels, known_labels=[0, 1])

    scores = scorer.score(torch.tensor([[0.05, 0.0], [12.0, 12.0]]))

    assert scores[1] > scores[0]


def test_mahalanobis_distance_scores_far_sample_higher_and_finite():
    features = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1])
    scorer = MahalanobisDistanceScorer(regularization=1e-3).fit(
        features,
        labels,
        known_labels=[0, 1],
    )

    scores = scorer.score(torch.tensor([[0.05, 0.0], [12.0, 12.0]]))

    assert np.isfinite(scores).all()
    assert scores[1] > scores[0]


def test_validation_threshold_uses_known_scores_only():
    scores = np.array([0.1, 0.2, 0.3, 10.0])
    labels = np.array([0, 0, 0, -1])

    threshold = select_validation_threshold(
        scores,
        labels,
        known_labels=[0],
        target_known_fpr=0.0,
    )

    assert threshold == pytest.approx(0.3)
    assert predict_known_unknown(scores, threshold).tolist() == [0, 0, 0, 1]


def test_validation_threshold_supports_fixed_and_lower_unknown_scores():
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    labels = np.array([0, 0, 0, -1])

    fixed = select_validation_threshold(
        scores,
        labels,
        mode="fixed",
        fixed_threshold=0.25,
    )
    lower_unknown = select_validation_threshold(
        scores,
        labels,
        target_known_fpr=0.0,
        score_direction="lower_unknown",
    )

    assert fixed == pytest.approx(0.25)
    assert lower_unknown == pytest.approx(0.7)
    assert predict_known_unknown(
        scores,
        lower_unknown,
        score_direction="lower_unknown",
    ).tolist() == [0, 0, 0, 1]


def test_no_rejection_scorer_never_rejects_with_infinite_threshold():
    scores = NoRejectionScorer().score(logits=torch.randn(3, 2))
    threshold = NoRejectionScorer().select_threshold()

    assert predict_known_unknown(scores, threshold).tolist() == [0, 0, 0]


def test_open_set_scorer_factory_builds_standalone_utilities():
    for name, expected_type in (
        ("msp", MSPScorer),
        ("energy", EnergyScorer),
        ("prototype_distance", PrototypeDistanceScorer),
        ("mahalanobis", MahalanobisDistanceScorer),
        ("no_rejection", NoRejectionScorer),
    ):
        scorer = build_open_set_scorer_from_config({"name": name})

        assert isinstance(scorer, expected_type)
        assert scorer.higher_is_unknown is True


def test_evt_reconstruction_is_not_a_standalone_scorer_factory():
    with pytest.raises(ValueError, match="CVAE-DQN EVT evaluation path"):
        build_open_set_scorer_from_config(
            {"name": "evt", "scorer": {"name": "evt_reconstruction"}}
        )
