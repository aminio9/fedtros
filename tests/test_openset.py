import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.evaluation import open_set as open_set_eval
from src.evaluation.open_set import calibrate_evt_thresholds, evaluate_open_set, fit_evt_models
from src.openset.evt import EVTModel, resolve_tail_fraction


class ConstantPrior(nn.Module):
    def forward(self, states):
        return torch.zeros(states.size(0), 2, device=states.device), torch.zeros(
            states.size(0), 2, device=states.device
        )


class ConstantRecognition(nn.Module):
    def forward(self, states, actions):
        _ = actions
        return torch.zeros(states.size(0), 2, device=states.device), torch.zeros(
            states.size(0), 2, device=states.device
        )


class PredictClassOne(nn.Module):
    num_actions = 2

    def forward(self, z, states):
        _ = z
        return torch.tensor([[0.0, 10.0]], device=states.device).repeat(states.size(0), 1)


class FeatureRuleQ(nn.Module):
    num_actions = 2

    def forward(self, z, states):
        _ = z
        class_one = states[:, 0] >= 0.5
        logits = torch.zeros(states.size(0), 2, device=states.device)
        logits[:, 0] = torch.where(
            class_one,
            torch.tensor(-5.0, device=states.device),
            torch.tensor(5.0, device=states.device),
        )
        logits[:, 1] = torch.where(
            class_one,
            torch.tensor(5.0, device=states.device),
            torch.tensor(-5.0, device=states.device),
        )
        return logits


class ZeroGenerator(nn.Module):
    def forward(self, z, actions):
        _ = actions
        return torch.zeros(z.size(0), 3, device=z.device)


def _toy_known_calibration_data(num_per_class=20):
    rows = []
    labels = []
    for idx in range(num_per_class):
        rows.append([0.08 + idx * 0.002, 0.0, 0.0])
        labels.append(0)
        rows.append([0.72 + idx * 0.002, 0.0, 0.0])
        labels.append(1)
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def test_evt_probability_increases_in_tail():
    model = EVTModel(tail_size_percent=0.5)
    model.fit(torch.tensor([0.1, 0.2, 0.4, 0.8]).numpy())

    assert model.predict_probability_unknown(0.0) == 0.0
    assert model.predict_probability_unknown(model.threshold_u + 0.1) > 0.0


def test_evt_tail_percent_one_means_one_percent_not_all_data():
    cfg = OmegaConf.create({"tail_percent": 1.0})
    fraction, source = resolve_tail_fraction(cfg)

    assert source == "tail_percent"
    assert fraction == pytest.approx(0.01)


def test_evt_tail_percent_ten_means_ten_percent():
    cfg = OmegaConf.create({"tail_percent": 10.0})
    fraction, source = resolve_tail_fraction(cfg)

    assert source == "tail_percent"
    assert fraction == pytest.approx(0.10)


def test_evt_legacy_tail_size_percent_requires_semantics_when_ambiguous():
    cfg = OmegaConf.create({"tail_size_percent": 1.0})

    with pytest.raises(ValueError, match="ambiguous"):
        resolve_tail_fraction(cfg)


def test_evt_invalid_tail_values_fail():
    with pytest.raises(ValueError):
        resolve_tail_fraction(OmegaConf.create({"tail_fraction": 0.0}))
    with pytest.raises(ValueError):
        resolve_tail_fraction(OmegaConf.create({"tail_percent": 101.0}))


def test_open_set_missing_evt_model_is_unknown(tmp_path):
    metrics = evaluate_open_set(
        features=torch.ones(2, 3),
        labels=torch.tensor([-1, -1]),
        batch_size=2,
        prior_net=ConstantPrior(),
        recognition_net=ConstantRecognition(),
        value_net_main=PredictClassOne(),
        generation_net=ZeroGenerator(),
        evt_models={0: EVTModel(0.5)},
        evt_meta={"global_delta": 0.5},
        class_names={0: "known_0", 1: "known_1"},
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert metrics["openset_unknown_recall"] == 1.0
    assert metrics["openset_missing_evt_model_count"] == 2.0
    assert "openset_global_delta" in metrics

    scores = pd.read_csv(tmp_path / "open_set_scores.csv")
    assert {"y_true", "raw_pred", "y_pred", "unknown_score", "is_unknown"}.issubset(
        scores.columns
    )

    open_set_metrics = json.loads((tmp_path / "open_set_metrics.json").read_text(encoding="utf-8"))
    assert open_set_metrics["openset_global_delta"] == metrics["openset_global_delta"]

    before_cm = pd.read_csv(tmp_path / "before_osr_confusion_matrix.csv", index_col=0)
    after_cm = pd.read_csv(tmp_path / "after_osr_confusion_matrix.csv", index_col=0)
    assert before_cm.shape == (3, 3)
    assert after_cm.shape == (3, 3)
    assert before_cm.loc["Unknown", "known_1"] == 2
    assert after_cm.loc["Unknown", "Unknown"] == 2


def test_open_set_eval_uses_configured_unknown_ids_and_threshold(tmp_path):
    evt_cfg = OmegaConf.create(
        {
            "decision_threshold": 0.5,
            "error_scale_factor": 1.0,
            "unknown_label_id": -7,
            "open_set_label_id": 77,
        }
    )

    metrics = evaluate_open_set(
        features=torch.ones(2, 3),
        labels=torch.tensor([-7, -7]),
        batch_size=2,
        prior_net=ConstantPrior(),
        recognition_net=ConstantRecognition(),
        value_net_main=PredictClassOne(),
        generation_net=ZeroGenerator(),
        evt_models={0: EVTModel(0.5)},
        evt_meta={},
        class_names={0: "known_0", 1: "known_1"},
        output_dir=tmp_path,
        device=torch.device("cpu"),
        evt_cfg=evt_cfg,
    )

    assert metrics["openset_unknown_recall"] == 1.0

    scores = pd.read_csv(tmp_path / "open_set_scores.csv")
    assert scores["y_true"].tolist() == [77, 77]
    assert scores["y_pred"].tolist() == [77, 77]


def test_open_set_eval_keeps_scores_equal_to_threshold_known(tmp_path):
    evt_model = EVTModel(0.5)
    evt_model.threshold_u = 1.0
    evt_model.gpd_params = (0.0, 0.0, 1.0)

    metrics = evaluate_open_set(
        features=torch.zeros(2, 3),
        labels=torch.tensor([1, -1]),
        batch_size=2,
        prior_net=ConstantPrior(),
        recognition_net=ConstantRecognition(),
        value_net_main=PredictClassOne(),
        generation_net=ZeroGenerator(),
        evt_models={1: evt_model},
        evt_meta={"global_delta": 0.0},
        class_names={0: "known_0", 1: "known_1"},
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    scores = pd.read_csv(tmp_path / "open_set_scores.csv")
    assert scores["unknown_score"].tolist() == [0.0, 0.0]
    assert scores["y_pred"].tolist() == [1, 1]
    assert metrics["openset_known_rejection_rate"] == 0.0
    assert metrics["openset_unknown_recall"] == 0.0

    sensitivity = pd.read_csv(tmp_path / "open_set_threshold_sensitivity.csv")
    threshold_zero = sensitivity.loc[sensitivity["threshold"].eq(0.0)].iloc[0]
    assert threshold_zero["known_rejection_rate"] == 0.0
    assert threshold_zero["unknown_recall"] == 0.0


def test_evt_fit_calibrate_evaluate_end_to_end_with_synthetic_reconstruction_errors(tmp_path):
    evt_cfg = OmegaConf.create(
        {
            "tail_fraction": 0.5,
            "min_errors_per_class": 4,
            "min_tail_size": 2,
            "target_known_fpr": 0.1,
            "threshold_mode": "validation_known_fpr",
            "decision_threshold": 0.5,
            "fixed_threshold": 0.5,
            "unknown_label_id": -1,
            "open_set_label_id": 99,
            "error_scale_factor": 1.0,
        }
    )
    prior_net = ConstantPrior()
    recognition_net = ConstantRecognition()
    value_net_main = FeatureRuleQ()
    generation_net = ZeroGenerator()
    device = torch.device("cpu")
    calibration_features, calibration_labels = _toy_known_calibration_data()

    evt_models = fit_evt_models(
        features=calibration_features,
        labels=calibration_labels,
        batch_size=8,
        evt_cfg=evt_cfg,
        prior_net=prior_net,
        recognition_net=recognition_net,
        value_net_main=value_net_main,
        generation_net=generation_net,
        device=device,
    )
    assert set(evt_models) == {0, 1}
    for model in evt_models.values():
        assert model.threshold_u is not None
        assert model.gpd_params is not None
        assert np.isfinite(model.threshold_u)
        assert np.isfinite(model.gpd_params).all()

    evt_meta = calibrate_evt_thresholds(
        features=calibration_features,
        labels=calibration_labels,
        batch_size=8,
        evt_models=evt_models,
        evt_cfg=evt_cfg,
        prior_net=prior_net,
        recognition_net=recognition_net,
        value_net_main=value_net_main,
        generation_net=generation_net,
        device=device,
    )
    assert evt_meta["tail_source"] == "tail_fraction"
    assert evt_meta["unknown_label_id"] == -1
    assert evt_meta["open_set_label_id"] == 99
    assert 0.0 < evt_meta["global_delta"] < 1.0

    metrics = evaluate_open_set(
        features=torch.tensor(
            [[0.09, 0.0, 0.0], [0.74, 0.0, 0.0], [0.10, 5.0, 5.0], [0.80, 5.0, 5.0]],
            dtype=torch.float32,
        ),
        labels=torch.tensor([0, 1, -1, -1], dtype=torch.long),
        batch_size=4,
        prior_net=prior_net,
        recognition_net=recognition_net,
        value_net_main=value_net_main,
        generation_net=generation_net,
        evt_models=evt_models,
        evt_meta=evt_meta,
        class_names={0: "known_0", 1: "known_1"},
        output_dir=tmp_path,
        device=device,
        evt_cfg=evt_cfg,
    )

    assert metrics["openset_known_acc"] == 1.0
    assert metrics["openset_known_rejection_rate"] == 0.0
    assert metrics["openset_unknown_recall"] == 1.0
    assert metrics["openset_unknown_precision"] == 1.0
    assert metrics["openset_missing_evt_model_count"] == 0.0

    scores = pd.read_csv(tmp_path / "open_set_scores.csv")
    assert scores["y_true"].tolist() == [0, 1, 99, 99]
    assert scores["raw_pred"].tolist() == [0, 1, 0, 1]
    assert scores["y_pred"].tolist() == [0, 1, 99, 99]
    assert (scores.loc[scores["is_unknown"].eq(0), "unknown_score"] <= evt_meta["global_delta"]).all()
    assert (scores.loc[scores["is_unknown"].eq(1), "unknown_score"] > evt_meta["global_delta"]).all()

    after_cm = pd.read_csv(tmp_path / "after_osr_confusion_matrix.csv", index_col=0)
    assert after_cm.loc["known_0", "known_0"] == 1
    assert after_cm.loc["known_1", "known_1"] == 1
    assert after_cm.loc["Unknown", "Unknown"] == 2

    saved_metrics = json.loads((tmp_path / "open_set_metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics["openset_global_delta"] == pytest.approx(evt_meta["global_delta"])


def test_oscr_curve_does_not_require_numpy_trapz(monkeypatch):
    monkeypatch.delattr(open_set_eval.np, "trapz", raising=False)

    fpr, ccr, thresholds, auoscr = open_set_eval._compute_oscr_curve(
        y_true=np.array([0, 0, 99, 99]),
        y_raw_pred=np.array([0, 1, 0, 1]),
        y_scores=np.array([0.1, 0.4, 0.3, 0.8]),
        open_set_label_id=99,
    )

    assert fpr.shape == ccr.shape == thresholds.shape
    assert 0.0 <= auoscr <= 1.0


def test_oscr_curve_accepts_scores_equal_to_threshold():
    fpr, ccr, thresholds, auoscr = open_set_eval._compute_oscr_curve(
        y_true=np.array([0, 99]),
        y_raw_pred=np.array([0, 0]),
        y_scores=np.array([0.0, 1.0]),
        open_set_label_id=99,
    )

    best_zero_fpr = np.flatnonzero(fpr == 0.0)[np.argmax(ccr[fpr == 0.0])]
    assert ccr[best_zero_fpr] == 1.0
    assert thresholds[best_zero_fpr] == 0.0
    assert 0.0 <= auoscr <= 1.0
