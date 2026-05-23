import json

import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.evaluation.openset_eval import evaluate_open_set
from src.openset.evt import EVTModel


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


class ZeroGenerator(nn.Module):
    def forward(self, z, actions):
        _ = actions
        return torch.zeros(z.size(0), 3, device=z.device)


def test_evt_probability_increases_in_tail():
    model = EVTModel(tail_size_percent=0.5)
    model.fit(torch.tensor([0.1, 0.2, 0.4, 0.8]).numpy())

    assert model.predict_probability_unknown(0.0) == 0.0
    assert model.predict_probability_unknown(model.threshold_u + 0.1) > 0.0


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
