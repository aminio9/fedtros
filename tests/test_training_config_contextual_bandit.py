"""Config contract for the contextual-bandit local RL defaults."""

from pathlib import Path

import yaml


def test_training_default_contextual_bandit_contract():
    cfg = yaml.safe_load(Path("src/configs/training/default.yaml").read_text())

    assert cfg["rl_mode"] == "contextual_bandit"
    assert cfg["gamma"] == 0.0
    assert "bandit_q" in cfg["loss_weights"]
    assert cfg["loss_weights"]["q_td"] == 0.25
    assert cfg["loss_weights"]["bandit_q"] == 1.0
    assert cfg["classification_loss"]["name"] == "focal"
    assert cfg["classification_loss"]["focal_gamma"] == 1.5
    assert cfg["classification_loss"]["use_class_weights"] is True
    assert cfg["imbalance"]["class_balanced_sampling"] is True
    assert cfg["imbalance"]["weighted_reward"] is True
    assert cfg["epsilon_start"] <= 0.30
    assert cfg["epsilon_end"] == 0.02


def test_training_default_has_missing_class_q_absent_mode():
    cfg = yaml.safe_load(Path("src/configs/training/default.yaml").read_text())
    assert cfg["missing_class_gradient"]["q_absent_mode"] in {"ignore", "weak_negative", "off"}
