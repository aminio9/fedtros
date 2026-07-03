from pathlib import Path

import yaml


def test_fmrl_ava_glow_config_contract():
    cfg = yaml.safe_load(Path("src/configs/method/fmrl_ava_glow.yaml").read_text())
    training = cfg["training"]
    strategy = cfg["federated"]["strategy"]

    assert training["rl_mode"] == "contextual_bandit"
    assert training["gamma"] == 0.0
    assert training["epsilon_start"] <= 0.30
    assert training["loss_weights"]["q_td"] == 0.25
    assert strategy["server_optimizer"] == "none"
    assert strategy["min_selected_clients"] == 9
    assert strategy["profile_balance_strength"] == 0.0
    assert "bandit_q" in training["loss_weights"]
    assert training["missing_class_gradient"]["enabled"] is True
