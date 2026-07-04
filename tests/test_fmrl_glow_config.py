from pathlib import Path

import yaml


def test_fmrl_ava_glow_config_contract():
    cfg = yaml.safe_load(Path("src/configs/method/fmrl_ava_glow.yaml").read_text())
    training = cfg["training"]
    strategy = cfg["federated"]["strategy"]

    assert training["rl_mode"] == "contextual_bandit"
    assert training["gamma"] == 0.0
    assert training["epsilon_start"] <= 0.20
    assert training["epsilon_decay_rate"] == 0.98
    assert training["loss_weights"]["prior_kl"] == 0.10
    assert training["loss_weights"]["q_td"] == 0.10
    assert training["loss_weights"]["bandit_q"] == 0.75
    assert training["loss_weights"]["classification"] == 3.0
    assert training["classification_loss"]["focal_gamma"] == 1.0
    assert training["kl"]["free_nats"] == 1.0
    assert training["kl"]["warmup_steps"] == 1000
    assert strategy["server_optimizer"] == "none"
    assert strategy["critic_blend"] == 0.0
    assert strategy["critic_activation_round"] == 999999
    assert strategy["local_proximal_mu"] == 0.001
    assert strategy["profile_balance_strength"] == 0.0
    assert training["missing_class_gradient"]["enabled"] is True
    assert training["missing_class_gradient"]["q_absent_mode"] == "ignore"


def test_fmrl_ava_glow_stable_is_fedavg_prior_config():
    cfg = yaml.safe_load(Path("src/configs/method/fmrl_ava_glow_stable.yaml").read_text())
    strategy = cfg["federated"]["strategy"]

    assert strategy["warmup_rounds"] == 100000
    assert strategy["min_selected_clients"] == "${federated.num_clients}"
    assert strategy["utility_strength"] == 0.0
    assert strategy["alignment_strength"] == 0.0
    assert strategy["drift_penalty_strength"] == 0.0
    assert strategy["critic_blend"] == 0.0
    assert strategy["server_optimizer"] == "none"


def test_fmrl_ava_glow_twa_config_contract():
    cfg = yaml.safe_load(Path("src/configs/method/fmrl_ava_glow_twa.yaml").read_text())
    strategy = cfg["federated"]["strategy"]

    assert cfg["experiment"]["method"] == "FMRL_AVA_GLOW_TWA"
    assert strategy["min_selected_clients"] == "${federated.num_clients}"
    assert strategy["sample_power"] == 0.75
    assert strategy["max_client_weight_fraction"] == 0.40
    assert strategy["utility_strength"] == 0.15
    assert strategy["alignment_strength"] == 0.03
    assert strategy["server_optimizer"] == "none"
    assert strategy["critic_blend"] == 0.0
    assert strategy["profile_balance_strength"] == 0.0


def test_fmrl_ava_glow_twa_uses_selective_latent_delta_scales():
    cfg = yaml.safe_load(Path("src/configs/method/fmrl_ava_glow_twa.yaml").read_text())
    scales = cfg["federated"]["strategy"]["module_delta_scales"]

    assert scales["prior_net"] == 0.25
    assert scales["recognition_net"] == 0.10
    assert scales["value_net_main"] == 1.0
    assert scales["generation_net"] == 0.0


def test_fmrl_ava_glow_stable_keeps_full_module_averaging_for_diagnosis():
    cfg = yaml.safe_load(Path("src/configs/method/fmrl_ava_glow_stable.yaml").read_text())
    scales = cfg["federated"]["strategy"]["module_delta_scales"]

    assert scales == {
        "prior_net": 1.0,
        "recognition_net": 1.0,
        "value_net_main": 1.0,
        "generation_net": 1.0,
    }
