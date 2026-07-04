from pathlib import Path

import yaml


def test_fedgpa_glow_method_config_contract():
    cfg = yaml.safe_load(Path("src/configs/method/fedgpa_glow.yaml").read_text())

    assert cfg["experiment"]["method"] == "FedGPA_GLOW"
    assert cfg["training"]["gamma"] == 0.0
    assert cfg["training"]["local_episodes_per_round"] == 5
    assert cfg["training"]["loss_weights"]["prototype"] == 0.05
    assert cfg["training"]["generator"]["enabled"] is False

    strategy = cfg["federated"]["strategy"]
    assert strategy["name"] == "fedgpa_glow"
    assert strategy["aggregation_strategy"] == "fedgpa_glow"
    assert strategy["fedgpa"]["mu"] == 0.5
    assert strategy["fedgpa"]["prototype_momentum"] == 0.80
    assert strategy["fedgpa"]["module_update_scales"]["prior_net"] == 0.50
    assert strategy["fedgpa"]["module_update_scales"]["recognition_net"] == 0.25
    assert strategy["fedgpa"]["module_update_scales"]["value_net_main"] == 1.0
    assert strategy["fedgpa"]["module_update_scales"]["generation_net"] == 0.0
