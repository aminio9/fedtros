from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.utils.config import validate_config


def _config_dir() -> str:
    return str((Path(__file__).resolve().parents[1] / "src" / "configs").resolve())


def test_hydra_config_loads_from_src_configs():
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="config_fl")

    assert cfg.model.state_dim == cfg.env_metadata.state_dim
    assert cfg.model.num_actions == cfg.env_metadata.num_actions
    assert cfg.server.min_fit_clients == cfg.preprocess.num_clients
    assert cfg.strategy.max_agents == cfg.preprocess.num_clients


def test_federated_client_count_drives_preprocessing_count():
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="config_fl", overrides=["federated.num_clients=20"])

    assert cfg.federated.num_clients == 20
    assert cfg.preprocess.num_clients == 20
    assert cfg.server.min_fit_clients == 20
    assert cfg.strategy.max_agents == 20


def test_validation_rejects_mismatched_client_counts():
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(
            config_name="config_fl",
            overrides=["federated.num_clients=10", "dataset.preprocessing.num_clients=3"],
        )

    with pytest.raises(ValueError, match=r"must match federated\.num_clients"):
        validate_config(cfg)
