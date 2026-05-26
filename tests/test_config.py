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
    assert cfg.paths.known_train_data.endswith("known_train.pt")
    assert cfg.evaluation.validation_data.endswith("validation.pt")


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


def test_experiment_config_uses_run_local_processed_dir():
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="config_fl", overrides=["experiment=exp1"])

    assert cfg.experiment.pipeline == "full"
    assert cfg.dataset.preprocessing.output_dir == "outputs/e1_FMRL_LA_iid_seed42/processed"
    assert cfg.dataset.preprocessing.iid is True
    assert cfg.federated.num_clients == 10
    assert cfg.evaluation.mode == "closed_set"


def test_open_set_experiment_uses_iid_run_local_processed_dir():
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="config_fl", overrides=["experiment=exp2"])

    assert cfg.experiment.pipeline == "full"
    assert cfg.dataset.preprocessing.output_dir == "outputs/e2_FMRL_LA_iid_seed42/processed"
    assert cfg.dataset.preprocessing.iid is True
    assert cfg.open_set.evt.enabled is True
    assert cfg.evaluation.mode == "open_set"


def test_method_overlay_updates_strategy_and_method():
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="config_fl", overrides=["experiment=exp3", "+method=fedavg"])

    assert cfg.experiment.method == "FedAvg"
    assert cfg.federated.strategy.name == "fedavg"
    assert cfg.federated.server.proximal_mu == 0.0


def test_suite_launcher_contains_child_commands():
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="config_fl", overrides=["experiment=all"])

    assert cfg.experiment.pipeline == "suite"
    assert len(cfg.experiment.suite_commands) == 6
    assert cfg.experiment.suite_commands[0][0] == "experiment=exp1"


def test_runtime_gpu_overlay_updates_device_preference():
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="config_fl", overrides=["experiment=ablation", "runtime=gpu"])

    assert cfg.device.prefer == "gpu"
    assert cfg.device.allow_cpu_fallback is False
    assert cfg.runtime.client_num_gpus == 1.0
    assert cfg.runtime.simulation_gpu_batches.enabled is True


def test_validation_rejects_cpu_fallback():
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="config_fl", overrides=["runtime.allow_cpu_fallback=true"])

    with pytest.raises(ValueError, match=r"Automatic CPU fallback is disabled"):
        validate_config(cfg)
