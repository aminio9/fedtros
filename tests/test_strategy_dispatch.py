import pytest
import torch
from types import SimpleNamespace
from omegaconf import OmegaConf

import src.federated.server as server


def _cfg(strategy: str):
    return OmegaConf.create(
        {
            "strategy": {"name": strategy},
            "method": {"canonical": strategy in {"fedtros", "fedtros_mc"}},
            "server": {
                "fraction_fit": 1.0,
                "fraction_evaluate": 1.0,
                "min_fit_clients": 1,
                "min_evaluate_clients": 1,
                "min_available_clients": 1,
                "num_rounds": 1,
                "proximal_mu": 0.1,
            },
            "training": {"local_epochs": 1, "batch_size": 8},
            "dataset": {"name": "test", "preprocessing": {"alpha": 0.5}},
        }
    )


@pytest.mark.parametrize("alias", ["fedtros", "fedtros_mc", "fedtros_pr", "fedtros_pr_legacy"])
def test_fedtros_aliases_resolve_to_fedtros_strategy(monkeypatch, alias):
    class DummyFedTROS:
        def __init__(self, *args, **kwargs):
            self.cfg = kwargs["cfg"]

    monkeypatch.setattr(server, "FedTROSStrategy", DummyFedTROS)
    monkeypatch.setattr(server, "_initial_parameters_from_checkpoint", lambda cfg, device: None)
    monkeypatch.setattr(server, "make_central_evaluate_fn", lambda *args, **kwargs: None)

    result = server.get_strategy(_cfg(alias))

    assert isinstance(result, DummyFedTROS)


def test_unknown_strategy_fails_instead_of_becoming_fedavg():
    with pytest.raises(ValueError, match="Unsupported federated strategy"):
        server.normalize_strategy_name("fedgpa")


def test_gpu_fast_runtime_profile_is_explicit():
    cfg = OmegaConf.load("src/configs/runtime/gpu_fast.yaml")
    assert cfg.device_prefer == "gpu"
    assert cfg.client_device_residency == "resident"
    assert cfg.local_batch_size == 512
    assert cfg.simulation_gpu_batches.batch_size == 2
    assert cfg.empty_cache_after_client is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable on this host")
def test_cuda_fit_device_metrics_report_actual_devices():
    client = server.FlowerClient.__new__(server.FlowerClient)
    client.cid = "test"
    client.agent = SimpleNamespace(student_model=torch.nn.Linear(2, 2, device="cuda"))
    client.features = torch.randn(2, 2, device="cuda")
    client._log_device_state = False
    metrics = client._attach_runtime_device_metrics({}, torch.device("cuda"))
    assert metrics["runtime/model_device"] == "cuda:0"
    assert metrics["runtime/feature_device"] == "cuda:0"
    assert metrics["runtime/batch_device"] == "cuda"
    assert metrics["runtime/cuda_memory_allocated_mb"] >= 0.0
