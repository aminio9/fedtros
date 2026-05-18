from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import flwr as fl
import torch
from flwr.common import Context
from omegaconf import DictConfig

from src.federated.client import FlowerClient
from src.federated.server import get_effective_num_rounds, get_strategy, run_server
from src.tracking.local import LocalRunTracker
from src.utils.config import resolve_path
from src.utils.utils import resolve_device_from_config, set_seed

logger = logging.getLogger(__name__)


def _client_data_path(cfg: DictConfig, project_root: Path, client_id: int) -> Path:
    path = resolve_path(
        project_root,
        Path(str(cfg.dataset.preprocessing.output_dir)) / f"client_{client_id}_train.pt",
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Client {client_id} tensor not found: {path}. Run scripts/preprocess.py first."
        )
    return path


def run_federated_simulation(
    cfg: DictConfig,
    *,
    project_root: Path,
    tracker: LocalRunTracker | None = None,
) -> dict[str, Any]:
    def client_fn(context: Context) -> fl.client.Client:
        partition_id = int(context.node_config["partition-id"])
        client_id = partition_id + 1
        client_seed = int(cfg.seed) + client_id
        set_seed(
            client_seed,
            deterministic=bool(cfg.device.deterministic),
            benchmark=bool(cfg.device.benchmark),
            use_deterministic_algorithms=bool(cfg.device.use_deterministic_algorithms),
        )
        device = torch.device("cpu")
        if str(cfg.device.prefer).lower() == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        data_path = _client_data_path(cfg, project_root, client_id)
        client = FlowerClient(
            cid=str(client_id),
            cfg=cfg,
            data_path=str(data_path),
            device=device,
        )
        return client.to_client()

    client_resources = {
        "num_cpus": float(cfg.federated.client_resources.num_cpus),
        "num_gpus": float(cfg.federated.client_resources.num_gpus),
    }
    logger.info(
        "Starting Flower simulation | clients=%d | logical_rounds=%d | flower_rounds=%d | strategy=%s",
        int(cfg.federated.num_clients),
        int(cfg.federated.num_rounds),
        get_effective_num_rounds(cfg),
        str(cfg.federated.strategy.name),
    )
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=int(cfg.federated.num_clients),
        config=fl.server.ServerConfig(num_rounds=get_effective_num_rounds(cfg)),
        strategy=get_strategy(cfg),
        client_resources=client_resources,
        ray_init_args={
            "include_dashboard": False,
            "log_to_driver": False,
            "configure_logging": True,
            "logging_level": logging.ERROR,
        },
    )
    summary = {
        "federated/rounds": int(cfg.federated.num_rounds),
        "federated/flower_rounds": get_effective_num_rounds(cfg),
        "federated/num_clients": int(cfg.federated.num_clients),
        "history": str(history),
    }
    if tracker:
        tracker.log_metrics(summary)
        tracker.write_json("federated_summary.json", summary)
    return summary


def run_federated_server(
    cfg: DictConfig,
    *,
    device: torch.device | None = None,
) -> None:
    run_server(cfg, device=device if device is not None else resolve_device_from_config(cfg))


def run_federated_client(
    cfg: DictConfig,
    *,
    project_root: Path,
    device: torch.device | None = None,
) -> None:
    client_id = int(cfg.federated.client_id)
    data_path = resolve_path(project_root, cfg.federated.client_data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Client tensor not found: {data_path}")
    client = FlowerClient(
        cid=str(client_id),
        cfg=cfg,
        data_path=str(data_path),
        device=device if device is not None else resolve_device_from_config(cfg),
    )
    fl.client.start_client(
        server_address=str(cfg.federated.server.address),
        client=client.to_client(),
    )
