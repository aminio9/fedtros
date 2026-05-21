from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import flwr as fl
import pandas as pd
import torch
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    ReconnectIns,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import Server
from omegaconf import DictConfig, OmegaConf

from src.artifacts.communication import build_communication_metrics
from src.federated.client import FlowerClient
from src.federated.server import (
    get_effective_num_rounds,
    get_strategy,
    init_global_agent_ref,
    run_server,
)
from src.tracking.local import LocalRunTracker
from src.utils.config import resolve_path
from src.utils.utils import resolve_device_from_config, set_seed

logger = logging.getLogger(__name__)


def _resolve_runtime_config(cfg: DictConfig) -> DictConfig:
    """Freeze Hydra interpolations before config crosses process boundaries."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


class _LegacyFlowerClientProxy(ClientProxy):
    """In-process Flower client proxy used for local simulations."""

    def __init__(self, client: FlowerClient):
        super().__init__(cid=str(client.cid))
        self.node_id = int(client.cid)
        self._client = client

    def get_properties(
        self, ins: GetPropertiesIns, timeout: float | None, group_id: int | None
    ) -> GetPropertiesRes:
        _ = ins, timeout, group_id
        return GetPropertiesRes(status=Status(code=Code.OK, message=""), properties={})

    def get_parameters(
        self, ins: GetParametersIns, timeout: float | None, group_id: int | None
    ) -> GetParametersRes:
        _ = timeout, group_id
        parameters = self._client.get_parameters(dict(ins.config))
        return GetParametersRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(parameters),
        )

    def fit(
        self, ins: FitIns, timeout: float | None, group_id: int | None
    ) -> FitRes:
        _ = timeout, group_id
        parameters, num_examples, metrics = self._client.fit(
            parameters_to_ndarrays(ins.parameters),
            dict(ins.config),
        )
        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(parameters),
            num_examples=int(num_examples),
            metrics=dict(metrics),
        )

    def evaluate(
        self, ins: EvaluateIns, timeout: float | None, group_id: int | None
    ) -> EvaluateRes:
        _ = timeout, group_id
        loss, num_examples, metrics = self._client.evaluate(
            parameters_to_ndarrays(ins.parameters),
            dict(ins.config),
        )
        return EvaluateRes(
            status=Status(code=Code.OK, message=""),
            loss=float(loss),
            num_examples=int(num_examples),
            metrics=dict(metrics),
        )

    def reconnect(
        self, ins: ReconnectIns, timeout: float | None, group_id: int | None
    ) -> DisconnectRes:
        _ = ins, timeout, group_id
        return DisconnectRes(reason="")


def _build_local_clients(
    cfg: DictConfig,
    *,
    project_root: Path,
) -> SimpleClientManager:
    client_manager = SimpleClientManager()
    for client_id in range(1, int(cfg.federated.num_clients) + 1):
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
        client_manager.register(_LegacyFlowerClientProxy(client))
    return client_manager


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


def _history_rows(history: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for round_num, value in getattr(history, "losses_distributed", []):
        rows.append(
            {
                "phase": "losses_distributed",
                "metric_name": "loss",
                "round": int(round_num),
                "metric_value": float(value),
            }
        )

    for phase, attr in (
        ("metrics_distributed_fit", "metrics_distributed_fit"),
        ("metrics_distributed", "metrics_distributed"),
        ("metrics_centralized", "metrics_centralized"),
    ):
        metrics = getattr(history, attr, {})
        if not isinstance(metrics, dict):
            continue
        for metric_name, series in metrics.items():
            for round_num, value in series:
                rows.append(
                    {
                        "phase": phase,
                        "metric_name": str(metric_name),
                        "round": int(round_num),
                        "metric_value": float(value),
                    }
                )

    return rows


def run_federated_simulation(
    cfg: DictConfig,
    *,
    project_root: Path,
    tracker: LocalRunTracker | None = None,
) -> dict[str, Any]:
    cfg = _resolve_runtime_config(cfg)
    init_global_agent_ref(cfg, resolve_device_from_config(cfg))
    client_manager = _build_local_clients(cfg, project_root=project_root)
    set_seed(
        int(cfg.seed),
        deterministic=bool(cfg.device.deterministic),
        benchmark=bool(cfg.device.benchmark),
        use_deterministic_algorithms=bool(cfg.device.use_deterministic_algorithms),
    )
    logger.info(
        "Starting Flower simulation | clients=%d | logical_rounds=%d | flower_rounds=%d | strategy=%s",
        int(cfg.federated.num_clients),
        int(cfg.federated.num_rounds),
        get_effective_num_rounds(cfg),
        str(cfg.federated.strategy.name),
    )
    server = Server(client_manager=client_manager, strategy=get_strategy(cfg))
    history, _elapsed_time = server.fit(num_rounds=get_effective_num_rounds(cfg), timeout=None)
    run_dir = tracker.run_dir if tracker is not None else resolve_path(project_root, cfg.tracking.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_rows = _history_rows(history)
    if history_rows:
        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(run_dir / "federated_history.csv", index=False)
        if tracker is not None:
            tracker.write_json(
                "federated_history.json",
                {
                    "num_rows": len(history_rows),
                    "phases": sorted({row["phase"] for row in history_rows}),
                    "metrics": sorted({row["metric_name"] for row in history_rows}),
                },
            )
        communication_df = build_communication_metrics(
            run_dir=run_dir,
            project_root=project_root,
            history_frame=history_df,
        )
        if not communication_df.empty:
            communication_df.to_csv(run_dir / "communication_metrics.csv", index=False)
            if tracker is not None:
                tracker.write_json(
                    "communication_metrics.json",
                    {
                        "num_rows": len(communication_df),
                        "method": str(cfg.experiment.method),
                        "max_round": int(communication_df["round"].max()),
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
