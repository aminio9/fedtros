from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
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

from src.federated.client import FlowerClient
from src.federated.server import (
    get_effective_num_rounds,
    get_strategy,
    init_global_agent_ref,
    run_server,
)
from src.experiment.run_services import MetricsSink
from src.utils.config import resolve_path, sync_model_dimensions_from_preprocessing
from src.utils.utils import resolve_device_from_config, set_device, set_seed

logger = logging.getLogger(__name__)


def _restore_resume_rng_state(cfg: DictConfig, project_root: Path) -> bool:
    """Restore the round-boundary RNG state after all resume-time construction.

    Building the server reference and client model bundles consumes random draws.
    Restoring in ``resume.py`` before that construction therefore cannot reproduce
    an uninterrupted next round.  The global checkpoint is the authoritative
    process RNG snapshot and must be applied immediately before ``Server.fit``.
    """
    raw_path = OmegaConf.select(cfg, "federated.resume_from", default=None)
    if not raw_path:
        return False
    checkpoint_path = resolve_path(project_root, raw_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("rng_state") or {}
    if not state:
        return False
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    logger.info("Restored round-boundary RNG state from %s", checkpoint_path)
    return True


def _ndarrays_nbytes(values: list[Any]) -> int:
    """Exact in-memory model payload bytes for the NumPy arrays Flower transports.

    This intentionally excludes protocol/TLS/framework headers.  E7 reports these as
    *model-parameter payload bytes*, matching the experiment contract.
    """
    total = 0
    for value in values:
        try:
            total += int(value.nbytes)
        except AttributeError:
            arr = __import__("numpy").asarray(value)
            total += int(arr.nbytes)
    return total


def _actual_communication_frame(history_frame: pd.DataFrame) -> pd.DataFrame:
    """Build per-round communication totals from client-reported actual array bytes."""
    if history_frame.empty or not {"round", "metric_name", "metric_value"}.issubset(history_frame.columns):
        return pd.DataFrame()
    names = {
        "communication/downlink_bytes",
        "communication/uplink_bytes",
        "communication/round_bytes",
    }
    part = history_frame[history_frame["metric_name"].astype(str).isin(names)].copy()
    if part.empty:
        return pd.DataFrame()
    part["metric_value"] = pd.to_numeric(part["metric_value"], errors="coerce")
    pivot = part.pivot_table(index="round", columns="metric_name", values="metric_value", aggfunc="sum").reset_index()
    for name in names:
        if name not in pivot.columns:
            pivot[name] = 0.0
    # Aggregated Flower metrics already represent the sum for communication keys;
    # pivot(sum) is safe because there is one distributed-aggregation row per round.
    pivot["communication/cumulative_bytes"] = pivot["communication/round_bytes"].fillna(0).cumsum()
    return pivot.sort_values("round").reset_index(drop=True)


def _actual_runtime_frame(history_frame: pd.DataFrame) -> pd.DataFrame:
    """Build one per-round runtime record from canonical Flower metrics.

    ``runtime/round_seconds`` is measured server wall time.  Client/VCT/student
    fields are synchronous critical-path (max-client) values produced by the
    fit-metric aggregator.  This avoids reconstructing compute cost from logs.
    """
    if history_frame.empty or not {"round", "metric_name", "metric_value"}.issubset(history_frame.columns):
        return pd.DataFrame()
    names = {
        "runtime/client_fit_seconds",
        "runtime/teacher_seconds",
        "runtime/student_seconds",
        "runtime/aggregation_seconds",
        "runtime/open_set_eval_seconds",
        "runtime/round_seconds",
    }
    part = history_frame[history_frame["metric_name"].astype(str).isin(names)].copy()
    if part.empty:
        return pd.DataFrame()
    part["metric_value"] = pd.to_numeric(part["metric_value"], errors="coerce")
    pivot = part.pivot_table(
        index="round", columns="metric_name", values="metric_value", aggfunc="max"
    ).reset_index()
    for name in names:
        if name not in pivot.columns:
            pivot[name] = 0.0
    accounted = (
        pivot["runtime/client_fit_seconds"].fillna(0.0)
        + pivot["runtime/aggregation_seconds"].fillna(0.0)
        + pivot["runtime/open_set_eval_seconds"].fillna(0.0)
    )
    pivot["runtime/orchestration_seconds"] = (
        pivot["runtime/round_seconds"].fillna(0.0) - accounted
    ).clip(lower=0.0)
    pivot["runtime/cumulative_seconds"] = pivot["runtime/round_seconds"].fillna(0.0).cumsum()
    return pivot.sort_values("round").reset_index(drop=True)


def _resolve_runtime_config(cfg: DictConfig) -> DictConfig:
    """Freeze Hydra interpolations before config crosses process boundaries."""
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    local_batch_size = OmegaConf.select(resolved, "runtime.local_batch_size", default=None)
    if local_batch_size not in (None, "???"):
        resolved.training.batch_size = int(local_batch_size)
    return resolved


def _simulation_gpu_batches_enabled(cfg: DictConfig) -> bool:
    value = OmegaConf.select(cfg, "runtime.simulation_gpu_batches.enabled", default=False)
    return bool(value)


def _simulation_gpu_batch_size(cfg: DictConfig) -> int:
    value = OmegaConf.select(cfg, "runtime.simulation_gpu_batches.batch_size", default=1)
    return max(1, int(value or 1))


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
        fit_start = time.perf_counter()
        try:
            downlink_arrays = parameters_to_ndarrays(ins.parameters)
            downlink_bytes = _ndarrays_nbytes(downlink_arrays)
            parameters, num_examples, metrics = self._client.fit(
                downlink_arrays,
                dict(ins.config),
            )
            uplink_bytes = _ndarrays_nbytes(parameters)
            metrics = dict(metrics)
            client_fit_seconds = float(time.perf_counter() - fit_start)
            metrics["client_fit_wall_time_sec"] = client_fit_seconds
            metrics["runtime/client_fit_seconds"] = client_fit_seconds
            metrics["communication/downlink_bytes"] = float(downlink_bytes)
            metrics["communication/uplink_bytes"] = float(uplink_bytes)
            metrics["communication/round_bytes"] = float(downlink_bytes + uplink_bytes)
        except Exception:
            logger.exception(
                "Local Flower client %s fit failed | config=%s",
                self.cid,
                dict(ins.config),
            )
            raise
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
        eval_start = time.perf_counter()
        loss, num_examples, metrics = self._client.evaluate(
            parameters_to_ndarrays(ins.parameters),
            dict(ins.config),
        )
        metrics = dict(metrics)
        metrics["client_evaluate_wall_time_sec"] = float(time.perf_counter() - eval_start)
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
    simulation_gpu_batching: bool,
    simulation_execution_device: torch.device,
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
        residency = str(
            OmegaConf.select(cfg, "runtime.client_device_residency", default="swap")
        ).lower()
        if residency not in {"swap", "resident"}:
            raise ValueError(
                "runtime.client_device_residency must be 'swap' or 'resident', "
                f"got {residency!r}"
            )
        device = (
            simulation_execution_device
            if simulation_gpu_batching and residency == "resident"
            else torch.device("cpu")
        )
        data_path = _client_data_path(cfg, project_root, client_id)
        client = FlowerClient(
            cid=str(client_id),
            cfg=cfg,
            data_path=str(data_path),
            device=device,
            simulation_gpu_batching=simulation_gpu_batching,
            simulation_execution_device=simulation_execution_device,
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
    tracker: MetricsSink | None = None,
) -> dict[str, Any]:
    sync_model_dimensions_from_preprocessing(cfg, project_root=project_root)
    cfg = _resolve_runtime_config(cfg)
    gpu_batching_enabled = _simulation_gpu_batches_enabled(cfg)
    gpu_batch_size = _simulation_gpu_batch_size(cfg) if gpu_batching_enabled else 1
    allow_fallback = bool(OmegaConf.select(cfg, "runtime.allow_cpu_fallback", default=False)) or bool(OmegaConf.select(cfg, "device.allow_cpu_fallback", default=False))
    if gpu_batching_enabled and not torch.cuda.is_available():
        if allow_fallback:
            gpu_batching_enabled = False
        else:
            raise RuntimeError(
                "runtime.simulation_gpu_batches.enabled=true but CUDA is unavailable. "
                "Disable GPU batching only for explicit CPU validation runs."
            )
    simulation_execution_device = (
        torch.device("cuda")
        if gpu_batching_enabled and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Keep the server-side reference on CPU. GPU batching only applies to the
    # in-process client workers in this simulation path.
    set_device("cpu")
    init_global_agent_ref(cfg, torch.device("cpu"))
    client_manager = _build_local_clients(
        cfg,
        project_root=project_root,
        simulation_gpu_batching=gpu_batching_enabled,
        simulation_execution_device=simulation_execution_device,
    )
    set_seed(
        int(cfg.seed),
        deterministic=bool(cfg.device.deterministic),
        benchmark=bool(cfg.device.benchmark),
        use_deterministic_algorithms=bool(cfg.device.use_deterministic_algorithms),
    )
    logger.info(
        "Starting Flower simulation | clients=%d | logical_rounds=%d | flower_rounds=%d | strategy=%s | simulation_device=%s | gpu_batching=%s | worker_concurrency=%d | local_batch_size=%d",
        int(cfg.federated.num_clients),
        int(cfg.federated.num_rounds),
        get_effective_num_rounds(cfg),
        str(cfg.federated.strategy.name),
        simulation_execution_device,
        gpu_batching_enabled,
        gpu_batch_size,
        int(cfg.training.batch_size),
    )
    server = Server(client_manager=client_manager, strategy=get_strategy(cfg, metrics_sink=tracker))
    if gpu_batching_enabled:
        server.set_max_workers(gpu_batch_size)
        logger.info(
            "Local Flower simulation worker batching enabled | worker_concurrency=%d | execution_device=%s | local_batch_size=%d",
            gpu_batch_size,
            simulation_execution_device,
            int(cfg.training.batch_size),
        )
    elif bool(OmegaConf.select(cfg, "device.deterministic", default=False)):
        # Local clients are threads in Flower's legacy in-process server.  PyTorch's
        # RNG is process-global, so concurrent deterministic clients still race for
        # random draws (data shuffles, VCT sampling, and synthetic boundaries).
        # Serialize deterministic validation/reproduction runs; independent runs
        # can still be parallelized safely by run_study.py in separate processes.
        server.set_max_workers(1)
        logger.info(
            "Deterministic local simulation enabled | max_workers=1 to isolate client RNG streams"
        )
    _restore_resume_rng_state(cfg, project_root)
    history, _elapsed_time = server.fit(num_rounds=get_effective_num_rounds(cfg), timeout=None)
    run_dir = tracker.run_dir if tracker is not None else resolve_path(project_root, cfg.tracking.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_rows = _history_rows(history)
    if history_rows:
        history_df = pd.DataFrame(history_rows)
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(metrics_dir / "federated_history.csv", index=False)
        if tracker is not None:
            tracker.write_json(
                "metadata/federated_history.json",
                {
                    "num_rows": len(history_rows),
                    "phases": sorted({row["phase"] for row in history_rows}),
                    "metrics": sorted({row["metric_name"] for row in history_rows}),
                },
            )
        communication_df = _actual_communication_frame(history_df)
        if not communication_df.empty:
            communication_df.to_csv(metrics_dir / "communication_round.csv", index=False)
            if tracker is not None:
                tracker.write_json(
                    "metadata/communication_summary.json",
                    {
                        "num_rows": len(communication_df),
                        "semantics": "actual_numpy_model_parameter_payload_bytes_excluding_protocol_headers",
                        "method": str(cfg.experiment.method),
                        "max_round": int(communication_df["round"].max()),
                        "total_bytes": float(communication_df["communication/round_bytes"].sum()),
                    },
                )
        runtime_df = _actual_runtime_frame(history_df)
        if not runtime_df.empty:
            runtime_df.to_csv(metrics_dir / "timing_round.csv", index=False)
            if tracker is not None:
                tracker.write_json(
                    "metadata/runtime_summary.json",
                    {
                        "num_rows": len(runtime_df),
                        "semantics": "server_round_wall_time_plus_synchronous_client_critical_path",
                        "max_round": int(runtime_df["round"].max()),
                        "total_round_wall_seconds": float(runtime_df["runtime/round_seconds"].sum()),
                    },
                )
    timing_summary: dict[str, float] = {}
    try:
        if history_rows:
            timing_frame = pd.DataFrame(history_rows)
            if {"metric_name", "metric_value"}.issubset(timing_frame.columns):
                for source_name, output_name in (
                    ("round_time_sec", "federated/avg_round_time_sec"),
                    ("server_aggregation_time_sec", "federated/avg_server_aggregation_time_sec"),
                    ("client_fit_wall_time_sec", "federated/avg_client_fit_time_sec"),
                ):
                    values = pd.to_numeric(
                        timing_frame.loc[timing_frame["metric_name"] == source_name, "metric_value"],
                        errors="coerce",
                    ).dropna()
                    if not values.empty:
                        timing_summary[output_name] = float(values.mean())
                        timing_summary[output_name.replace("avg_", "total_")] = float(values.sum())
    except Exception:
        logger.exception("Failed to compute federated timing summary; continuing.")

    summary = {
        "federated/rounds": int(cfg.federated.num_rounds),
        "federated/flower_rounds": get_effective_num_rounds(cfg),
        "federated/num_clients": int(cfg.federated.num_clients),
        "federated/total_training_time_sec": float(_elapsed_time),
        **timing_summary,
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
    sync_model_dimensions_from_preprocessing(cfg, project_root=Path.cwd())
    resolved_device = device if device is not None else resolve_device_from_config(cfg)
    logger.info("Launching standalone Flower server | requested_device=%s", resolved_device)
    run_server(cfg, device=resolved_device)


def run_federated_client(
    cfg: DictConfig,
    *,
    project_root: Path,
    device: torch.device | None = None,
) -> None:
    sync_model_dimensions_from_preprocessing(cfg, project_root=project_root)
    client_id = int(cfg.federated.client_id)
    data_path = resolve_path(project_root, cfg.federated.client_data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Client tensor not found: {data_path}")
    resolved_device = device if device is not None else resolve_device_from_config(cfg)
    logger.info(
        "Launching standalone Flower client | client_id=%d | requested_device=%s | data_path=%s",
        client_id,
        resolved_device,
        data_path,
    )
    client = FlowerClient(
        cid=str(client_id),
        cfg=cfg,
        data_path=str(data_path),
        device=resolved_device,
    )
    fl.client.start_client(
        server_address=str(cfg.federated.server.address),
        client=client.to_client(),
    )
