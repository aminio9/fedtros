import csv
import json
import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import (
    EvaluateIns,
    FitIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg, FedProx, Strategy
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.models.bundle import FedTROSModelBundle as Agent
from src.checkpointing.checkpoints import load_agent_checkpoint
from src.models.models import FedTROSModelFactory
logger = logging.getLogger("Server")

EPS = 1e-8


# Global reference to hold the model architecture for saving
GLOBAL_AGENT_REF: Agent | None = None


def _emit_round_metrics(
    metrics_sink: Any | None,
    *,
    server_round: int,
    phase: str,
    metrics: dict[str, Any] | None,
) -> None:
    """Persist/track round metrics through the generic run-service interface.

    The server never imports W&B.  ``RunServices`` implements ``log_metrics`` and
    fans the same structured payload out to the canonical local ResultStore and the
    configured interactive tracker.  We intentionally do not pass an explicit W&B
    step here: fit, central validation, and client evaluation may all occur within
    one federated round.  ``federated/round`` is the scientific x-axis while the
    tracker keeps a monotonically increasing event step.
    """
    if metrics_sink is None or not hasattr(metrics_sink, "log_metrics"):
        return
    payload: dict[str, Any] = {
        "federated/round": float(server_round),
        "federated/phase": str(phase),
    }
    for key, value in dict(metrics or {}).items():
        if isinstance(value, (int, float, str, bool)):
            payload[str(key)] = value
    metrics_sink.log_metrics(payload)


def aggregate_fit_metrics(metrics: list[tuple[int, dict[str, Scalar]]]) -> dict[str, Scalar]:
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated: dict[str, float] = {}
    for num_examples, m in metrics:
        for k, v in m.items():
            if not isinstance(v, (int, float)):
                continue
            key = str(k)
            if key.startswith("communication/"):
                # Communication is a round total across participating clients, not a
                # sample-weighted learning metric.
                aggregated[key] = aggregated.get(key, 0.0) + float(v)
            elif key in {
                "runtime/client_fit_seconds",
                "runtime/teacher_seconds",
                "runtime/student_seconds",
                "client_fit_wall_time_sec",
            }:
                # For synchronous FL the slowest participating client determines the
                # critical-path local-compute time.  Full per-client values remain in
                # metrics/client_metrics.csv for distributional analysis.
                aggregated[key] = max(aggregated.get(key, 0.0), float(v))
            else:
                aggregated[key] = aggregated.get(key, 0.0) + float(v) * (num_examples / max(total_examples, 1))
    return aggregated


def aggregate_evaluate_metrics(metrics: list[tuple[int, dict[str, Scalar]]]) -> dict[str, Scalar]:
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated: dict[str, float] = {}
    for num_examples, m in metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)):
                aggregated[k] = aggregated.get(k, 0.0) + float(v) * (num_examples / max(total_examples, 1))
    return aggregated


def _project_root() -> Path:
    try:
        return Path(get_original_cwd())
    except ValueError:
        return Path(os.getcwd())


def _resolve_path(path_like) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (_project_root() / path)


def _load_tensor_dataset_for_server(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        if "features" in data and "labels" in data:
            return data["features"].float(), data["labels"].long()
        if "X" in data and "y" in data:
            return data["X"].float(), data["y"].long()
    if isinstance(data, (tuple, list)) and len(data) >= 2:
        return data[0].float(), data[1].long()
    raise ValueError(f"Unsupported tensor dataset format in {path}")


def _metric_is_better(new_value: float, old_value: float | None, mode: str) -> bool:
    if old_value is None:
        return True
    mode = str(mode or "max").lower()
    return new_value < old_value if mode == "min" else new_value > old_value


def _update_best_checkpoint(
    *,
    cfg: DictConfig,
    round_num: int,
    metric_name: str,
    metric_value: float,
    candidate_checkpoint: dict[str, Any] | None = None,
) -> None:
    """Promote the current checkpoint when the monitor metric improves."""
    best_path = _resolve_path(cfg.checkpointing.best_model_path)
    latest_path = _resolve_path(cfg.checkpointing.latest_checkpoint_path)
    metadata_path = best_path.with_suffix(best_path.suffix + ".metrics.json")
    mode = str(OmegaConf.select(cfg, "checkpointing.monitor_mode", default="max"))

    previous_value: float | None = None
    if metadata_path.exists():
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            previous_value = float(previous.get("metric_value"))
        except Exception:
            previous_value = None

    if not _metric_is_better(float(metric_value), previous_value, mode):
        return

    best_path.parent.mkdir(parents=True, exist_ok=True)
    if candidate_checkpoint is not None:
        torch.save(candidate_checkpoint, best_path)
    elif latest_path.exists():
        shutil.copy2(latest_path, best_path)
    else:
        logger.warning("Cannot update best checkpoint; latest checkpoint missing: %s", latest_path)
        return

    metadata = {
        "round": int(round_num),
        "metric_name": str(metric_name),
        "metric_value": float(metric_value),
        "monitor_mode": mode,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(
        "Updated best checkpoint | round=%s | %s=%.6f | path=%s",
        round_num,
        metric_name,
        float(metric_value),
        best_path,
    )


def make_central_evaluate_fn(
    cfg: DictConfig,
    device: torch.device,
    metrics_sink: Any | None = None,
):
    """Build a server-side validation evaluator for clean FedAvg/FedProx.

    Client-side shared-test evaluation remains enabled; this central evaluator is
    only for clean validation logging and best-checkpoint selection.
    """
    if not bool(OmegaConf.select(cfg, "federated.central_evaluate.enabled", default=False)):
        return None

    eval_path_value = OmegaConf.select(
        cfg,
        "federated.central_evaluate.data_path",
        default=OmegaConf.select(cfg, "paths.validation_data", default=None),
    )
    if not eval_path_value:
        logger.warning("Central evaluation requested but no data path was configured.")
        return None

    eval_path = _resolve_path(eval_path_value)
    if not eval_path.exists():
        logger.warning("Central evaluation data missing: %s", eval_path)
        return None

    features, labels = _load_tensor_dataset_for_server(eval_path)
    batch_size = int(OmegaConf.select(cfg, "evaluation.batch_size", default=512))
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False)
    eval_agent = Agent(FedTROSModelFactory(cfg.model), cfg.training, device=device)
    prefix = str(OmegaConf.select(cfg, "federated.central_evaluate.prefix", default="val"))
    monitor_metric = str(OmegaConf.select(cfg, "checkpointing.monitor_metric", default="val/macro_f1"))

    logger.info(
        "Central evaluation enabled | data=%s | samples=%d | prefix=%s | monitor=%s",
        eval_path,
        int(labels.numel()),
        prefix,
        monitor_metric,
    )

    def evaluate(server_round: int, parameters, config: dict[str, Scalar]):
        _ = config
        eval_agent.set_federated_parameters(list(parameters), hard_target_update=True)
        eval_agent.student_model.eval()
        total_loss = 0.0
        total = 0
        y_true: list[int] = []
        y_pred: list[int] = []

        with torch.no_grad():
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(device).float()
                batch_labels = batch_labels.to(device).long()
                _, logits = eval_agent.student_model(batch_features)
                loss = F.cross_entropy(logits, batch_labels, reduction="sum")
                preds = logits.argmax(dim=1)
                total_loss += float(loss.item())
                total += int(batch_labels.numel())
                y_true.extend(batch_labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        avg_loss = total_loss / max(total, 1)
        accuracy = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
        balanced = float(balanced_accuracy_score(y_true, y_pred)) if y_true else 0.0
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0
        metrics: dict[str, Scalar] = {
            f"{prefix}/loss": float(avg_loss),
            f"{prefix}/accuracy": accuracy,
            f"{prefix}/balanced_accuracy": balanced,
            f"{prefix}/macro_f1": macro_f1,
            f"{prefix}/num_examples": int(total),
            "central_evaluate": 1.0,
        }
        logger.info(
            "Central %s evaluation | round=%s | loss=%.6f | acc=%.4f | bal_acc=%.4f | macro_f1=%.4f",
            prefix,
            server_round,
            avg_loss,
            accuracy,
            balanced,
            macro_f1,
        )
        if bool(cfg.checkpointing.save_best) and monitor_metric in metrics:
            _update_best_checkpoint(
                cfg=cfg,
                round_num=int(server_round),
                metric_name=monitor_metric,
                metric_value=float(metrics[monitor_metric]),
            )
        _emit_round_metrics(
            metrics_sink,
            server_round=int(server_round),
            phase="central_validation",
            metrics=metrics,
        )
        return float(avg_loss), metrics

    return evaluate


def init_global_agent_ref(cfg: DictConfig, device: torch.device):
    """
    Initialize a dummy Agent on the server.
    This is used solely to map the flat list of parameters from Federated Learning
    back into a state_dict for saving .pt files.
    """
    global GLOBAL_AGENT_REF
    if Agent is None or FedTROSModelFactory is None:
        logger.warning(
            "Could not import Agent/Models. Checkpoints will be saved as raw NumPy arrays only."
        )
        return

    try:
        model_factory = FedTROSModelFactory(cfg.model)
        GLOBAL_AGENT_REF = Agent(model_factory, cfg.training, device)
        logger.info(
            "Server model reference initialized | device=%s | strategy=%s | clients=%s",
            device,
            str(cfg.strategy.name),
            int(cfg.federated.num_clients),
        )
    except Exception as e:
        logger.error(f"Failed to initialize Global Agent reference: {e}")


def save_global_model(
    parameters: Parameters | list[np.ndarray],
    round_num: int,
    cfg: DictConfig,
    metrics: dict[str, Scalar] | None = None,
) -> None:
    """
    Universal function to save the global model state with Schema Version 2.
    """
    model_dir = _resolve_path(cfg.checkpointing.dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Convert Parameters object to List[np.ndarray] if necessary
    if isinstance(parameters, Parameters):
        weights_list = parameters_to_ndarrays(parameters)
    else:
        weights_list = parameters

    if GLOBAL_AGENT_REF is None:
        raise RuntimeError("Global agent reference is not initialized; cannot save checkpoint.")

    try:
        GLOBAL_AGENT_REF.set_federated_parameters(weights_list, hard_target_update=True)

        checkpoint = {
            "schema_version": 2,
            "round": int(round_num),
            "epoch": int(round_num),
            "global_step": int(round_num),
            "method": "FedTROS-PR",
            "method_id": "fedtros_pr",
            "teacher_type": "variational_classifier",
            "teacher_latent_dim": int(
                OmegaConf.select(
                    cfg,
                    "training.teacher_latent_dim",
                    default=OmegaConf.select(cfg, "model.latent_dim", default=64),
                )
            ),
            "config_hash": str(OmegaConf.select(cfg, "experiment.config_hash", default="")),
            "code_commit": str(OmegaConf.select(cfg, "experiment.git_commit", default="unknown_commit")),
            "git_dirty": bool(OmegaConf.select(cfg, "experiment.git_dirty", default=False)),
            "run_id": str(OmegaConf.select(cfg, "experiment.run_id", default=OmegaConf.select(cfg, "tracking.run_id", default=""))),
            "study_id": str(OmegaConf.select(cfg, "experiment.id", default="")),
            "stage": str(OmegaConf.select(cfg, "stage", default="development")),
            "metrics": {"federated/round": float(round_num), **dict(metrics or {})},
            "config": OmegaConf.to_container(cfg, resolve=True),
            "student_model": GLOBAL_AGENT_REF.student_model.state_dict(),
            "optimizer_student": GLOBAL_AGENT_REF.optimizer_student.state_dict(),
        }
        if bool(OmegaConf.select(cfg, "checkpointing.include_rng_state", default=True)):
            checkpoint["rng_state"] = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            }

        round_path = model_dir / f"global_model_round_{round_num:04d}.pt"
        torch.save(checkpoint, round_path)
        torch.save(checkpoint, model_dir / "global_model_latest.pt")
        torch.save(checkpoint, _resolve_path(cfg.checkpointing.latest_checkpoint_path))
        if bool(cfg.checkpointing.save_best):
            best_path = _resolve_path(cfg.checkpointing.best_model_path)
            monitor_metric = str(OmegaConf.select(cfg, "checkpointing.monitor_metric", default=""))
            has_monitor_metric = bool(monitor_metric and metrics and monitor_metric in metrics)
            if has_monitor_metric:
                _update_best_checkpoint(
                    candidate_checkpoint=checkpoint,
                    cfg=cfg,
                    round_num=round_num,
                    metric_name=monitor_metric,
                    metric_value=float(metrics[monitor_metric]),
                )
            elif not bool(OmegaConf.select(cfg, "federated.central_evaluate.enabled", default=False)):
                torch.save(checkpoint, best_path)

        metrics_path = model_dir / "federated_round_metrics.csv"
        write_header = not metrics_path.exists()
        with open(metrics_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["federated/round", "checkpoint_path"])
            if write_header:
                writer.writeheader()
            writer.writerow({"federated/round": round_num, "checkpoint_path": str(round_path)})
        logger.info("Saved global model checkpoint (schema_version 2) to %s", round_path)

    except Exception as exc:
        logger.exception("Failed to save canonical checkpoint: %s", exc)
        raise


def fit_config_fn(server_round: int) -> dict[str, fl.common.Scalar]:
    return {"server_round": server_round, "phase": "standard"}


def _initial_parameters_from_checkpoint(
    cfg: DictConfig, device: torch.device
) -> Parameters | None:
    """Return initial Flower parameters for resumed federated training."""
    resume_from = OmegaConf.select(cfg, "federated.resume_from", default=None)
    if not resume_from:
        return None
    if GLOBAL_AGENT_REF is None:
        init_global_agent_ref(cfg, device)
    if GLOBAL_AGENT_REF is None:
        raise RuntimeError("Global agent reference is not initialized; cannot resume FL.")

    checkpoint_path = _resolve_path(resume_from)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    strict = bool(cfg.checkpointing.strict_load)

    if isinstance(checkpoint, dict) and "student_model" in checkpoint:
        GLOBAL_AGENT_REF.student_model.load_state_dict(checkpoint["student_model"], strict=strict)
        if hasattr(GLOBAL_AGENT_REF, "student_anchor_model") and GLOBAL_AGENT_REF.student_anchor_model is not None:
            GLOBAL_AGENT_REF.student_anchor_model.load_state_dict(
                GLOBAL_AGENT_REF.student_model.state_dict(), strict=False
            )
        logger.info(
            "Loaded student resume checkpoint | path=%s | saved_round=%s",
            checkpoint_path,
            checkpoint.get("round", checkpoint.get("epoch", "unknown")),
        )
        return ndarrays_to_parameters(GLOBAL_AGENT_REF.get_student_parameters())

    load_agent_checkpoint(
        GLOBAL_AGENT_REF,
        checkpoint_path,
        device,
        strict=strict,
        load_optimizers=False,
    )
    logger.info("Loaded federated initial parameters from %s", checkpoint_path)
    return ndarrays_to_parameters(GLOBAL_AGENT_REF.get_federated_parameters())


# --- CUSTOM STRATEGIES WITH SAVING LOGIC ---


class SaveModelFedAvg(FedAvg):
    def __init__(self, cfg: DictConfig, *args, metrics_sink: Any | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.metrics_sink = metrics_sink

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:

        # Call original aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        total_bytes = sum(sum(len(t) for t in fit_res.parameters.tensors) for _, fit_res in results if fit_res.parameters.tensors)
        aggregated_metrics["communication/real_payload_bytes_rx"] = float(total_bytes)

        if aggregated_parameters is not None:
            # Save the model
            save_global_model(
                aggregated_parameters, server_round, self.cfg, metrics=dict(aggregated_metrics)
            )

        _emit_round_metrics(
            self.metrics_sink,
            server_round=int(server_round),
            phase="fit",
            metrics=dict(aggregated_metrics),
        )

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        _emit_round_metrics(
            self.metrics_sink,
            server_round=int(server_round),
            phase="client_evaluate",
            metrics=dict(metrics or {}),
        )
        return loss, metrics


class SaveModelFedProx(FedProx):
    def __init__(self, cfg: DictConfig, *args, metrics_sink: Any | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.metrics_sink = metrics_sink

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        total_bytes = sum(sum(len(t) for t in fit_res.parameters.tensors) for _, fit_res in results if fit_res.parameters.tensors)
        aggregated_metrics["communication/real_payload_bytes_rx"] = float(total_bytes)

        if aggregated_parameters is not None:
            save_global_model(
                aggregated_parameters, server_round, self.cfg, metrics=dict(aggregated_metrics)
            )

        _emit_round_metrics(
            self.metrics_sink,
            server_round=int(server_round),
            phase="fit",
            metrics=dict(aggregated_metrics),
        )

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        _emit_round_metrics(
            self.metrics_sink,
            server_round=int(server_round),
            phase="client_evaluate",
            metrics=dict(metrics or {}),
        )
        return loss, metrics


class FedTROSStrategy(FedAvg):
    """FedTROS strategy for federated teacher-regularized open-set learning.

    Only the compact student model is sent to and aggregated by the server.
    The private Variational Classifier Teacher (VCT) remains local on each client,
    preventing extreme non-IID clients from overwriting a shared global teacher.
    """

    def __init__(self, cfg: DictConfig, *args, metrics_sink: Any | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if GLOBAL_AGENT_REF is None:
            raise RuntimeError("FedTROS needs a server Agent reference with a student model.")
        self.cfg = cfg
        self.metrics_sink = metrics_sink
        self.monitor_path = _resolve_path(
            OmegaConf.select(cfg, "strategy.monitor_path", default="outputs/fedtros_pr_monitoring.jsonl")
        )
        self.monitor_path.parent.mkdir(parents=True, exist_ok=True)
        self.student_aggregation_mode = str(
            OmegaConf.select(cfg, "strategy.student_aggregation_mode", default="support_weighted_average")
        ).lower()
        self.support_min_weight = float(
            OmegaConf.select(cfg, "strategy.support_min_weight", default=0.01)
        )
        self.global_student_parameters = ndarrays_to_parameters(GLOBAL_AGENT_REF.get_student_parameters())
        self.resume_round_offset = int(
            OmegaConf.select(cfg, "federated.resume_round_offset", default=0) or 0
        )
        self._round_wall_start: dict[int, float] = {}

        expected_hidden = [512, 256, 128]
        actual_hidden = list(OmegaConf.select(cfg, "training.fedtros_student_hidden_dims", default=[]))
        anchor_weight = float(OmegaConf.select(cfg, "training.fedtros_global_anchor_weight", default=0.0) or 0.0)
        if self.student_aggregation_mode != "support_weighted_average":
            raise ValueError(
                "FedTROS contract violation: "
                f"student_aggregation_mode must be support_weighted_average, got {self.student_aggregation_mode!r}."
            )
        if anchor_weight < 0.0:
            raise ValueError(
                "FedTROS contract violation: training.fedtros_global_anchor_weight must be >= 0. "
                "A zero value is reserved for the predeclared A2 no-anchor ablation."
            )
        if [int(v) for v in actual_hidden] != expected_hidden:
            raise ValueError(
                "FedTROS contract violation: "
                f"student_hidden_dims must be {expected_hidden}, got {actual_hidden}."
            )

        self.round_open_set_eval_enabled = bool(
            OmegaConf.select(cfg, "open_set.evaluate_each_round", default=False)
        )
        self.round_open_set_eval_every_n = max(
            1, int(OmegaConf.select(cfg, "open_set.evaluate_every_n_rounds", default=1) or 1)
        )
        self.round_open_set_eval_dir = str(
            OmegaConf.select(cfg, "open_set.round_eval_dir", default="open_set_rounds")
            or "open_set_rounds"
        )
        self.round_open_set_save_scores = bool(
            OmegaConf.select(cfg, "open_set.save_round_scores", default=False)
        )

        canonical = bool(OmegaConf.select(cfg, "method.canonical", default=False))
        if canonical:
            self.round_open_set_eval_enabled = False
            logger.info(
                "--- Strategy: FedTROS-MC ---\n"
                "canonical=true\n"
                "anchor_statistic=kappa_i\n"
                "anchor_enabled=true\n"
                "aggregation=power_support_average\n"
                "gamma=0.500\n"
                f"student_hidden_dims={actual_hidden}\n"
                f"student_osr_decoder={bool(getattr(GLOBAL_AGENT_REF.student_model, 'osr_enabled', False))}\n"
                "rejection_backend=multicenter_conformal\n"
                "round_final_unknown_eval=false\n"
                f"resume_round_offset={self.resume_round_offset}"
            )
        else:
            logger.info(
                "FedTROS-PR (legacy) READY | "
                "student_aggregation_mode=%s | global_anchor_enabled=%s | "
                "global_anchor_weight=%.3f | student_hidden_dims=%s | student_layers=%d | "
                "support_min_weight=%.3f | round_open_set_eval=%s every=%d | "
                "student_osr_enabled=%s | resume_round_offset=%d",
                self.student_aggregation_mode,
                bool(anchor_weight > 0.0),
                anchor_weight,
                actual_hidden,
                len(GLOBAL_AGENT_REF.get_student_parameters()),
                self.support_min_weight,
                self.round_open_set_eval_enabled,
                self.round_open_set_eval_every_n,
                bool(getattr(GLOBAL_AGENT_REF.student_model, "osr_enabled", False)),
                self.resume_round_offset,
            )

    def _effective_round(self, server_round: int) -> int:
        return int(self.resume_round_offset) + int(server_round)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        _ = parameters
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_fit_clients,
        )
        effective_round = self._effective_round(server_round)
        self._round_wall_start[effective_round] = time.perf_counter()
        fit_ins = FitIns(
            self.global_student_parameters,
            {
                "server_round": effective_round,
                "logical_round": server_round,
                "phase": "fedtros_pr",
            },
        )
        logger.info(
            "FedTROS round=%d effective_round=%d sampled=%d",
            server_round,
            effective_round,
            len(clients),
        )
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        _ = parameters
        if self.fraction_evaluate == 0.0:
            return []
        clients = client_manager.sample(
            num_clients=self.min_evaluate_clients,
            min_num_clients=self.min_evaluate_clients,
        )
        evaluate_ins = EvaluateIns(
            self.global_student_parameters,
            {"server_round": server_round, "phase": "fedtros_pr"},
        )
        logger.info("FedTROS-PR evaluation round=%d clients=%d", server_round, len(clients))
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate every communicated student tensor using the canonical support rule.

        FedTROS-PR deliberately uses one transparent aggregation rule for classifier and
        optional OSR-branch tensors.  Quality-weighted OSR aggregation is not part of the
        canonical method and belongs only in an explicit ablation implementation.
        """
        aggregation_start = time.perf_counter()
        if failures:
            self._log_failures("FedTROS-PR", failures)

        records: list[dict[str, Any]] = []
        for client, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            if not weights:
                logger.warning("FedTROS client %s returned empty student weights; skipping", client.cid)
                continue
            records.append({
                "cid": getattr(client, "cid", "?"),
                "weights": weights,
                "num_examples": float(fit_res.num_examples),
                "metrics": dict(fit_res.metrics),
            })
        if not records:
            logger.warning("FedTROS round=%d has no usable client updates", server_round)
            return self.global_student_parameters, {}

        max_examples = max(float(r.get("num_examples", 0.0)) for r in records)
        support_weights = [self._client_support_weight(r, max_examples=max_examples) for r in records]
        for record, weight in zip(records, support_weights, strict=True):
            record["support_weight"] = float(weight)
        
        csv_path = _resolve_path(self.cfg.tracking.run_dir) / "client_support.csv"
        file_exists = csv_path.exists()
        with open(csv_path, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("round,client_id,n_i,q_i,kappa_i,aggregation_weight,anchor_lambda\n")
            for record in records:
                m = record["metrics"]
                f.write(f"{server_round},{record['cid']},{record['num_examples']},{m.get('q_i','')},{m.get('kappa_i','')},{record['support_weight']},{m.get('student_anchor_weight','')}\n")
        
        weight_sum = max(float(np.sum(support_weights)), EPS)


        base = parameters_to_ndarrays(self.global_student_parameters)
        avg_weights: list[np.ndarray] = []
        for layer_idx in range(len(base)):
            layer_sum = np.zeros_like(base[layer_idx])
            for record, support_weight in zip(records, support_weights, strict=True):
                layer_sum += float(support_weight) * record["weights"][layer_idx]
            avg_weights.append(layer_sum / weight_sum)

        distance_to_previous = self._weight_list_norm([
            before - after for before, after in zip(base, avg_weights, strict=True)
        ])
        self.global_student_parameters = ndarrays_to_parameters(avg_weights)

        fit_metrics = [(int(r["num_examples"]), r["metrics"]) for r in records]
        metrics = aggregate_fit_metrics(fit_metrics)

        total_bytes = sum(sum(len(t) for t in fit_res.parameters.tensors) for _, fit_res in results if fit_res.parameters.tensors)
        metrics["communication/real_payload_bytes_rx"] = float(total_bytes)

        macro_f1_values = [
            value for record in records for value in [
                self._first_numeric_metric(dict(record.get("metrics", {})),
                    ("local_student_f1_macro", "student_before_local_f1_macro", "f1_macro", "audit_f1"))
            ] if value is not None
        ]
        accuracy_values = [
            value for record in records for value in [
                self._first_numeric_metric(dict(record.get("metrics", {})),
                    ("local_student_accuracy", "student_before_local_accuracy", "accuracy"))
            ] if value is not None
        ]
        dispersion = self._distribution_metrics(macro_f1_values, prefix="client_macro_f1")
        if dispersion:
            dispersion.update({
                "mean_client_macro_f1": dispersion["client_macro_f1_mean"],
                "std_client_macro_f1": dispersion["client_macro_f1_std"],
                "worst_client_macro_f1": dispersion["client_macro_f1_worst"],
                "bottom10_client_macro_f1": dispersion["client_macro_f1_bottom10_mean"],
            })
        metrics.update(dispersion)
        metrics.update(self._distribution_metrics(accuracy_values, prefix="client_accuracy"))
        effective_round = self._effective_round(server_round)
        self._append_fit_client_rows(round_num=effective_round, records=records)

        metrics.update({
            "federated/participating_clients": float(len(records)),
            "federated/support_weight_mean": float(np.mean(support_weights)),
            "federated/support_weight_min": float(np.min(support_weights)),
            "federated/support_weight_max": float(np.max(support_weights)),
            "federated/global_update_norm": float(distance_to_previous),
            "federated/aggregation_support_weighted": 1.0,
        })

        self._save_student_checkpoint(avg_weights, effective_round, metrics)
        metrics["runtime/aggregation_seconds"] = float(time.perf_counter() - aggregation_start)
        metrics["server_aggregation_time_sec"] = metrics["runtime/aggregation_seconds"]

        round_eval_start = time.perf_counter()
        round_open_set_metrics = self._run_round_open_set_evaluation(effective_round)
        if round_open_set_metrics:
            metrics.update({f"round_{k}": v for k, v in round_open_set_metrics.items() if isinstance(v, (int, float))})
        metrics["runtime/open_set_eval_seconds"] = (
            float(time.perf_counter() - round_eval_start) if round_open_set_metrics else 0.0
        )
        metrics["open_set_round_eval_time_sec"] = metrics["runtime/open_set_eval_seconds"]
        round_start = self._round_wall_start.pop(effective_round, aggregation_start)
        metrics["runtime/round_seconds"] = float(time.perf_counter() - round_start)
        metrics["round_time_sec"] = metrics["runtime/round_seconds"]
        self._append_scalability_round_row(round_num=effective_round, metrics=metrics)

        self._write_monitor_event({
            "event": "fedtros_pr_support_aggregation",
            "server_round": effective_round,
            "logical_round": server_round,
            "clients": [r["cid"] for r in records],
            "support_weights": {str(r["cid"]): float(r["support_weight"]) for r in records},
            "global_update_norm": distance_to_previous,
            "metrics": metrics,
        })
        logger.info(
            "FedTROS-PR support aggregation | round=%d clients=%d update_norm=%.4f",
            effective_round, len(records), distance_to_previous,
        )
        _emit_round_metrics(
            self.metrics_sink,
            server_round=effective_round,
            phase="fit",
            metrics=metrics,
        )
        return self.global_student_parameters, metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if failures:
            self._log_failures("FedTROS-PR evaluation", failures)
        client_rows: list[dict[str, Any]] = []
        macro_values: list[float] = []
        accuracy_values: list[float] = []
        effective_round = self._effective_round(server_round)
        for client, evaluate_res in results:
            result_metrics = dict(getattr(evaluate_res, "metrics", {}) or {})
            macro_f1 = self._first_numeric_metric(result_metrics, ("student_f1_macro", "f1_macro", "macro_f1"))
            accuracy = self._first_numeric_metric(result_metrics, ("student_accuracy", "accuracy"))
            if macro_f1 is not None:
                macro_values.append(macro_f1)
            if accuracy is not None:
                accuracy_values.append(accuracy)
            if macro_f1 is not None or accuracy is not None:
                client_rows.append(
                    {
                        "round": int(effective_round),
                        "client_id": str(getattr(client, "cid", "?")),
                        "num_examples": int(getattr(evaluate_res, "num_examples", 0)),
                        "loss": float(getattr(evaluate_res, "loss", 0.0)),
                        "macro_f1": macro_f1,
                        "accuracy": accuracy,
                        "client_evaluate_wall_time_sec": self._first_numeric_metric(
                            result_metrics, ("client_evaluate_wall_time_sec",)
                        ),
                    }
                )
        for row in client_rows:
            self._append_csv_row(
                _resolve_path(Path(str(self.cfg.tracking.run_dir)) / "metrics" / "client_eval_metrics.csv"),
                row,
            )
        eval_distribution = self._distribution_metrics(macro_values, prefix="eval_client_macro_f1")
        if eval_distribution:
            metrics.update(eval_distribution)
        metrics.update(self._distribution_metrics(accuracy_values, prefix="eval_client_accuracy"))
        _emit_round_metrics(
            self.metrics_sink,
            server_round=effective_round,
            phase="client_evaluate",
            metrics=dict(metrics or {}),
        )
        return loss, metrics

    def _run_round_open_set_evaluation(self, server_round: int) -> dict[str, float]:
        """Run one server-side global-student open-set evaluation after aggregation.

        This hook is server-side and uses only the aggregated global student
        checkpoint saved for the current round.
        """
        if not self.round_open_set_eval_enabled:
            return {}
        if int(server_round) % int(self.round_open_set_eval_every_n) != 0:
            return {}
        if not bool(OmegaConf.select(self.cfg, "open_set.enabled", default=False)):
            return {}

        backend = str(
            OmegaConf.select(self.cfg, "open_set.detector", default="prototype_rank")
        ).lower()
        canonical = bool(OmegaConf.select(self.cfg, "method.canonical", default=False))
        if canonical or backend not in {"prototype_rank", "fedtros_pr"}:
            logger.info(
                "FedTROS round=%d open-set evaluation skipped: canonical mode or non-legacy backend.",
                server_round,
            )
            return {}

        eval_base = _resolve_path(OmegaConf.select(self.cfg, "evaluation.output_dir", default="outputs/evaluation"))
        output_dir = eval_base / self.round_open_set_eval_dir / f"round_{int(server_round):04d}"
        try:
            from src.evaluation.run import run_prototype_rank_evaluation

            logger.info(
                "[Round %d] Running server-side global open-set evaluation | backend=%s | output_dir=%s",
                server_round,
                backend,
                output_dir,
            )
            metrics = run_prototype_rank_evaluation(
                self.cfg,
                project_root=_project_root(),
                device=torch.device("cpu"),
                tracker=None,
                output_dir=output_dir,
                server_round=int(server_round),
                save_scores=bool(self.round_open_set_save_scores),
                append_round_metrics=True,
            )
            logger.info(
                "[Round %d] Global open-set eval complete | AUROC=%.4f | Unknown_Recall=%.4f | Known_FU=%.4f | MacroF1=%.4f",
                server_round,
                float(metrics.get("openset_auroc", 0.0)),
                float(metrics.get("openset_unknown_recall", 0.0)),
                float(metrics.get("openset_known_false_unknown_rate", 0.0)),
                float(metrics.get("openset_f1_macro", 0.0)),
            )
            return {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        except Exception as exc:
            logger.exception(
                "[Round %d] Global open-set evaluation failed; training will continue. Error: %s",
                server_round,
                exc,
            )
            return {"open_set_round_eval_failed": 1.0}

    @staticmethod
    def _first_numeric_metric(metrics: dict[str, Any], candidates: tuple[str, ...]) -> float | None:
        for candidate in candidates:
            value = metrics.get(candidate)
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                return float(value)
        return None

    @staticmethod
    def _distribution_metrics(values: list[float], *, prefix: str) -> dict[str, float]:
        finite = [float(v) for v in values if np.isfinite(float(v))]
        if not finite:
            return {}
        arr = np.asarray(finite, dtype=np.float64)
        bottom_k = max(1, int(np.ceil(0.10 * len(arr))))
        return {
            f"{prefix}_mean": float(np.mean(arr)),
            f"{prefix}_std": float(np.std(arr, ddof=0)),
            f"{prefix}_worst": float(np.min(arr)),
            f"{prefix}_best": float(np.max(arr)),
            f"{prefix}_bottom10_mean": float(np.mean(np.sort(arr)[:bottom_k])),
        }

    def _append_csv_row(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()), extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _append_fit_client_rows(self, *, round_num: int, records: list[dict[str, Any]]) -> None:
        client_metrics_path = _resolve_path(
            Path(str(self.cfg.tracking.run_dir)) / "metrics" / "client_metrics.csv"
        )
        for record in records:
            metrics = dict(record.get("metrics", {}))
            macro_f1 = self._first_numeric_metric(
                metrics,
                ("local_student_f1_macro", "student_before_local_f1_macro", "f1_macro", "audit_f1"),
            )
            accuracy = self._first_numeric_metric(
                metrics,
                ("local_student_accuracy", "student_before_local_accuracy", "accuracy"),
            )
            if macro_f1 is None and accuracy is None:
                continue
            row_data = {
                "round": int(round_num),
                "client_id": str(record.get("cid", "?")),
                "num_examples": float(record.get("num_examples", 0.0)),
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "support_weight": float(record.get("support_weight", 1.0)),
                "client_fit_wall_time_sec": self._first_numeric_metric(
                    metrics, ("client_fit_wall_time_sec",)
                ),
            }
            self._append_csv_row(client_metrics_path, row_data)

    def _append_scalability_round_row(self, *, round_num: int, metrics: dict[str, Any]) -> None:
        path = _resolve_path(Path(str(self.cfg.tracking.run_dir)) / "metrics" / "scalability_round_metrics.csv")
        keys = [
            "local_student_f1_macro",
            "local_student_accuracy",
            "mean_client_macro_f1",
            "std_client_macro_f1",
            "worst_client_macro_f1",
            "client_fit_wall_time_sec",
            "round_time_sec",
            "server_aggregation_time_sec",
            "open_set_round_eval_time_sec",
            "round_openset_f1_macro",
            "round_openset_overall_acc",
            "round_openset_known_acc",
            "round_openset_auroc",
            "round_openset_fpr95",
            "round_openset_unknown_recall",
        ]
        row: dict[str, Any] = {
            "round": int(round_num),
            "num_clients": int(OmegaConf.select(self.cfg, "federated.num_clients", default=0) or 0),
            "seed": int(OmegaConf.select(self.cfg, "seed", default=0) or 0),
            "alpha": float(OmegaConf.select(self.cfg, "dataset.preprocessing.alpha", default=0.0) or 0.0),
        }
        for key in keys:
            value = metrics.get(key)
            row[key] = float(value) if isinstance(value, (int, float)) else None
        self._append_csv_row(path, row)

    def _client_support_weight(self, record: dict[str, Any], *, max_examples: float) -> float:
        """Canonical FedTROS-PR support weight: n_i^gamma."""
        num_examples = max(float(record.get("num_examples", 0.0)), 0.0)
        canonical = bool(OmegaConf.select(self.cfg, "method.canonical", default=False))
        if canonical:
            gamma = float(OmegaConf.select(self.cfg, "strategy.gamma", default=0.5))
            return float(num_examples ** gamma)
        else:
            sample_factor = float(np.sqrt(num_examples / max(float(max_examples), 1.0)))
            min_weight = float(OmegaConf.select(self.cfg, "strategy.support_min_weight", default=0.01))
            return float(np.clip(sample_factor, min_weight, 1.0))

    def _save_student_checkpoint(
        self,
        weights: list[np.ndarray],
        round_num: int,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        model_dir = _resolve_path(self.cfg.checkpointing.dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        if GLOBAL_AGENT_REF is None:
            return
        GLOBAL_AGENT_REF.set_student_parameters(weights)
        canonical = bool(OmegaConf.select(self.cfg, "method.canonical", default=False))
        checkpoint = {
            "round": round_num,
            "epoch": round_num,
            "global_step": round_num,
            "schema_version": 2,
            "method": "FedTROS-MC" if canonical else "FedTROS-PR (legacy)",
            "method_id": "fedtros_mc" if canonical else "fedtros_pr_legacy",
            "teacher_type": "variational_classifier",
            "config_hash": str(OmegaConf.select(self.cfg, "experiment.config_hash", default="")),
            "code_commit": str(OmegaConf.select(self.cfg, "experiment.git_commit", default="unknown_commit")),
            "git_dirty": bool(OmegaConf.select(self.cfg, "experiment.git_dirty", default=False)),
            "run_id": str(OmegaConf.select(self.cfg, "experiment.run_id", default=OmegaConf.select(self.cfg, "tracking.run_id", default=""))),
            "study_id": str(OmegaConf.select(self.cfg, "experiment.id", default="")),
            "stage": str(OmegaConf.select(self.cfg, "stage", default="development")),
            "metrics": {"federated/round": float(round_num), **dict(metrics or {})},
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "student_model": GLOBAL_AGENT_REF.student_model.state_dict(),
            "optimizer_student": GLOBAL_AGENT_REF.optimizer_student.state_dict(),
        }
        if bool(OmegaConf.select(self.cfg, "checkpointing.include_rng_state", default=True)):
            checkpoint["rng_state"] = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            }
        prefix = "fedtros_mc" if canonical else "fedtros_pr_legacy"
        round_path = model_dir / f"{prefix}_student_round_{round_num:04d}.pt"
        latest_path = model_dir / f"{prefix}_student_latest.pt"
        torch.save(checkpoint, round_path)
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, _resolve_path(self.cfg.checkpointing.latest_checkpoint_path))

        logger.info("Saved FedTROS-PR student checkpoint to %s", round_path)

    @staticmethod
    def _weight_list_norm(weights: list[np.ndarray]) -> float:
        total = 0.0
        for layer in weights:
            layer64 = layer.astype(np.float64, copy=False)
            total += float(np.sum(layer64 * layer64))
        return float(np.sqrt(max(total, 0.0)))

    def _write_monitor_event(self, event: dict[str, Any]) -> None:
        try:
            with open(self.monitor_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write FedTROS monitor event: %s", exc)

    @staticmethod
    def _log_failures(context: str, failures) -> None:
        for idx, failure in enumerate(failures, start=1):
            if isinstance(failure, BaseException):
                logger.warning("%s failure %d: %r", context, idx, failure)
            elif isinstance(failure, tuple) and len(failure) == 2:
                client, result = failure
                logger.warning(
                    "%s failure %d | client=%s | result=%r",
                    context,
                    idx,
                    getattr(client, "cid", "?"),
                    result,
                )
            else:
                logger.warning("%s failure %d: %r", context, idx, failure)




def make_fit_config_fn(cfg: DictConfig, local_epochs: int, batch_size: int):
    def fit_config(server_round: int) -> dict[str, Any]:
        return {
            "server_round": int(server_round),
            "local_epochs": int(local_epochs),
            "batch_size": int(batch_size),
            "dataset_name": str(cfg.dataset.name),
            "alpha": float(cfg.dataset.preprocessing.alpha),
        }
    return fit_config


def get_strategy(cfg: DictConfig, metrics_sink: Any | None = None) -> Strategy:
    strat_name = str(cfg.strategy.name).lower()
    device = torch.device("cpu")
    logger.info(
        "Server strategy setup | strategy=%s | server_device=%s | logical_rounds=%d | flower_rounds=%d",
        strat_name,
        device,
        int(cfg.server.num_rounds),
        get_effective_num_rounds(cfg),
    )

    local_epochs = int(OmegaConf.select(cfg, "training.local_epochs", default=1))
    fit_config_fn = make_fit_config_fn(
        cfg=cfg,
        local_epochs=local_epochs,
        batch_size=cfg.training.batch_size,
    )
    central_evaluate_fn = make_central_evaluate_fn(cfg, device, metrics_sink=metrics_sink)

    args = dict(
        fraction_fit=cfg.server.fraction_fit,
        fraction_evaluate=cfg.server.fraction_evaluate,
        min_fit_clients=cfg.server.min_fit_clients,
        min_evaluate_clients=cfg.server.min_evaluate_clients,
        min_available_clients=cfg.server.min_available_clients,
        initial_parameters=_initial_parameters_from_checkpoint(cfg, device),
        on_fit_config_fn=fit_config_fn,
        evaluate_fn=central_evaluate_fn,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )

    if strat_name == "fedtros_pr":
        logger.info("--- Strategy: FedTROS-PR (Federated Teacher-Regularized Open-Set Recognition with Prototype-Rank Rejection) ---")
        return FedTROSStrategy(
            cfg=cfg,
            metrics_sink=metrics_sink,
            **args,
        )

    if strat_name == "fedprox":
        proximal_mu = float(cfg.server.proximal_mu)
        logger.info("--- Strategy: FedProx (mu=%.2f) with Model Saving ---", proximal_mu)
        return SaveModelFedProx(
            cfg=cfg,
            metrics_sink=metrics_sink,
            proximal_mu=proximal_mu,
            **args,
        )

    logger.info("--- Strategy: FedAvg with Model Saving ---")
    return SaveModelFedAvg(cfg=cfg, metrics_sink=metrics_sink, **args)


def get_effective_num_rounds(cfg: DictConfig) -> int:
    return int(cfg.server.num_rounds)



def run_server(cfg: DictConfig, device: torch.device | None = None) -> None:
    # Initialize the global agent reference for saving capabilities
    requested_device = device
    server_device = torch.device("cpu")
    init_global_agent_ref(cfg, server_device)

    strategy = get_strategy(cfg)

    logger.info(
        "Starting server at %s | requested_device=%s | effective_device=%s | flower_rounds=%s | logical_rounds=%s",
        cfg.server.address,
        requested_device,
        server_device,
        get_effective_num_rounds(cfg),
        cfg.server.num_rounds,
    )
    try:
        fl.server.start_server(
            server_address=cfg.server.address,
            config=fl.server.ServerConfig(num_rounds=get_effective_num_rounds(cfg)),
            strategy=strategy,
        )
    except RuntimeError as exc:
        if "Failed to bind to address" in str(exc):
            logger.error("Unable to bind to %s. Port in use.", cfg.server.address)
            raise SystemExit(1) from exc
        raise
