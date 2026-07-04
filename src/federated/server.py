import csv
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
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

from src.agents.agent import Agent
from src.checkpointing.checkpoints import load_agent_checkpoint
from src.federated.selection_utils import (
    alignment_multiplier,
    centered_utility,
    combine_utility_score,
    critic_utility_score,
    select_utility_records,
    validation_team_reward,
)
from src.models.models import OpenSetQChainModelFactory
logger = logging.getLogger("Server")

EPS = 1e-8

AUDIT_SCALAR_KEYS = (
    "reward_norm",
    "history_reward_norm",
    "f1_macro",
    "accuracy",
    "td_stability",
    "novelty",
    "class_entropy",
    "label_coverage",
    "generator_correct_frac",
    "steps_norm",
)

# Global reference to hold the model architecture for saving
GLOBAL_AGENT_REF: Agent | None = None


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


def make_central_evaluate_fn(cfg: DictConfig, device: torch.device):
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
    eval_agent = Agent(OpenSetQChainModelFactory(cfg.model), cfg.training, device=device)
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
        eval_agent.prior_net.eval()
        eval_agent.value_net_main.eval()
        total_loss = 0.0
        total = 0
        y_true: list[int] = []
        y_pred: list[int] = []

        with torch.no_grad():
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(device).float()
                batch_labels = batch_labels.to(device).long()
                mu, _ = eval_agent.prior_net(batch_features)
                logits = eval_agent.value_net_main(mu, batch_features)
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
        return float(avg_loss), metrics

    return evaluate


def init_global_agent_ref(cfg: DictConfig, device: torch.device):
    """
    Initialize a dummy Agent on the server.
    This is used solely to map the flat list of parameters from Federated Learning
    back into a state_dict for saving .pt files.
    """
    global GLOBAL_AGENT_REF
    if Agent is None or OpenSetQChainModelFactory is None:
        logger.warning(
            "Could not import Agent/Models. Checkpoints will be saved as raw NumPy arrays only."
        )
        return

    try:
        model_factory = OpenSetQChainModelFactory(cfg.model)
        # We don't need the optimizer or replay buffer on the server, just the nets
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
    Universal function to save the global model state.
    It saves two formats:
      1. .npz (Raw FL weights) - Good for resuming FL.
      2. .pt (PyTorch State Dict) - Good for inference/eval/deployment.
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
            "round": round_num,
            "epoch": round_num,
            "global_step": round_num,
            "metrics": {"federated/round": float(round_num), **dict(metrics or {})},
            "config": OmegaConf.to_container(cfg, resolve=True),
            "prior_net": GLOBAL_AGENT_REF.prior_net.state_dict(),
            "recognition_net": GLOBAL_AGENT_REF.recognition_net.state_dict(),
            "value_net_main": GLOBAL_AGENT_REF.value_net_main.state_dict(),
            "value_net_target": GLOBAL_AGENT_REF.value_net_target.state_dict(),
            "generation_net": (
                GLOBAL_AGENT_REF.generation_net.state_dict()
                if GLOBAL_AGENT_REF.generation_net
                else None
            ),
            "optimizer_prior": GLOBAL_AGENT_REF.optimizer_prior.state_dict(),
            "optimizer_q_rl": GLOBAL_AGENT_REF.optimizer_q_rl.state_dict(),
        }

        round_path = model_dir / f"global_model_round_{round_num:04d}.pt"
        torch.save(checkpoint, round_path)
        torch.save(checkpoint, model_dir / "global_model_latest.pt")
        torch.save(checkpoint, _resolve_path(cfg.checkpointing.latest_checkpoint_path))
        # latest_checkpoint.pt is the canonical final model. best_model.pt is
        # updated by monitored validation when available; otherwise, keep legacy
        # behavior only when central validation is disabled.
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
        logger.info("Saved global model checkpoint to %s", round_path)

    except Exception as e:
        logger.error("Failed to save PyTorch checkpoint: %s", e)


def fit_config_fn(server_round: int) -> dict[str, fl.common.Scalar]:
    return {"server_round": server_round, "phase": "standard"}


def _initial_parameters_from_checkpoint(
    cfg: DictConfig, device: torch.device
) -> Parameters | None:
    resume_from = OmegaConf.select(cfg, "federated.resume_from", default=None)
    if not resume_from:
        return None
    if GLOBAL_AGENT_REF is None:
        init_global_agent_ref(cfg, device)
    if GLOBAL_AGENT_REF is None:
        raise RuntimeError("Global agent reference is not initialized; cannot resume FL.")
    checkpoint_path = _resolve_path(resume_from)
    load_agent_checkpoint(
        GLOBAL_AGENT_REF,
        checkpoint_path,
        device,
        strict=bool(cfg.checkpointing.strict_load),
        load_optimizers=False,
    )
    logger.info("Loaded federated initial parameters from %s", checkpoint_path)
    return ndarrays_to_parameters(GLOBAL_AGENT_REF.get_federated_parameters())


# --- CUSTOM STRATEGIES WITH SAVING LOGIC ---


class SaveModelFedAvg(FedAvg):
    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

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

        if aggregated_parameters is not None:
            # Save the model
            save_global_model(
                aggregated_parameters, server_round, self.cfg, metrics=dict(aggregated_metrics)
            )

        return aggregated_parameters, aggregated_metrics


class SaveModelFedProx(FedProx):
    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            save_global_model(
                aggregated_parameters, server_round, self.cfg, metrics=dict(aggregated_metrics)
            )

        return aggregated_parameters, aggregated_metrics


class DKDFedOSStrategy(FedAvg):
    """Sentinel-style dynamic-KD strategy for the CVAE-DQN IDS stack.

    Only the compact student model is sent to and aggregated by the server.
    The CVAE-DQN teacher remains local for personalization and open-set
    reconstruction, preventing extreme non-IID clients from overwriting a shared
    global teacher with one-class local evidence.
    """

    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if GLOBAL_AGENT_REF is None:
            raise RuntimeError("DKD-FedOS needs a server Agent reference with a student model.")
        self.cfg = cfg
        self.server_lr = float(OmegaConf.select(cfg, "strategy.server_lr", default=1.0))
        self.momentum_beta = float(OmegaConf.select(cfg, "strategy.server_momentum", default=0.9))
        self.monitor_path = _resolve_path(
            OmegaConf.select(cfg, "strategy.monitor_path", default="outputs/dkd_fedos_monitoring.jsonl")
        )
        self.monitor_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_reliable_samples = float(
            OmegaConf.select(cfg, "strategy.min_reliable_samples", default=0.0)
        )
        self.student_aggregation_mode = str(
            OmegaConf.select(cfg, "strategy.student_aggregation_mode", default="reliability_weighted_average")
        ).lower()
        self.student_avg_warmup_rounds = int(
            OmegaConf.select(cfg, "strategy.student_avg_warmup_rounds", default=3)
        )
        self.normalized_momentum_beta = float(
            OmegaConf.select(cfg, "strategy.normalized_server_momentum", default=self.momentum_beta)
        )
        self.reliability_reference_samples = float(
            OmegaConf.select(cfg, "strategy.reliability_reference_samples", default=1000.0)
        )
        self.reliability_min_weight = float(
            OmegaConf.select(cfg, "strategy.reliability_min_weight", default=0.02)
        )
        self.reliability_coverage_power = float(
            OmegaConf.select(cfg, "strategy.reliability_coverage_power", default=1.0)
        )
        self.reliability_entropy_power = float(
            OmegaConf.select(cfg, "strategy.reliability_entropy_power", default=1.0)
        )
        self.global_student_parameters = ndarrays_to_parameters(GLOBAL_AGENT_REF.get_student_parameters())
        self.momentum: list[np.ndarray] | None = None

        expected_hidden = [512, 256, 128]
        actual_hidden = list(OmegaConf.select(cfg, "training.dkd_student_hidden_dims", default=[]))
        anchor_weight = float(OmegaConf.select(cfg, "training.dkd_global_anchor_weight", default=0.0))
        if self.student_aggregation_mode != "reliability_weighted_average":
            raise ValueError(
                "DKD-FedOS V5/V6 contract violation: "
                f"student_aggregation_mode must be reliability_weighted_average, got {self.student_aggregation_mode!r}."
            )
        if anchor_weight <= 0.0:
            raise ValueError(
                "DKD-FedOS V5/V6 contract violation: training.dkd_global_anchor_weight must be > 0."
            )
        if [int(v) for v in actual_hidden] != expected_hidden:
            raise ValueError(
                "DKD-FedOS V5/V6 contract violation: "
                f"student_hidden_dims must be {expected_hidden}, got {actual_hidden}."
            )

        logger.info(
            "DKD-FedOS V5 STUDENT-ANCHOR ACTIVE | "
            "student_aggregation_mode=%s | global_anchor_enabled=%s | "
            "global_anchor_weight=%.3f | student_hidden_dims=%s | student_layers=%d | "
            "min_reliable_samples=%.1f | warmup=%d",
            self.student_aggregation_mode,
            bool(anchor_weight > 0.0),
            anchor_weight,
            actual_hidden,
            len(GLOBAL_AGENT_REF.get_student_parameters()),
            self.min_reliable_samples,
            self.student_avg_warmup_rounds,
        )

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        _ = parameters
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_fit_clients,
        )
        fit_ins = FitIns(
            self.global_student_parameters,
            {"server_round": server_round, "phase": "dkd_fedos"},
        )
        logger.info("DKD-FedOS round=%d sampled=%d", server_round, len(clients))
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
            {"server_round": server_round, "phase": "dkd_fedos"},
        )
        logger.info("DKD-FedOS evaluation round=%d clients=%d", server_round, len(clients))
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(self, server_round: int, results, failures):
        if failures:
            self._log_failures("DKD-FedOS", failures)
        records = []
        for client, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            if not weights:
                logger.warning("DKD-FedOS client %s returned empty student weights; skipping", client.cid)
                continue
            records.append(
                {
                    "cid": getattr(client, "cid", "?"),
                    "weights": weights,
                    "num_examples": float(fit_res.num_examples),
                    "metrics": dict(fit_res.metrics),
                }
            )
        if records:
            max_client_examples = max(float(record.get("num_examples", 0.0)) for record in records)
            for record in records:
                record["reliability_weight"] = self._client_reliability_weight(
                    record, max_examples=max_client_examples
                )
        if not records:
            logger.warning("DKD-FedOS round=%d has no usable client updates", server_round)
            return self.global_student_parameters, {}

        reliable_records = [
            record for record in records if float(record["num_examples"]) >= self.min_reliable_samples
        ]
        excluded_records = [
            record for record in records if float(record["num_examples"]) < self.min_reliable_samples
        ]
        if not reliable_records:
            logger.warning(
                "DKD-FedOS round=%d has no clients above min_reliable_samples=%.1f; using all updates.",
                server_round,
                self.min_reliable_samples,
            )
            reliable_records = records
            excluded_records = []

        base = parameters_to_ndarrays(self.global_student_parameters)
        avg_weights = []
        reliability_weights = [float(record.get("reliability_weight", 1.0)) for record in reliable_records]
        weight_sum = max(float(np.sum(reliability_weights)), EPS)
        for layer_idx in range(len(base)):
            layer_sum = np.zeros_like(base[layer_idx])
            for record, reliability_weight in zip(reliable_records, reliability_weights, strict=True):
                layer_sum += float(reliability_weight) * record["weights"][layer_idx]
            avg_weights.append(layer_sum / weight_sum)

        distance_to_avg_before = self._weight_list_norm(
            [base_layer - avg_layer for base_layer, avg_layer in zip(base, avg_weights, strict=True)]
        )
        use_warm_average = self.student_aggregation_mode in {
            "equal_average",
            "avg",
            "average",
            "weighted_average",
            "reliability_weighted_average",
        } or (
            self.student_aggregation_mode == "warm_avg_then_normalized"
            and int(server_round) <= int(self.student_avg_warmup_rounds)
        )

        pseudo_gradients = []
        grad_norms = []
        normalized_grad_norms = []
        if use_warm_average:
            new_weights = avg_weights
            self.momentum = None
            aggregation_mode_used = (
                "reliability_weighted_average"
                if self.student_aggregation_mode == "reliability_weighted_average"
                else "equal_average_warmup"
            )
        else:
            for record in reliable_records:
                grad = [
                    base_layer - local_layer
                    for base_layer, local_layer in zip(base, record["weights"], strict=True)
                ]
                norm = self._weight_list_norm(grad)
                grad_norms.append(norm)
                normalized = [layer / max(norm, EPS) for layer in grad]
                normalized_grad_norms.append(self._weight_list_norm(normalized))
                pseudo_gradients.append(normalized)

            mean_grad = []
            for layer_idx in range(len(base)):
                layer_sum = np.zeros_like(base[layer_idx])
                for grad, reliability_weight in zip(pseudo_gradients, reliability_weights, strict=True):
                    layer_sum += float(reliability_weight) * grad[layer_idx]
                mean_grad.append(layer_sum / weight_sum)

            if self.momentum is None:
                self.momentum = [np.zeros_like(layer) for layer in mean_grad]
            beta = float(np.clip(self.normalized_momentum_beta, 0.0, 0.999))
            self.momentum = [
                (beta * old) + ((1.0 - beta) * grad)
                for old, grad in zip(self.momentum, mean_grad, strict=True)
            ]
            new_weights = [
                base_layer - (self.server_lr * mom_layer)
                for base_layer, mom_layer in zip(base, self.momentum, strict=True)
            ]
            aggregation_mode_used = "normalized_gradient"

        distance_to_avg_after = self._weight_list_norm(
            [new_layer - avg_layer for new_layer, avg_layer in zip(new_weights, avg_weights, strict=True)]
        )
        global_norm_before = self._weight_list_norm(base)
        global_norm_after = self._weight_list_norm(new_weights)
        avg_local_norm = self._weight_list_norm(avg_weights)

        self.global_student_parameters = ndarrays_to_parameters(new_weights)

        fit_metrics = [(int(record["num_examples"]), record["metrics"]) for record in reliable_records]
        metrics = aggregate_fit_metrics(fit_metrics)
        metrics.update(
            {
                "dkd_fedos_clients": float(len(records)),
                "dkd_fedos_included_clients": float(len(reliable_records)),
                "dkd_fedos_excluded_clients": float(len(excluded_records)),
                "dkd_fedos_mean_reliability_weight": float(np.mean(reliability_weights) if reliability_weights else 0.0),
                "dkd_fedos_min_reliability_weight": float(np.min(reliability_weights) if reliability_weights else 0.0),
                "dkd_fedos_max_reliability_weight": float(np.max(reliability_weights) if reliability_weights else 0.0),
                "dkd_fedos_mean_student_grad_norm": float(np.mean(grad_norms) if grad_norms else 0.0),
                "dkd_fedos_max_student_grad_norm": float(np.max(grad_norms) if grad_norms else 0.0),
                "dkd_fedos_mean_normalized_grad_norm": float(np.mean(normalized_grad_norms) if normalized_grad_norms else 0.0),
                "dkd_fedos_distance_to_avg_before": float(distance_to_avg_before),
                "dkd_fedos_distance_to_avg_after": float(distance_to_avg_after),
                "dkd_fedos_global_norm_before": float(global_norm_before),
                "dkd_fedos_global_norm_after": float(global_norm_after),
                "dkd_fedos_avg_local_norm": float(avg_local_norm),
            }
        )
        self._save_student_checkpoint(new_weights, server_round, metrics)
        self._write_monitor_event(
            {
                "event": "dkd_fedos_aggregation",
                "server_round": server_round,
                "clients": [record["cid"] for record in records],
                "included_clients": [record["cid"] for record in reliable_records],
                "excluded_clients": [record["cid"] for record in excluded_records],
                "grad_norms": grad_norms,
                "normalized_grad_norms": normalized_grad_norms,
                "aggregation_mode": aggregation_mode_used,
                "reliability_weights": {str(record["cid"]): float(record.get("reliability_weight", 1.0)) for record in reliable_records},
                "distance_to_avg_before": distance_to_avg_before,
                "distance_to_avg_after": distance_to_avg_after,
                "global_norm_before": global_norm_before,
                "global_norm_after": global_norm_after,
                "avg_local_norm": avg_local_norm,
                "metrics": metrics,
            }
        )
        logger.info(
            "DKD-FedOS aggregation | round=%d mode=%s clients=%d included=%d excluded=%d dist_to_avg %.4f->%.4f",
            server_round,
            aggregation_mode_used,
            len(records),
            len(reliable_records),
            len(excluded_records),
            distance_to_avg_before,
            distance_to_avg_after,
        )
        return self.global_student_parameters, metrics

    def _client_reliability_weight(self, record: dict[str, Any], *, max_examples: float) -> float:
        """Reliability score for student aggregation under quantity+label skew.

        v6 formula requested for DKD-FedOS stability:

            reliability_i = sqrt(num_samples_i / max_samples_in_round)
                            * label_coverage_i
                            * class_entropy_i

        Then clamp into [reliability_min_weight, 1.0].  This makes clients with
        many samples but one dominant class contribute only a small, explicit
        anchor-preserved update instead of dominating the global student.
        """
        metrics = record.get("metrics", {}) or {}
        num_examples = max(float(record.get("num_examples", 0.0)), 0.0)
        sample_factor = float(np.sqrt(num_examples / max(float(max_examples), 1.0)))
        sample_factor = float(np.clip(sample_factor, 0.0, 1.0))
        coverage = float(metrics.get("label_coverage", 0.0) or 0.0)
        entropy = float(metrics.get("class_entropy", 0.0) or 0.0)
        coverage_factor = max(0.0, min(1.0, coverage)) ** max(self.reliability_coverage_power, 0.0)
        entropy_factor = max(0.0, min(1.0, entropy)) ** max(self.reliability_entropy_power, 0.0)
        weight = sample_factor * coverage_factor * entropy_factor
        return float(np.clip(weight, self.reliability_min_weight, 1.0))

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
        checkpoint = {
            "round": round_num,
            "epoch": round_num,
            "global_step": round_num,
            "method": "dkd_fedos",
            "metrics": {"federated/round": float(round_num), **dict(metrics or {})},
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "student_model": GLOBAL_AGENT_REF.student_model.state_dict(),
        }
        round_path = model_dir / f"dkd_fedos_student_round_{round_num:04d}.pt"
        torch.save(checkpoint, round_path)
        torch.save(checkpoint, model_dir / "dkd_fedos_student_latest.pt")
        logger.info("Saved DKD-FedOS student checkpoint to %s", round_path)

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
            logger.warning("Failed to write DKD-FedOS monitor event: %s", exc)

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


class FedGPAStrategy(FedAvg):
    """
    FedGPA for the CVAE-DQN/open-set stack.

    Paper mapping:
    - FedGPA local-global prototype alignment -> latent/Q prototype regularization on clients.
    - FedGPA personalized aggregation -> per-client server models with separate weights for
      representation modules and Q/classifier modules.

    Model-stack mapping:
    - prior_net: slowly aggregated feature extractor.
    - recognition_net: frozen/slowly aggregated personalized latent module.
    - value_net_main: strongly aggregated Q classifier/head.
    - generation_net: frozen by default because open-set generator averaging was damaging logs.
    """

    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.device = torch.device("cpu")
        if GLOBAL_AGENT_REF is None:
            init_global_agent_ref(cfg, self.device)
        if GLOBAL_AGENT_REF is None:
            raise RuntimeError("FedGPA needs a server Agent reference for module slicing.")

        self.module_slices = self._build_module_slices()
        self.global_prototypes: dict[str, list[float]] = {}
        self.personalized_weights: dict[str, list[np.ndarray]] = {}
        self.reference_weights: list[np.ndarray] | None = None
        self.monitor_path = _resolve_path(
            OmegaConf.select(cfg, "strategy.monitor_path", default="runs/fedgpa_monitoring.jsonl")
        )
        self.monitor_path.parent.mkdir(parents=True, exist_ok=True)

        self.prototype_lambda = float(OmegaConf.select(cfg, "strategy.prototype_lambda", default=0.05))
        self.prototype_mu = float(OmegaConf.select(cfg, "strategy.prototype_mu", default=0.50))
        self.prototype_feature = str(
            OmegaConf.select(cfg, "strategy.prototype_feature", default="latent_q")
        )
        self.prior_mix = float(OmegaConf.select(cfg, "strategy.prior_mix", default=0.25))
        self.recognition_mix = float(
            OmegaConf.select(cfg, "strategy.recognition_mix", default=0.05)
        )
        self.value_mix = float(OmegaConf.select(cfg, "strategy.value_mix", default=1.0))
        self.generation_mix = float(OmegaConf.select(cfg, "strategy.generation_mix", default=0.0))
        self.classifier_self_weight = float(
            OmegaConf.select(cfg, "strategy.classifier_self_weight", default=0.25)
        )
        self.distance_temperature = max(
            float(OmegaConf.select(cfg, "strategy.distance_temperature", default=1.0)), EPS
        )

        logger.info(
            "FedGPA configured | lambda=%.4f mu=%.3f feature=%s mixes(prior=%.2f recog=%.2f value=%.2f gen=%.2f)",
            self.prototype_lambda,
            self.prototype_mu,
            self.prototype_feature,
            self.prior_mix,
            self.recognition_mix,
            self.value_mix,
            self.generation_mix,
        )

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        reference = self._parameters_to_weights(parameters)
        if self.reference_weights is None:
            self.reference_weights = reference
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        config = dict(self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {})
        config.update(
            {
                "phase": "fedgpa",
                "fedgpa_lambda": float(self.prototype_lambda),
                "fedgpa_feature": self.prototype_feature,
                "fedgpa_global_prototypes": json.dumps(self.global_prototypes, sort_keys=True),
            }
        )
        fit_pairs = []
        for client in clients:
            client_weights = self.personalized_weights.get(client.cid, self.reference_weights)
            if client_weights is None:
                client_weights = reference
            fit_pairs.append((client, FitIns(ndarrays_to_parameters(client_weights), config)))
        logger.info("FedGPA round=%d sampled=%d", server_round, len(fit_pairs))
        return fit_pairs

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        if self.fraction_evaluate == 0.0:
            return []
        reference = self._parameters_to_weights(parameters)
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        if sample_size == 0:
            return []
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        config = dict(self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn else {})
        config.setdefault("server_round", server_round)
        evaluate_pairs = []
        for client in clients:
            client_weights = self.personalized_weights.get(client.cid, self.reference_weights)
            if client_weights is None:
                client_weights = reference
            evaluate_pairs.append((client, EvaluateIns(ndarrays_to_parameters(client_weights), config)))
        logger.info("FedGPA evaluation round=%d clients=%d", server_round, len(evaluate_pairs))
        return evaluate_pairs

    def aggregate_fit(self, server_round: int, results, failures):
        if failures:
            self._log_failures("FedGPA", failures)
        if not results:
            return (
                ndarrays_to_parameters(self.reference_weights) if self.reference_weights is not None else None,
                {"fedgpa_clients": 0.0},
            )

        records = []
        for client, fit_res in sorted(results, key=lambda item: item[0].cid):
            weights = parameters_to_ndarrays(fit_res.parameters)
            if not weights:
                logger.warning("FedGPA client %s returned empty weights; skipping", client.cid)
                continue
            prototypes, counts = self._parse_prototype_metrics(fit_res.metrics)
            variance = self._float_metric(fit_res.metrics, "fedgpa_variance", 1.0)
            records.append(
                {
                    "cid": client.cid,
                    "weights": weights,
                    "num_examples": max(float(fit_res.num_examples), 1.0),
                    "prototypes": prototypes,
                    "counts": counts,
                    "variance": max(float(variance), EPS),
                    "metrics": dict(fit_res.metrics),
                }
            )
            records[-1]["reliability_weight"] = self._client_reliability_weight(records[-1])

        if not records:
            return (
                ndarrays_to_parameters(self.reference_weights) if self.reference_weights is not None else None,
                {"fedgpa_clients": 0.0},
            )

        self.global_prototypes = self._aggregate_global_prototypes(records)
        distance_matrix = self._client_distance_matrix(records)
        alpha = self._feature_weights(distance_matrix, records)
        beta = self._classifier_weights(distance_matrix, records)
        self._update_personalized_weights(records, alpha, beta)
        reference_weights = self._weighted_average_weight_lists(
            [self.personalized_weights[record["cid"]] for record in records],
            [record["num_examples"] for record in records],
        )
        self.reference_weights = reference_weights
        reference_params = ndarrays_to_parameters(reference_weights)
        save_global_model(reference_params, server_round, self.cfg)

        alpha_diag = float(np.mean(np.diag(alpha))) if alpha.size else 0.0
        beta_diag = float(np.mean(np.diag(beta))) if beta.size else 0.0
        metrics = {
            "fedgpa_clients": float(len(records)),
            "fedgpa_global_proto_classes": float(len(self.global_prototypes)),
            "fedgpa_alpha_self_mean": alpha_diag,
            "fedgpa_beta_self_mean": beta_diag,
            "fedgpa_distance_mean": float(distance_matrix.mean()) if distance_matrix.size else 0.0,
            "fedgpa_prior_mix": float(self.prior_mix),
            "fedgpa_recognition_mix": float(self.recognition_mix),
            "fedgpa_value_mix": float(self.value_mix),
            "fedgpa_generation_mix": float(self.generation_mix),
        }
        metrics.update(aggregate_fit_metrics([(int(r["num_examples"]), r["metrics"]) for r in records]))
        self._write_monitor_event(
            {
                "event": "fedgpa_aggregation",
                "server_round": server_round,
                "clients": [record["cid"] for record in records],
                "prototype_classes": sorted(self.global_prototypes.keys(), key=lambda x: int(x)),
                "alpha": alpha.tolist(),
                "beta": beta.tolist(),
                "distance_matrix": distance_matrix.tolist(),
                "metrics": metrics,
            }
        )
        logger.info(
            "FedGPA aggregation | round=%d clients=%d proto_classes=%d alpha_self=%.3f beta_self=%.3f",
            server_round,
            len(records),
            len(self.global_prototypes),
            alpha_diag,
            beta_diag,
        )
        return reference_params, metrics

    def _update_personalized_weights(
        self, records: list[dict[str, Any]], alpha: np.ndarray, beta: np.ndarray
    ) -> None:
        client_weights = [record["weights"] for record in records]
        for i, record in enumerate(records):
            own = record["weights"]
            new_weights = [np.array(layer, copy=True) for layer in own]
            self._blend_module(new_weights, own, client_weights, alpha[i], "prior_net", self.prior_mix)
            self._blend_module(
                new_weights, own, client_weights, alpha[i], "recognition_net", self.recognition_mix
            )
            self._blend_module(
                new_weights, own, client_weights, beta[i], "value_net_main", self.value_mix
            )
            self._blend_module(
                new_weights, own, client_weights, beta[i], "generation_net", self.generation_mix
            )
            self.personalized_weights[record["cid"]] = new_weights

    def _blend_module(
        self,
        target_weights: list[np.ndarray],
        own_weights: list[np.ndarray],
        client_weights: list[list[np.ndarray]],
        coefficients: np.ndarray,
        module_name: str,
        mix: float,
    ) -> None:
        indices = self.module_slices.get(module_name, [])
        if not indices:
            return
        mix = float(np.clip(mix, 0.0, 1.0))
        if mix <= 0.0:
            for idx in indices:
                target_weights[idx] = np.array(own_weights[idx], copy=True)
            return
        coeffs = np.asarray(coefficients, dtype=np.float64)
        coeffs = coeffs / max(float(coeffs.sum()), EPS)
        for idx in indices:
            aggregate = np.zeros_like(own_weights[idx])
            for coeff, weights in zip(coeffs, client_weights, strict=True):
                aggregate += weights[idx] * float(coeff)
            target_weights[idx] = ((1.0 - mix) * own_weights[idx]) + (mix * aggregate)

    def _aggregate_global_prototypes(self, records: list[dict[str, Any]]) -> dict[str, list[float]]:
        sums: dict[str, np.ndarray] = {}
        counts: dict[str, float] = {}
        for record in records:
            for class_key, proto in record["prototypes"].items():
                count = float(record["counts"].get(class_key, 0.0))
                if count <= 0:
                    continue
                vector = np.asarray(proto, dtype=np.float64)
                if vector.ndim != 1:
                    continue
                if class_key not in sums:
                    sums[class_key] = np.zeros_like(vector, dtype=np.float64)
                    counts[class_key] = 0.0
                if sums[class_key].shape != vector.shape:
                    continue
                sums[class_key] += count * vector
                counts[class_key] += count
        global_prototypes = {}
        for class_key, vector_sum in sums.items():
            denom = max(counts.get(class_key, 0.0), EPS)
            global_prototypes[class_key] = [float(x) for x in (vector_sum / denom).tolist()]
        return global_prototypes

    def _client_distance_matrix(self, records: list[dict[str, Any]]) -> np.ndarray:
        num_clients = len(records)
        distances = np.zeros((num_clients, num_clients), dtype=np.float64)
        for i, rec_i in enumerate(records):
            total_i = max(sum(float(v) for v in rec_i["counts"].values()), 1.0)
            for j, rec_j in enumerate(records):
                if i == j:
                    distances[i, j] = 0.0
                    continue
                weighted_distance = 0.0
                weight_sum = 0.0
                for class_key, proto_i in rec_i["prototypes"].items():
                    if class_key not in rec_j["prototypes"]:
                        continue
                    weight = float(rec_i["counts"].get(class_key, 0.0)) / total_i
                    vec_i = np.asarray(proto_i, dtype=np.float64)
                    vec_j = np.asarray(rec_j["prototypes"][class_key], dtype=np.float64)
                    if vec_i.shape != vec_j.shape:
                        continue
                    weighted_distance += weight * float(np.linalg.norm(vec_i - vec_j))
                    weight_sum += weight
                if weight_sum <= EPS:
                    distances[i, j] = 1.0 / EPS
                else:
                    distances[i, j] = weighted_distance / max(weight_sum, EPS)
        finite = distances[np.isfinite(distances)]
        if finite.size:
            cap = max(float(np.percentile(finite, 95)), EPS)
            distances = np.where(np.isfinite(distances), distances, cap)
            distances = np.minimum(distances, cap)
        return distances

    def _feature_weights(self, distances: np.ndarray, records: list[dict[str, Any]]) -> np.ndarray:
        if distances.size == 0:
            return distances
        positive = distances[distances > EPS]
        scale = float(np.median(positive)) if positive.size else 1.0
        scale = max(scale * self.distance_temperature, EPS)
        # Feature extractors should remain fairly global. A bounded exponential
        # similarity avoids the self-client infinity that raw 1 / P would create.
        sim = np.exp(-distances / scale)
        row_sums = sim.sum(axis=1, keepdims=True)
        sim_weights = sim / np.maximum(row_sums, EPS)
        sample_counts = np.asarray([record["num_examples"] for record in records], dtype=np.float64)
        sample_weights = sample_counts / max(float(sample_counts.sum()), EPS)
        mu = float(np.clip(self.prototype_mu, 0.0, 1.0))
        alpha = (mu * sim_weights) + ((1.0 - mu) * sample_weights.reshape(1, -1))
        alpha = alpha / np.maximum(alpha.sum(axis=1, keepdims=True), EPS)
        return alpha

    def _classifier_weights(self, distances: np.ndarray, records: list[dict[str, Any]]) -> np.ndarray:
        num_clients = len(records)
        if num_clients == 0:
            return np.zeros((0, 0), dtype=np.float64)
        variances = np.asarray([record["variance"] for record in records], dtype=np.float64)
        beta_rows = []
        for i in range(num_clients):
            # Practical diagonal-Q simplex solution: beta_j proportional to 1 / Q_j.
            q_diag = variances + (distances[i] / self.distance_temperature) + EPS
            raw = 1.0 / np.maximum(q_diag, EPS)
            raw = raw / max(float(raw.sum()), EPS)
            local_prior = np.zeros(num_clients, dtype=np.float64)
            local_prior[i] = 1.0
            self_weight = float(np.clip(self.classifier_self_weight, 0.0, 1.0))
            beta = ((1.0 - self_weight) * raw) + (self_weight * local_prior)
            beta = beta / max(float(beta.sum()), EPS)
            beta_rows.append(beta)
        return np.stack(beta_rows, axis=0)

    def _weighted_average_weight_lists(
        self, weight_lists: list[list[np.ndarray]], weights: list[float]
    ) -> list[np.ndarray]:
        coeffs = np.asarray(weights, dtype=np.float64)
        coeffs = coeffs / max(float(coeffs.sum()), EPS)
        averaged = []
        for layer_idx in range(len(weight_lists[0])):
            layer = np.zeros_like(weight_lists[0][layer_idx])
            for coeff, weight_list in zip(coeffs, weight_lists, strict=True):
                layer += weight_list[layer_idx] * float(coeff)
            averaged.append(layer)
        return averaged

    def _parse_prototype_metrics(
        self, metrics: dict[str, Scalar]
    ) -> tuple[dict[str, list[float]], dict[str, int]]:
        try:
            prototypes_raw = json.loads(str(metrics.get("fedgpa_prototypes", "{}")))
        except json.JSONDecodeError:
            prototypes_raw = {}
        try:
            counts_raw = json.loads(str(metrics.get("fedgpa_counts", "{}")))
        except json.JSONDecodeError:
            counts_raw = {}
        prototypes: dict[str, list[float]] = {}
        counts: dict[str, int] = {}
        if isinstance(prototypes_raw, dict):
            for key, value in prototypes_raw.items():
                if not isinstance(value, list):
                    continue
                try:
                    class_key = str(int(key))
                    vector = [float(v) for v in value]
                except (TypeError, ValueError):
                    continue
                if vector:
                    prototypes[class_key] = vector
        if isinstance(counts_raw, dict):
            for key, value in counts_raw.items():
                try:
                    counts[str(int(key))] = int(value)
                except (TypeError, ValueError):
                    continue
        return prototypes, counts

    def _build_module_slices(self) -> dict[str, list[int]]:
        assert GLOBAL_AGENT_REF is not None
        prior_len = len(GLOBAL_AGENT_REF.prior_net.state_dict())
        recog_len = len(GLOBAL_AGENT_REF.recognition_net.state_dict())
        value_len = len(GLOBAL_AGENT_REF.value_net_main.state_dict())
        gen_len = (
            len(GLOBAL_AGENT_REF.generation_net.state_dict())
            if GLOBAL_AGENT_REF.generation_net is not None
            else 0
        )
        cursor = 0
        slices: dict[str, list[int]] = {}
        slices["prior_net"] = list(range(cursor, cursor + prior_len))
        cursor += prior_len
        slices["recognition_net"] = list(range(cursor, cursor + recog_len))
        cursor += recog_len
        slices["value_net_main"] = list(range(cursor, cursor + value_len))
        cursor += value_len
        slices["generation_net"] = list(range(cursor, cursor + gen_len))
        logger.info(
            "FedGPA module slices | prior=%d recognition=%d value=%d generation=%d total=%d",
            prior_len,
            recog_len,
            value_len,
            gen_len,
            cursor + gen_len,
        )
        return slices

    def _parameters_to_weights(self, parameters: Parameters | list[np.ndarray]) -> list[np.ndarray]:
        return parameters if isinstance(parameters, list) else parameters_to_ndarrays(parameters)

    def _write_monitor_event(self, event: dict[str, Any]) -> None:
        try:
            with open(self.monitor_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write FedGPA monitor event: %s", exc)

    @staticmethod
    def _float_metric(metrics: dict[str, Scalar], key: str, default: float = 0.0) -> float:
        value = metrics.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

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


class FMRLAdaptiveVectorAlignedAggregationStrategy(FedAvg):
    """
    FMRL-AVA for CF-MARLOS-AVA.

    Stage A follows the FMRL-LA paper's two-phase communication idea:
    clients train locally and upload hidden/reward/profile metadata so the
    server-side critic/mixer can estimate utility and select clients.

    Stage B follows the FedAWA paper's client-vector idea: selected clients
    upload model weights, the server forms local deltas, preserves the FedAvg
    sample-count prior, modulates it with learned utilities, and applies a
    bounded update-vector alignment multiplier before aggregation.

    CF-MARLOS-AVA-specific changes: utility features come from CVAE-DQN intrusion
    diagnostics, validation/support rewards train the mixer instead of directly
    weighting aggregation, and all utility/alignment factors are bounded so
    IID-like rounds stay close to FedAvg.
    """

    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .server_models import AsyncCritic, CentralizedAggregator

        self.cfg = cfg
        self.device = torch.device("cpu")
        self.latent_dim = int(cfg.model.latent_dim)
        self.max_agents = int(cfg.strategy.max_agents)
        self.scalar_dim = len(AUDIT_SCALAR_KEYS)
        self.client_feature_dim = self.latent_dim + self.scalar_dim
        self.state_dim = self.client_feature_dim * self.max_agents

        self.utility_threshold = float(cfg.strategy.utility_threshold)
        self.utility_temperature = max(float(cfg.strategy.utility_temperature), EPS)
        self.aggregation_lr = float(cfg.strategy.aggregation_lr)
        self.min_selected_clients = int(cfg.strategy.min_selected_clients)
        self.max_selected_fraction = float(cfg.strategy.max_selected_fraction)
        self.max_utility = float(cfg.strategy.max_utility)
        self.min_utility = float(OmegaConf.select(cfg, "strategy.min_utility", default=0.25))
        self.utility_strength = float(
            OmegaConf.select(cfg, "strategy.utility_strength", default=1.0)
        )
        self.critic_blend = float(OmegaConf.select(cfg, "strategy.critic_blend", default=0.15))
        self.alignment_strength = float(
            OmegaConf.select(cfg, "strategy.alignment_strength", default=0.50)
        )
        self.min_alignment_multiplier = float(
            OmegaConf.select(cfg, "strategy.min_alignment_multiplier", default=0.50)
        )
        self.max_alignment_multiplier = float(
            OmegaConf.select(cfg, "strategy.max_alignment_multiplier", default=2.00)
        )
        self.validation_reward_blend = float(
            OmegaConf.select(cfg, "strategy.validation_reward_blend", default=0.85)
        )
        self.validation_reward_ema_decay = float(
            OmegaConf.select(cfg, "strategy.validation_reward_ema_decay", default=0.80)
        )
        self.validation_reward_weights = self._read_weight_map(
            cfg,
            "strategy.team_reward_weights",
            {
                "closed_set_f1": 0.30,
                "balanced_accuracy": 0.20,
                "open_set_auroc": 0.20,
                "open_set_unknown_f1": 0.15,
                "open_set_rejection": 0.15,
            },
        )
        self.support_reward_weights = self._read_weight_map(
            cfg,
            "strategy.support_reward_weights",
            {
                "local_f1_macro": 0.35,
                "balanced_accuracy": 0.15,
                "td_stability": 0.20,
                "coverage_quality": 0.15,
                "generator_correct_frac": 0.10,
                "communication": 0.05,
            },
            fallback_path="strategy.system_utility_weights",
        )
        self.warmup_rounds = int(cfg.strategy.warmup_rounds)

        self.monitor_path = _resolve_path(cfg.strategy.monitor_path)
        self.monitor_path.parent.mkdir(parents=True, exist_ok=True)

        self._AsyncCriticClass = AsyncCritic
        self.critics: dict[str, torch.nn.Module] = {}
        self.aggregator = CentralizedAggregator(
            num_agents=self.max_agents,
            state_dim=self.state_dim,
            hidden_dim=int(cfg.strategy.mixer_hidden_dim),
        ).to(self.device)
        self.optimizer = optim.Adam(self.aggregator.parameters(), lr=float(cfg.strategy.fmrl_lr))

        self.is_training_phase = True
        self.saved_global_parameters: Parameters | None = None
        self.stage1_data_cache: dict[str, dict[str, Any]] = {}
        self.selected_clients_cache: list[fl.server.client_proxy.ClientProxy] = []
        self.last_phase_a_clients: list[fl.server.client_proxy.ClientProxy] = []
        self.utilities_cache: dict[str, float] = {}
        self.selection_records: list[dict[str, Any]] = []
        self.client_order: list[str] = []
        self.last_validation_metrics: dict[str, float] = {}
        self.last_validation_team_reward: float | None = None
        self.validation_team_reward_ema: float | None = None
        self.last_support_reward: float = 0.0
        self.last_team_reward_target: float = 0.0

        logger.info(
            "FMRL-AVA configured | max_agents=%d scalar_dim=%d threshold=%.4f aggregation_lr=%.3f",
            self.max_agents,
            self.scalar_dim,
            self.utility_threshold,
            self.aggregation_lr,
        )

    def _get_critic(self, cid: str) -> torch.nn.Module:
        if cid not in self.critics:
            critic = self._AsyncCriticClass(
                hidden_dim=int(self.cfg.strategy.critic_hidden_dim),
                latent_dim=self.latent_dim,
                scalar_dim=self.scalar_dim,
            ).to(self.device)
            self.optimizer.add_param_group({"params": critic.parameters()})
            self.critics[cid] = critic
        return self.critics[cid]

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        if self.is_training_phase:
            clients = client_manager.sample(
                num_clients=self.min_fit_clients,
                min_num_clients=self.min_fit_clients,
            )
            self.last_phase_a_clients = list(clients)
            self.saved_global_parameters = parameters
            fit_ins = FitIns(parameters, {"server_round": server_round, "phase": "train"})
            logger.info("FMRL-AVA round=%d phase=A sampled=%d", server_round, len(clients))
            return [(client, fit_ins) for client in clients]

        if not self.selected_clients_cache:
            self.selected_clients_cache = list(self.last_phase_a_clients)
            self.utilities_cache = {client.cid: 1.0 for client in self.selected_clients_cache}
            logger.warning(
                "FMRL-AVA round=%d phase=B had no selected clients; "
                "requesting uploads from the previous Phase A sample to avoid a stuck round.",
                server_round,
            )
        fit_ins = FitIns(parameters, {"server_round": server_round, "phase": "upload"})
        logger.info(
            "FMRL-AVA round=%d phase=B selected=%d",
            server_round,
            len(self.selected_clients_cache),
        )
        return [(client, fit_ins) for client in self.selected_clients_cache]

    def aggregate_fit(self, server_round: int, results, failures):
        if self.is_training_phase:
            self._phase_a_select_clients(server_round, results, failures)
            self.is_training_phase = False
            return self.saved_global_parameters, {}

        new_params, metrics = self._phase_b_aggregate(server_round, results, failures)
        if new_params is not None:
            save_global_model(new_params, server_round, self.cfg)
        self.is_training_phase = True
        return new_params, metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if metrics:
            validation_reward = validation_team_reward(metrics, weights=self.validation_reward_weights)
            if self.validation_team_reward_ema is None:
                self.validation_team_reward_ema = validation_reward
            else:
                decay = float(np.clip(self.validation_reward_ema_decay, 0.0, 1.0))
                self.validation_team_reward_ema = (
                    decay * self.validation_team_reward_ema
                    + (1.0 - decay) * validation_reward
                )
            self.last_validation_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float, np.floating))
            }
            self.last_validation_team_reward = float(validation_reward)
            logger.info(
                "FMRL-AVA validation reward | round=%d reward=%.4f ema=%.4f",
                server_round,
                validation_reward,
                self.validation_team_reward_ema,
            )
        self._write_monitor_event(
                {
                    "event": "validation_reward",
                    "server_round": server_round,
                    "logical_round": self._logical_round(server_round),
                    "validation_reward": float(validation_reward),
                    "validation_reward_ema": float(self.validation_team_reward_ema),
                    "validation_metrics": self.last_validation_metrics,
                }
            )
        return loss, metrics

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        if not self.is_training_phase:
            logger.info(
                "FMRL-AVA round=%d skipping evaluation after metadata-only phase.",
                server_round,
            )
            return []
        return super().configure_evaluate(server_round, parameters, client_manager)

    def _phase_a_select_clients(self, server_round: int, results, failures) -> None:
        self.stage1_data_cache = {}
        self.utilities_cache = {}
        self.selection_records = []
        self.client_order = []

        if failures:
            logger.warning("FMRL-AVA phase A failures: %d", len(failures))
            self._log_failures("FMRL-AVA phase A", failures)

        for client, fit_res in sorted(results, key=lambda item: item[0].cid):
            try:
                parsed = self._parse_client_metrics(client.cid, fit_res.metrics)
            except Exception as exc:
                logger.warning("Skipping client %s metadata: %s", client.cid, exc)
                continue

            critic = self._get_critic(client.cid)
            critic.eval()
            with torch.no_grad():
                critic_raw_utility = critic(parsed["h"], parsed["scalars"]).item()
            critic_score = critic_utility_score(
                critic_raw_utility,
                utility_temperature=self.utility_temperature,
            )
            audit_score = float(parsed["metrics"]["audit_score"])
            combined_score = combine_utility_score(
                audit_score=audit_score,
                critic_score=critic_score,
                critic_blend=self.critic_blend,
            )

            record = {
                "client": client,
                "cid": client.cid,
                "utility": 0.0,
                "audit_score": audit_score,
                "critic_raw_utility": float(critic_raw_utility),
                "critic_score": critic_score,
                "combined_score": combined_score,
                "selected": False,
                **parsed["metrics"],
            }
            self.stage1_data_cache[client.cid] = parsed
            self.selection_records.append(record)
            self.client_order.append(client.cid)

        self._calibrate_round_utilities(server_round)
        selected = self._select_records(server_round)
        positive_selected = [record for record in selected if float(record.get("utility", 0.0)) > 0.0]
        if (not selected or len(positive_selected) < self.min_selected_clients) and self.last_phase_a_clients:
            logger.warning(
                "FMRL-AVA phase A produced insufficient positive utilities; falling back to "
                "uniform utility for the sampled client set (%d clients).",
                len(self.last_phase_a_clients),
            )
            fallback_clients = list(self.last_phase_a_clients) if not selected else [record["client"] for record in selected]
            self.selected_clients_cache = fallback_clients
            self.utilities_cache = {client.cid: 1.0 for client in self.selected_clients_cache}
            for record in self.selection_records:
                if record["cid"] in self.utilities_cache:
                    record["utility"] = 1.0
        else:
            self.selected_clients_cache = [record["client"] for record in selected]
            self.utilities_cache = {record["cid"]: record["utility"] for record in selected}

        self._log_selection_table(server_round)
        self._write_monitor_event(
            {
                "event": "phase_a_selection",
                "server_round": server_round,
                "logical_round": self._logical_round(server_round),
                "available_clients": len(self.selection_records),
                "selected_clients": len(selected),
                "records": [
                    {k: v for k, v in record.items() if k != "client"}
                    for record in self.selection_records
                ],
            }
        )

    def _phase_b_aggregate(self, server_round: int, results, failures):
        if failures:
            logger.warning("FMRL-AVA phase B failures: %d", len(failures))
            self._log_failures("FMRL-AVA phase B", failures)
        if not results or self.saved_global_parameters is None:
            return self.saved_global_parameters, {"fmrl_ava_selected_clients": 0.0}

        base_weights = parameters_to_ndarrays(self.saved_global_parameters)
        upload_records = []
        pending_uploads: list[dict[str, Any]] = []
        total_base_aggregation_weight = 0.0
        total_aggregation_weight = 0.0
        weighted_deltas = [np.zeros_like(layer) for layer in base_weights]
        reference_delta = [np.zeros_like(layer) for layer in base_weights]

        for client, fit_res in sorted(results, key=lambda item: item[0].cid):
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            if len(client_weights) != len(base_weights):
                logger.warning("Client %s parameter count mismatch; skipping.", client.cid)
                continue

            utility = float(self.utilities_cache.get(client.cid, 1.0))
            if self._is_warmup(server_round):
                utility = 1.0
            utility = max(utility, EPS)
            num_examples = max(float(fit_res.num_examples), 1.0)
            base_aggregation_weight = num_examples * utility
            deltas = [
                client_layer - base_layer
                for client_layer, base_layer in zip(client_weights, base_weights, strict=True)
            ]
            for idx, delta in enumerate(deltas):
                reference_delta[idx] += delta * base_aggregation_weight

            delta_norm = self._parameter_delta_norm(client_weights, base_weights)
            pending_uploads.append(
                {
                    "cid": client.cid,
                    "utility": utility,
                    "base_aggregation_weight": base_aggregation_weight,
                    "delta_norm": delta_norm,
                    "num_examples": num_examples,
                    "recent_reward": float(fit_res.metrics.get("recent_reward", 0.0)),
                    "local_f1_macro": float(fit_res.metrics.get("local_f1_macro", 0.0)),
                    "local_balanced_accuracy": float(
                        fit_res.metrics.get(
                            "local_balanced_accuracy",
                            fit_res.metrics.get("balanced_accuracy", 0.0),
                        )
                    ),
                    "policy_accuracy": float(fit_res.metrics.get("policy_accuracy", 0.0)),
                    "td_error": float(fit_res.metrics.get("td_error", 0.0)),
                    "td_stability": float(
                        fit_res.metrics.get(
                            "td_stability",
                            1.0 / (1.0 + max(float(fit_res.metrics.get("td_error", 0.0)), 0.0)),
                        )
                    ),
                    "coverage_quality": float(
                        fit_res.metrics.get(
                            "coverage_quality",
                            0.5
                            * float(fit_res.metrics.get("class_entropy", 0.0))
                            + 0.5 * float(fit_res.metrics.get("label_coverage", 0.0)),
                        )
                    ),
                    "generator_correct_frac": float(
                        fit_res.metrics.get("generator_correct_frac", 0.0)
                    ),
                    "_deltas": deltas,
                }
            )
            total_base_aggregation_weight += base_aggregation_weight

        if not pending_uploads or total_base_aggregation_weight <= EPS:
            return self.saved_global_parameters, {"fmrl_ava_selected_clients": 0.0}

        reference_delta = [delta / total_base_aggregation_weight for delta in reference_delta]
        for record in pending_uploads:
            deltas = record.pop("_deltas")
            alignment_cosine = self._delta_cosine(deltas, reference_delta)
            alignment_multiplier = self._alignment_multiplier(alignment_cosine)
            aggregation_weight = float(record["base_aggregation_weight"]) * alignment_multiplier
            for idx, delta in enumerate(deltas):
                weighted_deltas[idx] += delta * aggregation_weight
            record["alignment_cosine"] = alignment_cosine
            record["alignment_multiplier"] = alignment_multiplier
            record["aggregation_weight"] = aggregation_weight
            upload_records.append(record)
            total_aggregation_weight += aggregation_weight

        if total_aggregation_weight <= EPS:
            return self.saved_global_parameters, {"fmrl_ava_selected_clients": 0.0}

        new_weights = [
            base_layer + self.aggregation_lr * (delta / total_aggregation_weight)
            for base_layer, delta in zip(base_weights, weighted_deltas, strict=True)
        ]
        selected_fraction = len(upload_records) / max(len(self.selection_records), 1)
        system_utility = self._compute_system_utility(upload_records, selected_fraction)
        train_metrics = self._train_server_models(system_utility)

        self._write_monitor_event(
            {
                "event": "phase_b_aggregation",
                "server_round": server_round,
                "logical_round": self._logical_round(server_round),
                "selected_fraction": selected_fraction,
                "system_utility": system_utility,
                "validation_team_reward": float(self.validation_team_reward_ema or 0.0),
                "validation_team_reward_raw": float(self.last_validation_team_reward or 0.0),
                "support_reward": float(self.last_support_reward),
                "total_base_aggregation_weight": total_base_aggregation_weight,
                "total_utility": total_aggregation_weight,
                "total_aggregation_weight": total_aggregation_weight,
                "uploads": upload_records,
                **train_metrics,
            }
        )
        logger.info(
            "FMRL-AVA aggregation | round=%d selected=%d/%d system_utility=%.4f validation=%.4f support=%.4f base_weight=%.4f aligned_weight=%.4f",
            server_round,
            len(upload_records),
            max(len(self.selection_records), 1),
            system_utility,
            float(self.validation_team_reward_ema or 0.0),
            float(self.last_support_reward),
            total_base_aggregation_weight,
            total_aggregation_weight,
        )

        metrics = {
            "fmrl_ava_selected_clients": float(len(upload_records)),
            "fmrl_ava_selected_fraction": float(selected_fraction),
            "fmrl_ava_system_utility": float(system_utility),
            "fmrl_ava_total_utility": float(total_aggregation_weight),
            "fmrl_ava_total_base_aggregation_weight": float(total_base_aggregation_weight),
            "fmrl_ava_total_aggregation_weight": float(total_aggregation_weight),
            **train_metrics,
        }
        return ndarrays_to_parameters(new_weights), metrics

    def _parse_client_metrics(self, _cid: str, metrics: dict[str, Scalar]) -> dict[str, Any]:
        hidden_raw = json.loads(str(metrics.get("hidden_info", "[]")))
        hidden = np.asarray(hidden_raw, dtype=np.float32).reshape(-1)
        if hidden.size < self.latent_dim:
            hidden = np.pad(hidden, (0, self.latent_dim - hidden.size))
        hidden = hidden[: self.latent_dim]

        recent_reward = self._float_metric(metrics, "recent_reward")
        history_reward = self._float_metric(metrics, "history_reward")
        steps_per_episode = max(float(self.cfg.training.steps_per_episode), 1.0)
        max_round_reward = steps_per_episode * max(
            float(self.cfg.training.local_episodes_per_round), 1.0
        )
        reward_norm = float(np.clip(0.5 + 0.5 * (recent_reward / steps_per_episode), 0.0, 1.0))
        history_norm = float(0.5 + 0.5 * np.tanh(history_reward / max(max_round_reward, 1.0)))

        audit_f1 = self._float_metric(
            metrics, "audit_f1", self._float_metric(metrics, "utility_loss")
        )
        local_f1 = self._float_metric(metrics, "local_f1_macro", audit_f1)
        local_acc = self._float_metric(
            metrics, "local_accuracy", self._float_metric(metrics, "policy_accuracy")
        )
        td_error = self._float_metric(
            metrics, "td_error", self._float_metric(metrics, "avg_td_loss")
        )
        kl_div = self._float_metric(metrics, "kl_div")
        total_steps = self._float_metric(metrics, "total_steps")
        generator_frac = self._float_metric(metrics, "generator_correct_frac", 0.5)

        coverage_quality = 0.5 * float(
            np.clip(self._float_metric(metrics, "class_entropy"), 0.0, 1.0)
        ) + 0.5 * float(np.clip(self._float_metric(metrics, "label_coverage"), 0.0, 1.0))
        scalar_values = {
            "reward_norm": reward_norm,
            "history_reward_norm": history_norm,
            "f1_macro": float(np.clip(local_f1, 0.0, 1.0)),
            "accuracy": float(np.clip(local_acc, 0.0, 1.0)),
            "td_stability": float(1.0 / (1.0 + max(td_error, 0.0))),
            "novelty": float(np.tanh(max(kl_div, 0.0))),
            "class_entropy": float(np.clip(self._float_metric(metrics, "class_entropy"), 0.0, 1.0)),
            "label_coverage": float(
                np.clip(self._float_metric(metrics, "label_coverage"), 0.0, 1.0)
            ),
            "generator_correct_frac": float(np.clip(generator_frac, 0.0, 1.0)),
            "steps_norm": float(np.clip(total_steps / max_round_reward, 0.0, 1.0)),
        }
        audit_score = float(
            (0.25 * scalar_values["f1_macro"])
            + (0.20 * scalar_values["accuracy"])
            + (0.20 * scalar_values["td_stability"])
            + (0.15 * coverage_quality)
            + (0.10 * scalar_values["reward_norm"])
            + (0.05 * scalar_values["history_reward_norm"])
            + (0.05 * scalar_values["generator_correct_frac"])
        )
        scalar_tensor = torch.tensor(
            [[scalar_values[key] for key in AUDIT_SCALAR_KEYS]],
            dtype=torch.float32,
            device=self.device,
        )
        hidden_tensor = torch.tensor([hidden], dtype=torch.float32, device=self.device)
        feature_tensor = torch.cat([hidden_tensor, scalar_tensor], dim=1)

        return {
            "h": hidden_tensor,
            "scalars": scalar_tensor,
            "feature": feature_tensor,
            "metrics": {
                "recent_reward": recent_reward,
                "history_reward": history_reward,
                "td_error": td_error,
                "kl_div": kl_div,
                "local_f1_macro": local_f1,
                "local_accuracy": local_acc,
                "class_entropy": scalar_values["class_entropy"],
                "label_coverage": scalar_values["label_coverage"],
                "local_num_examples": self._float_metric(metrics, "local_num_examples"),
                "total_steps": total_steps,
                "audit_score": audit_score,
                "td_stability": scalar_values["td_stability"],
                "coverage_quality": coverage_quality,
            },
        }

    def _select_records(self, server_round: int) -> list[dict[str, Any]]:
        return select_utility_records(
            self.selection_records,
            server_round=server_round,
            min_selected_clients=self.min_selected_clients,
            max_selected_fraction=self.max_selected_fraction,
            warmup_rounds=self.warmup_rounds,
        )

    def _calibrate_round_utilities(self, server_round: int) -> None:
        if not self.selection_records:
            return
        if self._is_warmup(server_round):
            for record in self.selection_records:
                record["utility"] = 1.0
            return

        round_mean_score = float(
            np.mean([record.get("combined_score", 0.0) for record in self.selection_records])
        )
        for record in self.selection_records:
            record["utility"] = centered_utility(
                score=float(record.get("combined_score", 0.0)),
                round_mean_score=round_mean_score,
                utility_strength=self.utility_strength,
                min_utility=self.min_utility,
                max_utility=self.max_utility,
                utility_threshold=self.utility_threshold,
            )

    def _train_server_models(self, system_utility: float) -> dict[str, float]:
        if not self.stage1_data_cache:
            return {}
        self.optimizer.zero_grad()
        self.aggregator.train()
        for critic in self.critics.values():
            critic.train()

        utilities, global_state = self._current_utility_tensor()
        prediction = self.aggregator(utilities, global_state)
        target = torch.tensor([[system_utility]], dtype=torch.float32, device=self.device)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.aggregator.parameters(), 1.0)
        for critic in self.critics.values():
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        self.optimizer.step()

        return {
            "fmrl_ava_mixer_loss": float(loss.item()),
            "fmrl_ava_predicted_system_utility": float(prediction.detach().item()),
            "fmrl_ava_team_reward_target": float(system_utility),
            "fmrl_ava_validation_reward": float(self.validation_team_reward_ema or 0.0),
            "fmrl_ava_support_reward": float(self.last_support_reward),
        }

    def _current_utility_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        utilities = []
        features = []
        for cid in self.client_order[: self.max_agents]:
            data = self.stage1_data_cache[cid]
            record = next((r for r in self.selection_records if r["cid"] == cid), None)
            utility_value = float(record["utility"]) if record is not None else 0.0
            utilities.append(torch.tensor([utility_value], device=self.device))
            features.append(data["feature"].view(-1))

        if not utilities:
            utilities.append(torch.zeros(1, device=self.device))
            features.append(torch.zeros(self.client_feature_dim, device=self.device))

        utility_tensor = torch.stack(utilities).view(1, -1)
        if utility_tensor.shape[1] < self.max_agents:
            utility_tensor = F.pad(utility_tensor, (0, self.max_agents - utility_tensor.shape[1]))

        global_state = torch.cat(features).view(1, -1)
        target_dim = self.state_dim
        if global_state.shape[1] < target_dim:
            global_state = F.pad(global_state, (0, target_dim - global_state.shape[1]))
        return utility_tensor, global_state[:, :target_dim]

    def _compute_support_reward(
        self, upload_records: list[dict[str, float]], selected_fraction: float
    ) -> float:
        if not upload_records:
            self.last_support_reward = 0.0
            return 0.0

        weights = np.asarray(
            [record.get("num_examples", 1.0) for record in upload_records], dtype=float
        )
        weights = np.maximum(weights, EPS)

        def avg(key: str) -> float:
            values = np.asarray([record.get(key, 0.0) for record in upload_records], dtype=float)
            return float(np.average(values, weights=weights))

        components = {
            "local_f1_macro": float(np.clip(avg("local_f1_macro"), 0.0, 1.0)),
            "balanced_accuracy": float(np.clip(avg("local_balanced_accuracy"), 0.0, 1.0)),
            "td_stability": float(np.clip(avg("td_stability"), 0.0, 1.0)),
            "coverage_quality": float(np.clip(avg("coverage_quality"), 0.0, 1.0)),
            "generator_correct_frac": float(np.clip(avg("generator_correct_frac"), 0.0, 1.0)),
            "communication": float(1.0 - np.clip(selected_fraction, 0.0, 1.0)),
        }
        denom = sum(
            max(float(self.support_reward_weights.get(name, 0.0)), 0.0) for name in components
        ) or 1.0
        support_reward = sum(
            max(float(self.support_reward_weights.get(name, 0.0)), 0.0) * components[name]
            for name in components
        ) / denom
        self.last_support_reward = float(np.clip(support_reward, 0.0, 1.0))
        return self.last_support_reward

    def _compute_system_utility(
        self, upload_records: list[dict[str, float]], selected_fraction: float
    ) -> float:
        support_reward = self._compute_support_reward(upload_records, selected_fraction)
        validation_reward = self.validation_team_reward_ema
        if validation_reward is None:
            self.last_team_reward_target = support_reward
            return support_reward

        blend = float(np.clip(self.validation_reward_blend, 0.0, 1.0))
        system_utility = float(
            np.clip((blend * validation_reward) + ((1.0 - blend) * support_reward), 0.0, 1.0)
        )
        self.last_team_reward_target = system_utility
        return system_utility

    def _log_selection_table(self, server_round: int) -> None:
        logger.info("-" * 112)
        logger.info(
            "%-8s | %-5s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s",
            "Client",
            "Sel",
            "Utility",
            "Reward",
            "F1",
            "Acc",
            "TD",
            "KL",
            "Entropy",
        )
        logger.info("-" * 112)
        for record in self.selection_records:
            logger.info(
                "%-8s | %-5s | %-8.4f | %-8.2f | %-8.3f | %-8.3f | %-8.4f | %-8.4f | %-8.3f",
                record["cid"][:8],
                "YES" if record["selected"] else "NO",
                record["utility"],
                record["recent_reward"],
                record["local_f1_macro"],
                record["local_accuracy"],
                record["td_error"],
                record["kl_div"],
                record["class_entropy"],
            )
        logger.info("FMRL-AVA selection round=%d", server_round)
        logger.info("-" * 112)

    def _write_monitor_event(self, event: dict[str, Any]) -> None:
        try:
            with open(self.monitor_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write FMRL-AVA monitor event: %s", exc)

    @staticmethod
    def _log_failures(context: str, failures) -> None:
        for idx, failure in enumerate(failures, start=1):
            if isinstance(failure, BaseException):
                logger.warning("%s failure %d: %r", context, idx, failure)
                continue
            if isinstance(failure, tuple) and len(failure) == 2:
                client, result = failure
                logger.warning(
                    "%s failure %d | client=%s | result=%r",
                    context,
                    idx,
                    getattr(client, "cid", "?"),
                    result,
                )
                continue
            logger.warning("%s failure %d: %r", context, idx, failure)

    @staticmethod
    def _parameter_delta_norm(weights: list[np.ndarray], base_weights: list[np.ndarray]) -> float:
        total = 0.0
        for layer, base_layer in zip(weights, base_weights, strict=True):
            diff = layer - base_layer
            total += float(np.sum(diff * diff))
        return float(np.sqrt(total))

    @staticmethod
    def _float_metric(metrics: dict[str, Scalar], key: str, default: float = 0.0) -> float:
        value = metrics.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _read_weight_map(
        cfg: DictConfig,
        path: str,
        defaults: dict[str, float],
        *,
        fallback_path: str | None = None,
    ) -> dict[str, float]:
        resolved: dict[str, float] = {}
        for key, default in defaults.items():
            value = OmegaConf.select(cfg, f"{path}.{key}", default=None)
            if value is None and fallback_path is not None:
                value = OmegaConf.select(cfg, f"{fallback_path}.{key}", default=None)
            resolved[key] = float(default if value is None else value)
        return resolved

    def _alignment_multiplier(self, cosine: float) -> float:
        return alignment_multiplier(
            cosine,
            alignment_strength=self.alignment_strength,
            min_multiplier=self.min_alignment_multiplier,
            max_multiplier=self.max_alignment_multiplier,
        )

    @staticmethod
    def _delta_cosine(delta_a: list[np.ndarray], delta_b: list[np.ndarray]) -> float:
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for layer_a, layer_b in zip(delta_a, delta_b, strict=True):
            a64 = layer_a.astype(np.float64, copy=False).reshape(-1)
            b64 = layer_b.astype(np.float64, copy=False).reshape(-1)
            dot += float(np.dot(a64, b64))
            norm_a += float(np.dot(a64, a64))
            norm_b += float(np.dot(b64, b64))
        denom = np.sqrt(max(norm_a, 0.0)) * np.sqrt(max(norm_b, 0.0))
        if denom <= EPS:
            return 0.0
        return float(np.clip(dot / denom, -1.0, 1.0))

    @staticmethod
    def _logical_round(server_round: int) -> int:
        return int((server_round + 1) // 2)

    def _is_warmup(self, server_round: int) -> bool:
        return self._logical_round(server_round) <= self.warmup_rounds


def aggregate_fit_metrics(
    fit_metrics: list[tuple[int, dict[str, Scalar]]],
) -> dict[str, float]:
    if not fit_metrics:
        return {}
    total_examples = sum(num for num, _ in fit_metrics)
    if total_examples == 0:
        return {}
    aggregated = {}
    for num, metrics in fit_metrics:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                aggregated[k] = aggregated.get(k, 0.0) + (float(v) * num)
    return {k: v / total_examples for k, v in aggregated.items()}


def aggregate_evaluate_metrics(
    eval_metrics: list[tuple[int, dict[str, Scalar]]],
) -> dict[str, float]:
    if not eval_metrics:
        return {}
    total_examples = sum(num for num, _ in eval_metrics)
    if total_examples == 0:
        return {}
    aggregated = {}
    for num, metrics in eval_metrics:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                aggregated[k] = aggregated.get(k, 0.0) + (float(v) * num)
    return {k: v / total_examples for k, v in aggregated.items()}


def get_strategy(cfg: DictConfig) -> Strategy:
    strat_name = str(cfg.strategy.name).lower()
    device = torch.device("cpu")
    logger.info(
        "Server strategy setup | strategy=%s | server_device=%s | logical_rounds=%d | flower_rounds=%d",
        strat_name,
        device,
        int(cfg.server.num_rounds),
        get_effective_num_rounds(cfg),
    )

    central_evaluate_fn = make_central_evaluate_fn(cfg, device)

    args = dict(
        fraction_fit=cfg.server.fraction_fit,
        fraction_evaluate=cfg.server.fraction_evaluate,
        min_fit_clients=cfg.server.min_fit_clients,
        min_evaluate_clients=cfg.server.min_evaluate_clients,
        min_available_clients=cfg.server.min_available_clients,
        initial_parameters=_initial_parameters_from_checkpoint(cfg, device),
        on_fit_config_fn=fit_config_fn,
        evaluate_fn=central_evaluate_fn,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )

    if strat_name == "fmrl_ava":
        logger.info(
            "--- Strategy: FMRL-AVA (Adaptive Vector-Aligned Aggregation) with Model Saving ---"
        )
        return FMRLAdaptiveVectorAlignedAggregationStrategy(cfg=cfg, **args)

    if strat_name == "dkd_fedos":
        logger.info("--- Strategy: DKD-FedOS (dynamic KD student aggregation) with Model Saving ---")
        return DKDFedOSStrategy(
            cfg=cfg,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            **args,
        )

    if strat_name == "fedgpa":
        logger.info("--- Strategy: FedGPA (prototype-personalized aggregation) with Model Saving ---")
        return FedGPAStrategy(
            cfg=cfg,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            **args,
        )

    if strat_name == "fedprox":
        proximal_mu = float(cfg.server.proximal_mu)
        logger.info("--- Strategy: FedProx (mu=%.2f) with Model Saving ---", proximal_mu)
        return SaveModelFedProx(
            cfg=cfg,
            proximal_mu=proximal_mu,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            **args,
        )

    logger.info("--- Strategy: FedAvg with Model Saving ---")
    return SaveModelFedAvg(cfg=cfg, fit_metrics_aggregation_fn=aggregate_fit_metrics, **args)


def get_effective_num_rounds(cfg: DictConfig) -> int:
    configured_rounds = int(cfg.server.num_rounds)
    if str(cfg.strategy.name).lower() == "fmrl_ava":
        return configured_rounds * 2
    return configured_rounds


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
