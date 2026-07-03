import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from flwr.common import (
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
from src.checkpointing.checkpoints import (
    CheckpointState,
    load_agent_checkpoint,
    metric_improved,
    select_checkpoint_metric,
    write_checkpoint_metadata,
)
from src.federated.class_aware import class_aware_aggregation_records
from src.federated.selection_utils import (
    alignment_multiplier,
    centered_utility,
    combine_utility_score,
    critic_utility_score,
    select_utility_records,
    validation_team_reward,
)
from src.models.cvae_dqn import OpenSetQChainModelFactory
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
    parameters: Parameters | list[np.ndarray], round_num: int, cfg: DictConfig
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
            "metrics": {"federated/round": float(round_num)},
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
        state = CheckpointState(
            epoch=round_num,
            global_step=round_num,
            metrics=checkpoint["metrics"],
            best_metric=None,
        )
        torch.save(checkpoint, round_path)
        torch.save(checkpoint, model_dir / "global_model_latest.pt")
        torch.save(checkpoint, _resolve_path(cfg.checkpointing.latest_checkpoint_path))
        torch.save(checkpoint, _resolve_path(cfg.checkpointing.last_model_path))
        write_checkpoint_metadata(
            cfg,
            _resolve_path(cfg.checkpointing.last_model_path),
            state,
        )

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


def _maybe_save_best_validation_checkpoint(
    cfg: DictConfig,
    server_round: int,
    metrics: dict[str, Scalar],
    best_metric: float | None,
) -> float | None:
    """Promote the last federated checkpoint to best only from validation metrics."""
    selected_metric = select_checkpoint_metric(
        dict(metrics),
        monitor_metric=str(cfg.checkpointing.monitor_metric),
    )
    if selected_metric is None:
        return best_metric
    metric_name, metric_value = selected_metric
    if not metric_improved(
        metric_value,
        best_metric,
        mode=str(cfg.checkpointing.monitor_mode),
    ):
        return best_metric

    source_path = _resolve_path(cfg.checkpointing.last_model_path)
    if not source_path.exists():
        source_path = _resolve_path(cfg.checkpointing.latest_checkpoint_path)
    if not source_path.exists():
        logger.warning("Cannot save best checkpoint; no latest/last checkpoint exists yet.")
        return best_metric

    best_path = _resolve_path(cfg.checkpointing.best_model_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    state = CheckpointState(
        epoch=server_round,
        global_step=server_round,
        metrics=dict(metrics),
        best_metric=metric_value,
    )
    try:
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        checkpoint["metrics"] = dict(metrics)
        checkpoint["best_metric"] = metric_value
        checkpoint["selected_metric_name"] = metric_name
        checkpoint["selected_metric_value"] = metric_value
        checkpoint["epoch"] = server_round
        checkpoint["round"] = server_round
        checkpoint["global_step"] = server_round
        torch.save(checkpoint, best_path)
    except Exception:
        logger.exception("Failed to rewrite best checkpoint payload from %s.", source_path)
        return best_metric

    write_checkpoint_metadata(
        cfg,
        best_path,
        state,
        selected_metric_name=metric_name,
        selected_metric_value=metric_value,
        is_best=True,
    )
    logger.info(
        "Saved best federated checkpoint | round=%d | %s=%.6f",
        server_round,
        metric_name,
        metric_value,
    )
    return metric_value


def _positive_evaluate_results(server_round: int, results):
    if not results:
        return results

    positive_results = []
    zero_example_clients = []
    for client, evaluate_res in results:
        num_examples = int(getattr(evaluate_res, "num_examples", 0) or 0)
        if num_examples > 0:
            positive_results.append((client, evaluate_res))
        else:
            zero_example_clients.append(str(getattr(client, "cid", "?")))

    if zero_example_clients:
        shown_clients = ", ".join(zero_example_clients[:10])
        if len(zero_example_clients) > 10:
            shown_clients = f"{shown_clients}, ..."
        logger.warning(
            "Ignoring %d zero-example evaluation result(s) in round %d | clients=%s",
            len(zero_example_clients),
            server_round,
            shown_clients,
        )
    if not positive_results:
        logger.warning(
            "Skipping round %d evaluation aggregation because all %d result(s) had zero examples.",
            server_round,
            len(results),
        )
    return positive_results


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
        self.best_validation_metric: float | None = None

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
            save_global_model(aggregated_parameters, server_round, self.cfg)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        filtered_results = _positive_evaluate_results(server_round, results)
        if results and not filtered_results:
            return None, {}
        loss, metrics = super().aggregate_evaluate(server_round, filtered_results, failures)
        if metrics:
            self.best_validation_metric = _maybe_save_best_validation_checkpoint(
                self.cfg,
                server_round,
                metrics,
                self.best_validation_metric,
            )
        return loss, metrics


class SaveModelFedProx(FedProx):
    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.best_validation_metric: float | None = None

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
            save_global_model(aggregated_parameters, server_round, self.cfg)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        filtered_results = _positive_evaluate_results(server_round, results)
        if results and not filtered_results:
            return None, {}
        loss, metrics = super().aggregate_evaluate(server_round, filtered_results, failures)
        if metrics:
            self.best_validation_metric = _maybe_save_best_validation_checkpoint(
                self.cfg,
                server_round,
                metrics,
                self.best_validation_metric,
            )
        return loss, metrics


class FedMADEClassAwareAggregationStrategy(FedAvg):
    """
    FedMADE-inspired dynamic aggregation for intrusion-detection non-IID data.

    This is a server-side method: clients train normally, then the server keeps
    FedAvg's sample-count prior and changes aggregation weights using local
    class histograms, local quality metrics, and traffic-profile cluster balance.
    It avoids the two-phase FMRL-AVA path and is therefore directly comparable
    to FedAvg/FedProx in communication rounds.
    """

    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.best_validation_metric: float | None = None
        self.num_classes = int(cfg.model.num_actions)
        self.rare_class_strength = float(
            OmegaConf.select(cfg, "strategy.rare_class_strength", default=1.0)
        )
        self.quality_weight_blend = float(
            OmegaConf.select(cfg, "strategy.quality_weight_blend", default=0.35)
        )
        self.cluster_balance_strength = float(
            OmegaConf.select(cfg, "strategy.cluster_balance_strength", default=0.50)
        )
        self.min_multiplier = float(
            OmegaConf.select(cfg, "strategy.min_weight_multiplier", default=0.25)
        )
        self.max_multiplier = float(
            OmegaConf.select(cfg, "strategy.max_weight_multiplier", default=3.00)
        )
        self.label_smoothing = float(
            OmegaConf.select(cfg, "strategy.label_smoothing", default=1.0)
        )
        default_monitor_path = f"{cfg.tracking.run_dir}/fedmade_monitoring.jsonl"
        self.monitor_path = _resolve_path(
            OmegaConf.select(cfg, "strategy.monitor_path", default=default_monitor_path)
        )
        self.monitor_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "FedMADE-style class-aware aggregation configured | classes=%d rare_strength=%.3f "
            "quality_blend=%.3f cluster_strength=%.3f multiplier=[%.3f, %.3f]",
            self.num_classes,
            self.rare_class_strength,
            self.quality_weight_blend,
            self.cluster_balance_strength,
            self.min_multiplier,
            self.max_multiplier,
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if failures:
            self._log_failures("FedMADE aggregate", failures)
        if not results:
            return None, {"fedmade_clients": 0.0}

        payloads: list[tuple[dict[str, Any], list[np.ndarray]]] = []
        for client, fit_res in sorted(results, key=lambda item: item[0].cid):
            weights = parameters_to_ndarrays(fit_res.parameters)
            if not weights:
                logger.warning("Client %s returned empty parameters; skipping.", client.cid)
                continue

            metrics = dict(fit_res.metrics)
            label_histogram = metrics.get(
                "full_label_histogram",
                metrics.get("label_histogram", "{}"),
            )
            per_class_recall = metrics.get(
                "local_per_class_recall",
                metrics.get("per_class_recall", None),
            )
            payloads.append(
                (
                    {
                        "cid": client.cid,
                        "num_examples": max(float(fit_res.num_examples), 1.0),
                        "quality": self._client_quality(metrics),
                        "label_histogram": label_histogram,
                        "per_class_recall": per_class_recall,
                        "class_entropy": self._float_metric(metrics, "class_entropy"),
                        "label_coverage": self._float_metric(metrics, "label_coverage"),
                    },
                    weights,
                )
            )

        if not payloads:
            return None, {"fedmade_clients": 0.0}

        aggregation_records = class_aware_aggregation_records(
            [record for record, _weights in payloads],
            num_classes=self.num_classes,
            rare_class_strength=self.rare_class_strength,
            quality_weight_blend=self.quality_weight_blend,
            cluster_balance_strength=self.cluster_balance_strength,
            min_multiplier=self.min_multiplier,
            max_multiplier=self.max_multiplier,
            label_smoothing=self.label_smoothing,
        )

        first_weights = payloads[0][1]
        weighted_layers = [np.zeros_like(layer) for layer in first_weights]
        total_weight = 0.0
        used_records: list[dict[str, Any]] = []

        for record, (_raw_record, weights) in zip(aggregation_records, payloads, strict=True):
            if len(weights) != len(first_weights):
                logger.warning("Client %s parameter count mismatch; skipping.", record["cid"])
                continue
            aggregation_weight = max(float(record["aggregation_weight"]), EPS)
            for idx, layer in enumerate(weights):
                weighted_layers[idx] += layer * aggregation_weight
            total_weight += aggregation_weight
            used_records.append(record)

        if total_weight <= EPS or not used_records:
            logger.warning("FedMADE weights were empty; falling back to FedAvg aggregation.")
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(
                server_round, results, failures
            )
            if aggregated_parameters is not None:
                save_global_model(aggregated_parameters, server_round, self.cfg)
            return aggregated_parameters, aggregated_metrics

        new_weights = [layer / total_weight for layer in weighted_layers]
        aggregated_parameters = ndarrays_to_parameters(new_weights)
        save_global_model(aggregated_parameters, server_round, self.cfg)

        selected_fraction = len(used_records) / max(len(results), 1)
        class_multiplier_mean = float(
            np.mean([record["class_multiplier"] for record in used_records])
        )
        quality_multiplier_mean = float(
            np.mean([record["quality_multiplier"] for record in used_records])
        )
        cluster_multiplier_mean = float(
            np.mean([record["cluster_multiplier"] for record in used_records])
        )
        max_weight_share = float(
            max(record["aggregation_weight"] for record in used_records) / total_weight
        )

        self._write_monitor_event(
            {
                "event": "fedmade_aggregation",
                "server_round": server_round,
                "selected_fraction": selected_fraction,
                "total_aggregation_weight": total_weight,
                "records": [self._serializable_record(record) for record in used_records],
            }
        )

        logger.info(
            "FedMADE aggregation | round=%d clients=%d total_weight=%.4f "
            "class_mult=%.4f quality_mult=%.4f cluster_mult=%.4f max_share=%.4f",
            server_round,
            len(used_records),
            total_weight,
            class_multiplier_mean,
            quality_multiplier_mean,
            cluster_multiplier_mean,
            max_weight_share,
        )

        return aggregated_parameters, {
            "fedmade_clients": float(len(used_records)),
            "fedmade_selected_fraction": float(selected_fraction),
            "fedmade_total_aggregation_weight": float(total_weight),
            "fedmade_class_multiplier_mean": class_multiplier_mean,
            "fedmade_quality_multiplier_mean": quality_multiplier_mean,
            "fedmade_cluster_multiplier_mean": cluster_multiplier_mean,
            "fedmade_max_weight_share": max_weight_share,
        }

    def aggregate_evaluate(self, server_round: int, results, failures):
        filtered_results = _positive_evaluate_results(server_round, results)
        if results and not filtered_results:
            return None, {}
        loss, metrics = super().aggregate_evaluate(server_round, filtered_results, failures)
        if metrics:
            self.best_validation_metric = _maybe_save_best_validation_checkpoint(
                self.cfg,
                server_round,
                metrics,
                self.best_validation_metric,
            )
        return loss, metrics

    @staticmethod
    def _float_metric(metrics: dict[str, Scalar], key: str, default: float = 0.0) -> float:
        value = metrics.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _client_quality(self, metrics: dict[str, Scalar]) -> float:
        for key in (
            "local_f1_macro",
            "f1_macro",
            "local_balanced_accuracy",
            "balanced_accuracy",
            "policy_accuracy",
            "accuracy",
        ):
            if key in metrics:
                return float(np.clip(self._float_metric(metrics, key), 0.0, 1.0))
        return 1.0

    def _write_monitor_event(self, event: dict[str, Any]) -> None:
        try:
            with open(self.monitor_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write FedMADE monitor event: %s", exc)

    @staticmethod
    def _serializable_record(record: dict[str, Any]) -> dict[str, Any]:
        serializable: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            else:
                serializable[key] = value
        return serializable

    @staticmethod
    def _log_failures(context: str, failures) -> None:
        for idx, failure in enumerate(failures, start=1):
            if isinstance(failure, BaseException):
                logger.warning("%s failure %d: %r", context, idx, failure)
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
        self.critic_activation_round = int(
            OmegaConf.select(cfg, "strategy.critic_activation_round", default=40)
        )
        self.critic_active_blend = float(
            OmegaConf.select(cfg, "strategy.critic_active_blend", default=0.05)
        )
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
        self.profile_balance_strength = float(
            OmegaConf.select(cfg, "strategy.profile_balance_strength", default=0.0)
        )
        self.profile_quality_blend = float(
            OmegaConf.select(cfg, "strategy.profile_quality_blend", default=0.0)
        )
        self.profile_cluster_strength = float(
            OmegaConf.select(cfg, "strategy.profile_cluster_strength", default=0.0)
        )
        self.profile_min_multiplier = float(
            OmegaConf.select(cfg, "strategy.profile_min_multiplier", default=0.25)
        )
        self.profile_max_multiplier = float(
            OmegaConf.select(cfg, "strategy.profile_max_multiplier", default=3.0)
        )
        self.profile_label_smoothing = float(
            OmegaConf.select(cfg, "strategy.profile_label_smoothing", default=1.0)
        )
        self.drift_penalty_strength = float(
            OmegaConf.select(cfg, "strategy.drift_penalty_strength", default=0.0)
        )
        self.drift_min_multiplier = float(
            OmegaConf.select(cfg, "strategy.drift_min_multiplier", default=0.50)
        )
        self.server_optimizer_name = str(
            OmegaConf.select(cfg, "strategy.server_optimizer", default="none")
        ).lower()
        self.server_beta1 = float(OmegaConf.select(cfg, "strategy.server_beta1", default=0.0))
        self.server_beta2 = float(OmegaConf.select(cfg, "strategy.server_beta2", default=0.99))
        self.server_tau = float(OmegaConf.select(cfg, "strategy.server_tau", default=1e-3))
        self.server_momentum: list[np.ndarray] | None = None
        self.server_second_moment: list[np.ndarray] | None = None
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
        self.best_validation_metric: float | None = None
        self.last_support_reward: float = 0.0
        self.last_team_reward_target: float = 0.0

        logger.info(
            "FMRL-AVA configured | max_agents=%d scalar_dim=%d threshold=%.4f "
            "aggregation_lr=%.3f profile_strength=%.3f server_optimizer=%s",
            self.max_agents,
            self.scalar_dim,
            self.utility_threshold,
            self.aggregation_lr,
            self.profile_balance_strength,
            self.server_optimizer_name,
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
        filtered_results = _positive_evaluate_results(server_round, results)
        if results and not filtered_results:
            return None, {}
        loss, metrics = super().aggregate_evaluate(server_round, filtered_results, failures)
        if metrics:
            self.best_validation_metric = _maybe_save_best_validation_checkpoint(
                self.cfg,
                server_round,
                metrics,
                self.best_validation_metric,
            )
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
            active_blend = 0.0
            if self._logical_round(server_round) >= self.critic_activation_round:
                active_blend = min(float(self.critic_blend), float(self.critic_active_blend))
            combined_score = combine_utility_score(
                audit_score=audit_score,
                critic_score=critic_score,
                critic_blend=active_blend,
            )

            record = {
                "client": client,
                "cid": client.cid,
                "utility": 0.0,
                "audit_score": audit_score,
                "critic_raw_utility": float(critic_raw_utility),
                "critic_score": critic_score,
                "combined_score": combined_score,
                "critic_blend_active": float(active_blend),
                "selected": False,
                "label_histogram": fit_res.metrics.get(
                    "full_label_histogram",
                    fit_res.metrics.get("label_histogram", "{}"),
                ),
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

            delta_norm = self._parameter_delta_norm(client_weights, base_weights)
            pending_uploads.append(
                {
                    "cid": client.cid,
                    "utility": utility,
                    "utility_weighted_examples": base_aggregation_weight,
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
                    "quality": self._client_quality(fit_res.metrics),
                    "label_histogram": fit_res.metrics.get(
                        "full_label_histogram",
                        fit_res.metrics.get("label_histogram", "{}"),
                    ),
                    "per_class_recall": fit_res.metrics.get(
                        "local_per_class_recall",
                        fit_res.metrics.get("per_class_recall", None),
                    ),
                    "_deltas": deltas,
                }
            )

        if not pending_uploads:
            return self.saved_global_parameters, {"fmrl_ava_selected_clients": 0.0}

        reference_weight = 0.0
        for record in pending_uploads:
            weight = max(float(record["num_examples"]), 1.0)
            for idx, delta in enumerate(record["_deltas"]):
                reference_delta[idx] += delta * weight
            reference_weight += weight
        reference_delta = [delta / max(reference_weight, EPS) for delta in reference_delta]

        self._apply_profile_multipliers(pending_uploads)
        self._apply_drift_multipliers(pending_uploads)
        for record in pending_uploads:
            base_aggregation_weight = (
                float(record["utility_weighted_examples"])
                * float(record.get("profile_multiplier", 1.0))
                * float(record.get("drift_multiplier", 1.0))
            )
            record["base_aggregation_weight"] = max(base_aggregation_weight, EPS)
            total_base_aggregation_weight += record["base_aggregation_weight"]

        if total_base_aggregation_weight <= EPS:
            return self.saved_global_parameters, {"fmrl_ava_selected_clients": 0.0}

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

        normalized_delta = [delta / total_aggregation_weight for delta in weighted_deltas]
        new_weights = [
            base_layer + delta
            for base_layer, delta in zip(
                base_weights,
                self._server_optimized_delta(normalized_delta),
                strict=True,
            )
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

    def _parse_hist(self, value: Any, num_classes: int) -> np.ndarray:
        try:
            raw = json.loads(value) if isinstance(value, str) else value
        except Exception:
            raw = {}
        hist = np.zeros(num_classes, dtype=np.float64)
        if isinstance(raw, dict):
            for key, item in raw.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < num_classes:
                    hist[idx] = float(item)
        return hist

    def _coverage_safe_select(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            return []

        n = len(records)
        num_classes = int(self.cfg.model.num_actions)
        min_clients = min(
            max(int(self.min_selected_clients), int(np.ceil(0.9 * n))),
            n,
        )

        selected = sorted(
            records,
            key=lambda r: float(r.get("utility", 1.0)),
            reverse=True,
        )

        while len(selected) > min_clients:
            candidate = min(
                selected,
                key=lambda r: (
                    float(r.get("utility", 1.0)),
                    -float(r.get("td_error", 0.0)),
                ),
            )

            remaining = [r for r in selected if r is not candidate]
            total_hist = np.zeros(num_classes, dtype=np.float64)

            for record in remaining:
                total_hist += self._parse_hist(
                    record.get("label_histogram", record.get("full_label_histogram", "{}")),
                    num_classes,
                )

            if np.any(total_hist <= 0):
                break

            selected = remaining

        selected_ids = {str(record["cid"]) for record in selected}
        for record in records:
            record["selected"] = str(record["cid"]) in selected_ids

        return selected

    def _select_records(self, server_round: int) -> list[dict[str, Any]]:
        if self._is_warmup(server_round):
            for record in self.selection_records:
                record["selected"] = True
            return list(self.selection_records)

        return self._coverage_safe_select(self.selection_records)

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
        mixer_loss = F.mse_loss(prediction, target)

        critic_losses = []
        baseline = float(self.last_team_reward_target or system_utility)
        advantage = float(system_utility - baseline)

        for record in self.selection_records:
            cid = str(record["cid"])
            if cid not in self.stage1_data_cache:
                continue

            data = self.stage1_data_cache[cid]
            critic = self._get_critic(cid)

            pred = torch.sigmoid(
                critic(data["h"], data["scalars"]) / self.utility_temperature
            )

            selected_bonus = 1.0 if bool(record.get("selected", False)) else 0.0
            base = float(record.get("audit_score", record.get("combined_score", 0.5)))

            target_value = float(
                np.clip(
                    base + 0.20 * advantage * selected_bonus,
                    0.0,
                    1.0,
                )
            )

            critic_target = torch.tensor(
                [[target_value]],
                dtype=torch.float32,
                device=self.device,
            )

            critic_losses.append(F.mse_loss(pred, critic_target))

        critic_loss = (
            torch.stack(critic_losses).mean()
            if critic_losses
            else torch.zeros((), dtype=torch.float32, device=self.device)
        )

        loss = mixer_loss + 0.25 * critic_loss
        loss.backward()

        params = list(self.aggregator.parameters())
        for critic in self.critics.values():
            params.extend(list(critic.parameters()))

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        self.optimizer.step()
        self.last_team_reward_target = float(system_utility)

        return {
            "fmrl_ava_mixer_loss": float(mixer_loss.item()),
            "fmrl_ava_critic_loss": float(critic_loss.item()),
            "fmrl_ava_total_server_loss": float(loss.item()),
            "fmrl_ava_predicted_system_utility": float(prediction.detach().item()),
            "fmrl_ava_team_reward_target": float(system_utility),
            "fmrl_ava_validation_reward": float(self.validation_team_reward_ema or 0.0),
            "fmrl_ava_support_reward": float(self.last_support_reward),
            "fmrl_ava_server_advantage": float(advantage),
        }

    def _current_utility_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        utilities = []
        features = []

        for cid in self.client_order[: self.max_agents]:
            data = self.stage1_data_cache[cid]
            critic = self._get_critic(cid)

            raw_utility = critic(data["h"], data["scalars"])
            utility = torch.sigmoid(raw_utility / self.utility_temperature)

            utilities.append(utility.view(1))
            features.append(data["feature"].view(-1))

        if not utilities:
            utilities.append(torch.zeros(1, device=self.device))
            features.append(torch.zeros(self.client_feature_dim, device=self.device))

        utility_tensor = torch.stack(utilities).view(1, -1)

        if utility_tensor.shape[1] < self.max_agents:
            utility_tensor = F.pad(
                utility_tensor,
                (0, self.max_agents - utility_tensor.shape[1]),
            )

        global_state = torch.cat(features).view(1, -1)

        if global_state.shape[1] < self.state_dim:
            global_state = F.pad(
                global_state,
                (0, self.state_dim - global_state.shape[1]),
            )

        return utility_tensor, global_state[:, : self.state_dim]

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
            return support_reward

        blend = float(np.clip(self.validation_reward_blend, 0.0, 1.0))
        system_utility = float(
            np.clip((blend * validation_reward) + ((1.0 - blend) * support_reward), 0.0, 1.0)
        )
        return system_utility

    def _apply_profile_multipliers(self, records: list[dict[str, Any]]) -> None:
        if (
            self.profile_balance_strength <= EPS
            and self.profile_quality_blend <= EPS
            and self.profile_cluster_strength <= EPS
        ):
            for record in records:
                record["profile_multiplier"] = 1.0
                record["class_multiplier"] = 1.0
                record["quality_multiplier"] = 1.0
                record["cluster_multiplier"] = 1.0
            return

        weighted_records = class_aware_aggregation_records(
            records,
            num_classes=int(self.cfg.model.num_actions),
            rare_class_strength=self.profile_balance_strength,
            quality_weight_blend=self.profile_quality_blend,
            cluster_balance_strength=self.profile_cluster_strength,
            min_multiplier=self.profile_min_multiplier,
            max_multiplier=self.profile_max_multiplier,
            label_smoothing=self.profile_label_smoothing,
        )
        by_cid = {str(record["cid"]): record for record in weighted_records}
        for record in records:
            weighted = by_cid.get(str(record["cid"]))
            if weighted is None:
                record["profile_multiplier"] = 1.0
                record["class_multiplier"] = 1.0
                record["quality_multiplier"] = 1.0
                record["cluster_multiplier"] = 1.0
                continue
            num_examples = max(float(weighted.get("num_examples", record["num_examples"])), EPS)
            record["profile_multiplier"] = float(weighted["aggregation_weight"]) / num_examples
            record["class_multiplier"] = float(weighted.get("class_multiplier", 1.0))
            record["quality_multiplier"] = float(weighted.get("quality_multiplier", 1.0))
            record["cluster_multiplier"] = float(weighted.get("cluster_multiplier", 1.0))
            record["class_score"] = float(weighted.get("class_score", 1.0))
            record["cluster_id"] = str(weighted.get("cluster_id", "unknown"))

    def _apply_drift_multipliers(self, records: list[dict[str, Any]]) -> None:
        if self.drift_penalty_strength <= EPS:
            for record in records:
                record["drift_multiplier"] = 1.0
            return

        norms = np.asarray([float(record.get("delta_norm", 0.0)) for record in records], dtype=float)
        positive = norms[norms > EPS]
        reference_norm = float(np.median(positive)) if positive.size else 0.0
        if reference_norm <= EPS:
            for record in records:
                record["drift_multiplier"] = 1.0
            return

        lower = max(float(self.drift_min_multiplier), EPS)
        for record in records:
            relative_excess = max((float(record.get("delta_norm", 0.0)) / reference_norm) - 1.0, 0.0)
            multiplier = float(np.exp(-self.drift_penalty_strength * relative_excess))
            record["drift_multiplier"] = float(np.clip(multiplier, lower, 1.0))

    def _server_optimized_delta(self, normalized_delta: list[np.ndarray]) -> list[np.ndarray]:
        optimizer_name = self.server_optimizer_name
        if optimizer_name in {"", "none", "fedavg"}:
            return [self.aggregation_lr * delta for delta in normalized_delta]
        if optimizer_name not in {"adam", "yogi"}:
            logger.warning(
                "Unknown FMRL-AVA server_optimizer=%s; using plain weighted delta.",
                optimizer_name,
            )
            return [self.aggregation_lr * delta for delta in normalized_delta]

        if self.server_momentum is None or len(self.server_momentum) != len(normalized_delta):
            self.server_momentum = [np.zeros_like(delta) for delta in normalized_delta]
        if self.server_second_moment is None or len(self.server_second_moment) != len(normalized_delta):
            self.server_second_moment = [np.zeros_like(delta) for delta in normalized_delta]

        beta1 = float(np.clip(self.server_beta1, 0.0, 0.999))
        beta2 = float(np.clip(self.server_beta2, 0.0, 0.999))
        tau = max(float(self.server_tau), EPS)
        optimized_delta: list[np.ndarray] = []
        for idx, delta in enumerate(normalized_delta):
            delta64 = delta.astype(np.float64, copy=False)
            self.server_momentum[idx] = (beta1 * self.server_momentum[idx]) + (
                (1.0 - beta1) * delta64
            )
            delta_sq = delta64 * delta64
            if optimizer_name == "yogi":
                self.server_second_moment[idx] = self.server_second_moment[idx] - (
                    (1.0 - beta2)
                    * delta_sq
                    * np.sign(self.server_second_moment[idx] - delta_sq)
                )
            else:
                self.server_second_moment[idx] = (beta2 * self.server_second_moment[idx]) + (
                    (1.0 - beta2) * delta_sq
                )
            denom = np.sqrt(np.maximum(self.server_second_moment[idx], 0.0)) + tau
            step = self.aggregation_lr * (self.server_momentum[idx] / denom)
            optimized_delta.append(step.astype(delta.dtype, copy=False))
        return optimized_delta

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

    def _client_quality(self, metrics: dict[str, Scalar]) -> float:
        for key in (
            "local_f1_macro",
            "f1_macro",
            "local_balanced_accuracy",
            "balanced_accuracy",
            "policy_accuracy",
            "accuracy",
        ):
            if key in metrics:
                return float(np.clip(self._float_metric(metrics, key), 0.0, 1.0))
        return 1.0

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

    args = dict(
        fraction_fit=cfg.server.fraction_fit,
        fraction_evaluate=cfg.server.fraction_evaluate,
        min_fit_clients=cfg.server.min_fit_clients,
        min_evaluate_clients=cfg.server.min_evaluate_clients,
        min_available_clients=cfg.server.min_available_clients,
        initial_parameters=_initial_parameters_from_checkpoint(cfg, device),
        on_fit_config_fn=fit_config_fn,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )

    if strat_name == "fmrl_ava":
        logger.info(
            "--- Strategy: FMRL-AVA (Adaptive Vector-Aligned Aggregation) with Model Saving ---"
        )
        return FMRLAdaptiveVectorAlignedAggregationStrategy(cfg=cfg, **args)

    if strat_name in {"fedmade", "class_aware", "class_aware_dynamic"}:
        logger.info("--- Strategy: FedMADE-style class-aware aggregation with Model Saving ---")
        return FedMADEClassAwareAggregationStrategy(cfg=cfg, **args)

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
