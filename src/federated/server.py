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
        torch.save(checkpoint, round_path)
        torch.save(checkpoint, model_dir / "global_model_latest.pt")
        torch.save(checkpoint, _resolve_path(cfg.checkpointing.latest_checkpoint_path))
        if bool(cfg.checkpointing.save_best):
            best_path = _resolve_path(cfg.checkpointing.best_model_path)
            if not best_path.exists():
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
            save_global_model(aggregated_parameters, server_round, self.cfg)

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
            save_global_model(aggregated_parameters, server_round, self.cfg)

        return aggregated_parameters, aggregated_metrics


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
