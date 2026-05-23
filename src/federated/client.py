import json
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

import flwr as fl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from flwr.common import Parameters, parameters_to_ndarrays
from omegaconf import DictConfig
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.agents.agent import Agent
from src.agents.policy import EpsilonGreedyPolicy, EpsilonScheduler
from src.evaluation.openset_eval import (
    calibrate_evt_thresholds,
    evaluate_open_set,
    fit_evt_models,
)
from src.models.models import OpenSetQChainModelFactory
from src.openset.evt import save_evt_collection, save_evt_meta
from src.rl.environment import BlockchainIntrusionEnv
from src.rl.local_training import run_local_training_round
from src.rl.replay_buffer import ExperienceReplayBuffer
from src.utils.utils import get_device, project_root

PROJECT_ROOT = project_root()


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


class FlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient implementing Fed-Per/FedAvg with hybrid logic."""

    def __init__(
        self,
        cid: str,
        cfg: DictConfig,
        data_path: str,
        device: torch.device | None = None,
    ):
        self.cid = cid
        self.logger = logging.getLogger(f"Client.{cid}")
        self.cfg = cfg
        self.data_path = data_path
        self.device = device if device is not None else get_device(logger=self.logger)

        self._move_data_to_device = bool(cfg.device.move_data_to_device)

        self.logger.info("Client %s: Initializing...", cid)
        self.model_factory = OpenSetQChainModelFactory(cfg.model)

        # NON-IID FIX: Pass global number of actions to Environment
        self.env = BlockchainIntrusionEnv(
            processed_data_path=self.data_path,
            steps_per_episode=cfg.training.steps_per_episode,
            device=self.device,
            logger=self.logger,
            move_data_to_device=self._move_data_to_device,
            global_num_actions=cfg.model.num_actions,
        )

        # -----------------------------------------------------------
        # VALIDATION: Relaxed for Non-IID
        # -----------------------------------------------------------
        if cfg.model.state_dim != self.env.feature_dim:
            raise ValueError(
                f"State Dim mismatch on client {cid}: "
                f"Config({cfg.model.state_dim}) != Env({self.env.feature_dim})"
            )

        # It is CRITICAL if env has MORE actions than the model can predict
        if self.env.num_actions_nt > cfg.model.num_actions:
            raise ValueError(
                f"CRITICAL: Environment has more classes ({self.env.num_actions_nt}) "
                f"than Model output nodes ({cfg.model.num_actions}). Increase config.model.num_actions!"
            )

        # It is OKAY (Non-IID) if env has FEWER actions. Just warn.
        if self.env.num_actions_nt < cfg.model.num_actions:
            self.logger.warning(
                f"Client {cid} Non-IID: Env has {self.env.num_actions_nt} classes, "
                f"Model has {cfg.model.num_actions}. Missing classes will have 0 reward locally."
            )

        self.agent = Agent(self.model_factory, cfg.training, self.device, logger=self.logger)
        self.buffer = ExperienceReplayBuffer(cfg.training.replay_buffer_size)
        self.policy = EpsilonGreedyPolicy(
            self.agent.prior_net,
            self.agent.value_net_main,
            cfg.model.num_actions,
            self.device,
        )
        self.epsilon_scheduler = EpsilonScheduler(cfg.training)

        # Cache
        self.cached_weights: list[np.ndarray] = []
        self.cached_metrics: dict[str, Any] = {}
        self.lifetime_reward = 0.0
        self.local_data_profile = self._build_local_data_profile()

        # Directories
        figures_root = _resolve_project_path(cfg.paths.figures_dir)
        self.client_figure_dir: Path = figures_root / "clients" / f"client_{cid}"
        self.client_figure_dir.mkdir(parents=True, exist_ok=True)

        evt_root = _resolve_project_path(cfg.paths.evt_dir)
        self.evt_output_dir: Path = evt_root / f"client_{cid}"
        self.evt_output_dir.mkdir(parents=True, exist_ok=True)
        self.openset_output_dir: Path = self.client_figure_dir / "openset"
        self.openset_output_dir.mkdir(parents=True, exist_ok=True)

        # Closed-set init
        self.eval_enabled: bool = False
        self.eval_loader: DataLoader | None = None
        self.eval_class_names: list[str] = []
        self.eval_output_dir: Path | None = self.client_figure_dir
        self.closed_set_data_path: Path | None = None
        self.open_set_data_path: Path | None = None
        self._init_closed_set_evaluation()

        self.logger.info("Client %s: Initialization complete.", cid)

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        _ = config
        return self.agent.get_federated_parameters()

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        self.agent.set_federated_parameters(parameters, hard_target_update=True)

    def fit(
        self, parameters: list[np.ndarray] | Parameters, config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        # 1. Determine Phase
        phase = config.get("phase", "standard")
        round_num = config.get("server_round", "?")
        strategy_name = str(self.cfg.strategy.name).lower()
        proximal_mu = float(self.cfg.server.proximal_mu) if strategy_name == "fedprox" else 0.0

        # =========================================================
        # STANDARD FIT (FedAvg / FedProx)
        # =========================================================
        if phase == "standard":
            phase_name = "FedProx" if proximal_mu > 0.0 else "FedAvg"
            self.logger.info(f"Client {self.cid} [{phase_name}]: Round {round_num}")

            param_list = (
                parameters if isinstance(parameters, list) else parameters_to_ndarrays(parameters)
            )
            self.set_parameters(param_list)
            num_steps_trained, metrics = self._perform_training_loop(proximal_mu=proximal_mu)
            metrics.setdefault("total_steps", float(num_steps_trained))
            updated_params = self.agent.get_federated_parameters()
            num_examples = int(self.local_data_profile["local_num_examples"])
            return updated_params, num_examples, metrics

        # =========================================================
        # FMRL PHASE A: TRAIN & AUDIT (Calculate H_i and W_i)
        # =========================================================
        if phase == "train":
            self.logger.info(f"Client {self.cid} [FMRL Phase A]: Round {round_num}")

            # A. Update Global Model
            param_list = (
                parameters if isinstance(parameters, list) else parameters_to_ndarrays(parameters)
            )
            self.set_parameters(param_list)

            # B. Train
            num_steps_trained, metrics = self._perform_training_loop()

            # C. Generate Audit Metadata
            self.lifetime_reward += metrics.get("total_reward", 0.0)

            # Stage-one metadata: hidden state plus local utility diagnostics.
            audit_signals = self._calculate_audit_signals()

            # D. Cache Everything
            self.cached_weights = self.agent.get_federated_parameters()
            self.cached_metrics = metrics.copy()

            # Prepare the payload for the server critic
            self.cached_metrics.update(
                {
                    "cid": self.cid,
                    "total_steps": float(num_steps_trained),
                    "local_num_examples": self.local_data_profile["local_num_examples"],
                    "class_entropy": self.local_data_profile["class_entropy"],
                    "label_coverage": self.local_data_profile["label_coverage"],
                    "label_histogram": self.local_data_profile["label_histogram"],
                    "hidden_info": json.dumps(audit_signals["mu_vector"]),
                    "recent_reward": metrics.get("avg_reward_per_episode", 0.0),
                    "history_reward": self.lifetime_reward,
                    "audit_f1": audit_signals["audit_f1"],
                    "utility_loss": audit_signals["audit_f1"],
                    "td_error": audit_signals["td_error"],
                    "kl_div": audit_signals["kl_div"],
                    "audit_batch_size": float(audit_signals["audit_batch_size"]),
                }
            )

            self.logger.info(
                f"   > Client {self.cid}: Caching weights. Signals -> "
                f"Reward: {metrics.get('avg_reward_per_episode', 0):.2f}, "
                f"TD: {audit_signals['td_error']:.4f}, "
                f"KL: {audit_signals['kl_div']:.4f}"
            )

            # E. Return EMPTY weights (Server decides if it wants them later)
            num_examples = int(self.local_data_profile["local_num_examples"])
            return [], num_examples, self.cached_metrics

        # =========================================================
        # FMRL PHASE B: UPLOAD (If selected by Critic)
        # =========================================================
        if phase == "upload":
            self.logger.info(f"Client {self.cid} [FMRL Phase B]: Selected! Uploading.")

            if not self.cached_weights:
                self.logger.error("   > Error: No cached weights found!")
                return self.agent.get_federated_parameters(), 0, {"error": 1.0}

            # Return the weights we cached in Phase A
            num_examples = int(self.local_data_profile["local_num_examples"])
            return self.cached_weights, num_examples, self.cached_metrics

        self.logger.warning(f"Unknown Phase: {phase}")
        return [], 0, {}

    # --- INTERNAL TRAINING HELPER ---
    def _build_local_data_profile(self) -> dict[str, Any]:
        labels = self.env.all_labels_a_t.detach().cpu().long()
        num_actions = int(self.cfg.model.num_actions)
        counts = torch.bincount(labels.clamp(min=0), minlength=num_actions)[:num_actions]
        total = int(labels.numel())
        probs = counts.float() / max(total, 1)
        nonzero = probs[probs > 0]
        entropy = (
            float((-(nonzero * torch.log(nonzero))).sum().item() / np.log(num_actions))
            if num_actions > 1 and nonzero.numel() > 0
            else 0.0
        )
        coverage = float((counts > 0).sum().item() / max(num_actions, 1))
        return {
            "local_num_examples": float(total),
            "class_entropy": entropy,
            "label_coverage": coverage,
            "label_histogram": json.dumps(
                {str(i): int(counts[i].item()) for i in range(num_actions)},
                sort_keys=True,
            ),
        }

    def _perform_training_loop(self, proximal_mu: float = 0.0) -> tuple[int, dict[str, Any]]:
        """Shared training logic for both Standard and FMRL modes."""
        num_steps_trained, metrics = run_local_training_round(
            agent=self.agent,
            env=self.env,
            buffer=self.buffer,
            policy=self.policy,
            epsilon_scheduler=self.epsilon_scheduler,
            cfg_training=self.cfg.training,
            device=self.device,
            proximal_mu=proximal_mu,
            logger=self.logger,
        )

        # Optional Generator Training
        generator_cfg = self.cfg.generator_training
        if bool(generator_cfg.enabled):
            try:
                features = self.env.all_features_s.clone()
                labels = self.env.all_labels_a_t.clone()
                gen_metrics = self.agent.train_generation_network(
                    features,
                    labels,
                    generator_cfg,
                    proximal_mu=proximal_mu,
                    logger=self.logger,
                )
                metrics.update(gen_metrics)
            except Exception as exc:
                self.logger.warning(f"Generator training failed: {exc}")

        # Optional Local Eval
        self._run_local_eval_logic(metrics, "LOCAL")
        return num_steps_trained, metrics

    def _calculate_audit_signals(self) -> dict[str, Any]:
        """
        Calculates audit signals including TD Error (Surprise), F1 (Competence),
        and KL Divergence (Novelty).
        """
        # 1. Sample Batch. Use smaller audit batches early instead of returning
        # zeros until the replay buffer reaches the training batch size.
        audit_batch_size = min(len(self.buffer), int(self.cfg.training.batch_size))
        if audit_batch_size <= 0:
            return {
                "mu_vector": [0.0] * self.cfg.model.latent_dim,
                "td_error": 0.0,
                "audit_f1": 0.0,
                "kl_div": 0.0,
                "audit_batch_size": 0,
            }

        # 2. Unpack the 6 items from your buffer
        batch = self.buffer.sample(audit_batch_size, self.device)
        states, actions, rewards, next_states, dones, true_actions = batch

        # Prepare Networks
        self.agent.prior_net.eval()
        self.agent.recognition_net.eval()
        self.agent.value_net_main.eval()
        self.agent.value_net_target.eval()

        with torch.no_grad():
            # ---------------------------------------------------------
            # 1. Hidden State (Context) & KL Divergence (Novelty)
            # ---------------------------------------------------------
            # Prior P(z|s)
            mu_prior, logvar_prior = self.agent.prior_net(states)

            # Posterior Q(z|s, a_true) (Using true actions for best context estimation)
            mu_post, logvar_post = self.agent.recognition_net(states, true_actions)

            # Calculate KL(Q || P) - Manually to avoid import dependency issues
            # KL = 0.5 * sum(exp(var_q - var_p) + (mu_p - mu_q)^2 / exp(var_p) - 1 + var_p - var_q)
            # We use the simplified version for diagonal Gaussians:
            # KL = -0.5 * sum(1 + log_var_q - log_var_p - (var_q + (mu_q - mu_p)^2) / var_p)
            var_p = torch.exp(logvar_prior)
            var_q = torch.exp(logvar_post)
            kl_element = -0.5 * (
                1 + logvar_post - logvar_prior - (var_q + (mu_post - mu_prior).pow(2)) / var_p
            )
            kl_div = kl_element.sum(dim=1).mean().item()

            avg_mu_vector = mu_prior.mean(dim=0).cpu().numpy().tolist()

            # ---------------------------------------------------------
            # 2. TD-Error (RL Surprise)
            # ---------------------------------------------------------
            # Current Q(s, a)
            q_values = self.agent.value_net_main(mu_prior, states)
            current_q = q_values.gather(1, actions)

            # Target Q(s', a')
            next_mu, _ = self.agent.prior_net(next_states)
            next_q_values = self.agent.value_net_target(next_mu, next_states)
            max_next_q = next_q_values.max(1)[0].unsqueeze(1)

            # Bellman Target
            target_q = rewards + (self.cfg.training.gamma * max_next_q * (1 - dones))

            # Calculate TD Error
            td_error = F.l1_loss(current_q, target_q).item()

            # ---------------------------------------------------------
            # 3. COMPETENCE SIGNAL: Batch F1 Score
            # ---------------------------------------------------------
            fresh_preds = q_values.argmax(dim=1)

            # Move to CPU for sklearn
            y_true = true_actions.cpu().numpy().flatten()
            y_pred = fresh_preds.cpu().numpy().flatten()

            # Calculate Macro F1
            batch_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0.0)

        # Restore Training Mode
        self.agent.prior_net.train()
        self.agent.recognition_net.train()
        self.agent.value_net_main.train()
        self.agent.value_net_target.train()

        return {
            "mu_vector": avg_mu_vector,
            "td_error": td_error,
            "audit_f1": float(batch_f1),
            "kl_div": float(kl_div),
            "audit_batch_size": int(audit_batch_size),
        }

    def _run_local_eval_logic(self, metrics: dict[str, float], _prefix: str):
        if self.eval_enabled:
            _, _, local_metrics = self._evaluate_closed_set(0, prefix="LOCAL_PRE_AGG")
            metrics.update({f"local_{k}": v for k, v in local_metrics.items()})

    def evaluate(
        self, parameters: list[np.ndarray] | Parameters, config: dict[str, Any]
    ) -> tuple[float, int, dict[str, float]]:
        round_num = config.get("server_round", "?")
        self.logger.info("Client %s: evaluate() called for round %s", self.cid, round_num)

        param_list = (
            parameters if isinstance(parameters, list) else parameters_to_ndarrays(parameters)
        )
        self.set_parameters(param_list)

        try:
            round_index = int(round_num)
        except (TypeError, ValueError):
            round_index = 0

        metrics: dict[str, float] = {}

        if not self.eval_enabled or self.eval_loader is None:
            return 0.0, 0, {}

        loss, num_examples, cs_metrics = self._evaluate_closed_set(
            round_index, prefix="GLOBAL_POST_AGG"
        )
        metrics.update(cs_metrics)

        # EVT Logic (Optional)
        evt_cfg = self.cfg.evt
        if bool(evt_cfg.enabled):
            try:
                features = self.env.all_features_s.clone()
                labels = self.env.all_labels_a_t.clone()
                evt_metrics = self._fit_evt_and_run_openset_eval(
                    features, labels, evt_cfg, prefix="GLOBAL"
                )
                metrics.update(evt_metrics)
            except Exception as e:
                self.logger.error(f"EVT Eval failed: {e}")

        metrics["cid"] = self.cid
        return loss, num_examples, metrics

    # ------------------------------------------------------------------
    # Closed-set evaluation helpers
    # ------------------------------------------------------------------

    def _init_closed_set_evaluation(self) -> None:
        """Prepare dataloader and output folder for closed-set evaluation."""
        paths_cfg = self.cfg.paths

        test_data_key = f"test_closed_client_{self.cid}"
        shared_rel = getattr(paths_cfg, "shared_closed_set_test_data", None)
        generic_rel = getattr(paths_cfg, "closed_set_test_data", None)
        client_rel = getattr(paths_cfg, test_data_key, None)

        test_data_rel = None
        for candidate in (shared_rel, generic_rel, client_rel):
            if candidate:
                test_data_rel = candidate
                break

        class_names_rel = getattr(paths_cfg, "class_names", None)
        if not (test_data_rel and class_names_rel):
            self.logger.warning("Client %s: paths for %s missing.", self.cid, test_data_key)
            return

        test_data_path = _resolve_project_path(test_data_rel)
        class_names_path = _resolve_project_path(class_names_rel)
        self.closed_set_data_path = test_data_path

        try:
            data_device = self.device if self._move_data_to_device else torch.device("cpu")
            data = torch.load(test_data_path, map_location="cpu", weights_only=True)
            features = data["features"].to(device=data_device).float()
            labels = data["labels"].to(device=data_device).long()
        except Exception as exc:
            self.logger.error("Client %s: failed to load closed-set test data: %s", self.cid, exc)
            return

        dataset = TensorDataset(features, labels)
        self.eval_loader = DataLoader(
            dataset, batch_size=self.cfg.training.batch_size, shuffle=False
        )
        self.eval_class_names = self._load_class_names(
            class_names_path, int(self.cfg.model.num_actions)
        )
        self.logger.info(
            "Client %s: Closed-set eval data loaded (%s) | samples=%d",
            self.cid,
            test_data_path.name,
            len(dataset),
        )

        self.eval_enabled = True
        self.logger.info("Client %s: Closed-set eval enabled using %s", self.cid, test_data_path.name)

    def _load_class_names(self, path: Path, num_actions: int) -> list[str]:
        if path.exists():
            try:
                with open(path, encoding="utf-8") as fp:
                    raw = json.load(fp)
                sorted_items = sorted(((int(k), v) for k, v in raw.items()), key=lambda x: x[0])
                return [name for _, name in sorted_items]
            except Exception as exc:
                self.logger.warning("Client class-names load failed (%s): %s", path, exc)
        return [f"class_{idx}" for idx in range(num_actions)]

    def _run_closed_set_inference(self) -> tuple[float, np.ndarray, np.ndarray]:
        assert self.eval_loader is not None
        self.agent.prior_net.eval()
        self.agent.value_net_main.eval()

        total_loss = 0.0
        total_samples = 0
        all_true: list[int] = []
        all_pred: list[int] = []

        with torch.no_grad():
            for features, labels in self.eval_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                mu_p, _ = self.agent.prior_net(features)
                q_values = self.agent.value_net_main(mu_p, features)
                loss = F.cross_entropy(q_values, labels, reduction="mean")

                preds = q_values.argmax(dim=1)
                all_true.extend(labels.cpu().numpy().tolist())
                all_pred.extend(preds.cpu().numpy().tolist())

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples else 0.0
        return avg_loss, np.array(all_true), np.array(all_pred)

    def _save_client_report(self, round_index: int, report: str, prefix: str) -> None:
        if not self.eval_output_dir:
            return
        path = self.eval_output_dir / f"report_{prefix}_round_{round_index:03d}.txt"
        with suppress(Exception):
            path.write_text(report)

    def _plot_client_confusion_matrix(
        self, round_index: int, y_true: np.ndarray, y_pred: np.ndarray, prefix: str
    ) -> None:
        if not self.eval_output_dir:
            return

        labels = list(range(len(self.eval_class_names)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.imshow(cm_norm, cmap=plt.get_cmap("Blues"), vmin=0, vmax=1)
        ax.set_xticks(range(len(self.eval_class_names)))
        ax.set_yticks(range(len(self.eval_class_names)))
        ax.set_xticklabels(self.eval_class_names, rotation=45, ha="right")
        ax.set_yticklabels(self.eval_class_names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Client {self.cid} {prefix} CM (Round {round_index})")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm_norm[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val > 0.5 else "black",
                    fontsize=9,
                )

        fig.tight_layout()
        fig_path = self.eval_output_dir / f"cm_{prefix}_round_{round_index:03d}.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)

    def _evaluate_closed_set(
        self, round_index: int, prefix: str = "GLOBAL"
    ) -> tuple[float, int, dict[str, float]]:
        loss, y_true, y_pred = self._run_closed_set_inference()
        num_examples = int(y_true.size)
        accuracy = float((y_true == y_pred).mean()) if num_examples else 0.0
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        if num_examples == 0:
            return loss, 0, {"accuracy": accuracy, "f1_macro": 0.0, "num_examples": 0}

        report = classification_report(
            y_true,
            y_pred,
            target_names=self.eval_class_names,
            digits=4,
            zero_division=0,
        )
        self._save_client_report(round_index, report, prefix)
        self._plot_client_confusion_matrix(round_index, y_true, y_pred, prefix)

        self.logger.info(f"\n[Client {self.cid} | {prefix}] Closed-Set Report:\n{report}")

        return (
            loss,
            num_examples,
            {
                "accuracy": accuracy,
                "f1_macro": f1,
                "num_examples": num_examples,
            },
        )

    def _fit_evt_and_run_openset_eval(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        evt_cfg: DictConfig,
        prefix: str = "GLOBAL",
    ) -> dict[str, float]:
        paths_cfg = self.cfg.paths

        test_open_key = f"test_open_client_{self.cid}"
        shared_open_rel = getattr(paths_cfg, "shared_open_set_test_data", None)
        generic_open_rel = getattr(paths_cfg, "open_set_test_data", None)
        client_open_rel = getattr(paths_cfg, test_open_key, None)

        open_set_rel = None
        for candidate in (shared_open_rel, generic_open_rel, client_open_rel):
            if candidate:
                open_set_rel = candidate
                break

        class_names_rel = getattr(paths_cfg, "class_names", None)
        if not (open_set_rel and class_names_rel):
            self.logger.warning("Client %s: open-set paths missing for %s", self.cid, test_open_key)
            return {}

        try:
            evt_models = fit_evt_models(
                features=features.float(),
                labels=labels.long(),
                batch_size=int(self.cfg.training.batch_size),
                evt_cfg=evt_cfg,
                prior_net=self.agent.prior_net,
                recognition_net=self.agent.recognition_net,
                value_net_main=self.agent.value_net_main,
                generation_net=self.agent.generation_net,
                device=self.device,
                logger=self.logger,
            )
        except Exception:
            self.logger.exception("Client %s: EVT fitting failed.", self.cid)
            return {}

        evt_model_path = self.evt_output_dir / f"evt_models_{prefix}.pkl"
        save_evt_collection(evt_models, evt_model_path, logger=self.logger)

        try:
            meta = calibrate_evt_thresholds(
                features=features.float(),
                labels=labels.long(),
                batch_size=int(self.cfg.training.batch_size),
                evt_models=evt_models,
                evt_cfg=evt_cfg,
                prior_net=self.agent.prior_net,
                recognition_net=self.agent.recognition_net,
                value_net_main=self.agent.value_net_main,
                generation_net=self.agent.generation_net,
                device=self.device,
                logger=self.logger,
            )
        except Exception:
            self.logger.exception("Client %s: EVT threshold calibration failed.", self.cid)
            return {}

        default_delta = float(evt_cfg.decision_threshold)
        meta.setdefault("global_delta", default_delta)
        save_evt_meta(meta, self.evt_output_dir / f"evt_meta_{prefix}.json", logger=self.logger)

        open_set_path = _resolve_project_path(open_set_rel)
        class_names_path = _resolve_project_path(class_names_rel)
        self.open_set_data_path = open_set_path

        try:
            with open(class_names_path, encoding="utf-8") as fh:
                class_map = {int(k): v for k, v in json.load(fh).items()}

            data = torch.load(open_set_path, map_location="cpu", weights_only=True)
            open_features = (
                data["features"]
                .to(device=(self.device if self._move_data_to_device else torch.device("cpu")))
                .float()
            )
            open_labels = (
                data["labels"]
                .to(device=(self.device if self._move_data_to_device else torch.device("cpu")))
                .long()
            )
            openset_examples = int(open_labels.numel())
            num_unknown = int((open_labels == -1).sum().item())
            self.logger.info(
                "Client %s: Open-set eval data loaded (%s) | total=%d | unknown=%d",
                self.cid,
                open_set_path.name,
                openset_examples,
                num_unknown,
            )
        except Exception as exc:
            self.logger.error("Client %s: Error loading open set data: %s", self.cid, exc)
            return {}

        metrics = evaluate_open_set(
            features=open_features,
            labels=open_labels,
            batch_size=int(self.cfg.training.batch_size),
            prior_net=self.agent.prior_net,
            recognition_net=self.agent.recognition_net,
            value_net_main=self.agent.value_net_main,
            generation_net=self.agent.generation_net,
            evt_models=evt_models,
            evt_meta=meta,
            class_names=class_map,
            output_dir=self.openset_output_dir,
            device=self.device,
            evt_cfg=evt_cfg,
            report_to_stdout=False,
            logger=self.logger,
        )
        metrics["openset_examples"] = openset_examples
        metrics["openset_unknown"] = num_unknown
        metrics["open_set_dataset"] = open_set_path.name
        self.logger.info(
            "[Client %s | %s] Open-Set AUROC: %.4f",
            self.cid,
            prefix,
            metrics.get("openset_auroc", 0.0),
        )
        return metrics
