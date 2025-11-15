import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import flwr as fl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from flwr.common import Parameters, parameters_to_ndarrays
from omegaconf import DictConfig
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# get_device helper
try:
    from .utils import get_device
except ImportError:  # pragma: no cover - standalone usage
    from utils import get_device

try:
    from .agent import Agent
    from .environment import BlockchainIntrusionEnv
    from .exceptions import ConfigMismatchError
    from .local_training import run_local_training_round
    from .models import OpenSetQChainModelFactory
    from .policy import EpsilonGreedyPolicy, EpsilonScheduler
    from .replay_buffer import ExperienceReplayBuffer
except ImportError:  # pragma: no cover - standalone usage
    from agent import Agent
    from environment import BlockchainIntrusionEnv
    from exceptions import ConfigMismatchError
    from local_training import run_local_training_round
    from models import OpenSetQChainModelFactory
    from policy import EpsilonGreedyPolicy, EpsilonScheduler
    from replay_buffer import ExperienceReplayBuffer

logger = logging.getLogger("Client")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_project_path(path_like: Union[str, Path]) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


class FlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient implementing the Fed-Per agent."""

    def __init__(self, cid: str, cfg: DictConfig, data_path: str):
        self.cid = cid
        self.cfg = cfg
        self.data_path = data_path
        self.device = get_device()

        logger.info("Client %s: Initializing...", cid)

        self.model_factory = OpenSetQChainModelFactory(cfg.model)
        self.env = BlockchainIntrusionEnv(
            processed_data_path=self.data_path,
            steps_per_episode=cfg.training.steps_per_episode,
        )

        if (
            cfg.model.state_dim != self.env.feature_dim
            or cfg.model.num_actions != self.env.num_actions_nt
        ):
            raise ConfigMismatchError(
                f"Config/Env mismatch on client {cid}. "
                f"Config (s:{cfg.model.state_dim}, a:{cfg.model.num_actions}), "
                f"Env (s:{self.env.feature_dim}, a:{self.env.num_actions_nt}). "
            "Ensure 'env_metadata' in 'config_fl.yaml' matches your processed data."
            )

        self.agent = Agent(self.model_factory, cfg.training, self.device)
        self.buffer = ExperienceReplayBuffer(cfg.training.replay_buffer_size)
        self.policy = EpsilonGreedyPolicy(
            self.agent.prior_net, self.agent.value_net_main, cfg.model.num_actions, self.device
        )
        self.epsilon_scheduler = EpsilonScheduler(cfg.training)

        # Closed-set evaluation artifacts
        self.eval_enabled: bool = False
        self.eval_loader: Optional[DataLoader] = None
        self.eval_class_names: List[str] = []
        self.eval_output_dir: Optional[Path] = None
        self._init_closed_set_evaluation()

        logger.info("Client %s: Initialization complete.", cid)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        logger.debug("Client %s: get_parameters called", self.cid)
        return self.agent.get_federated_parameters()

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        logger.debug("Client %s: set_parameters called", self.cid)
        self.agent.set_federated_parameters(parameters, hard_target_update=True)

    def fit(
        self, parameters: Union[List[np.ndarray], Parameters], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        round_num = config.get("server_round", "?")
        logger.info("Client %s: fit() called for round %s", self.cid, round_num)

        param_list = (
            parameters if isinstance(parameters, list) else parameters_to_ndarrays(parameters)
        )
        self.set_parameters(param_list)

        num_steps_trained, metrics = run_local_training_round(
            agent=self.agent,
            env=self.env,
            buffer=self.buffer,
            policy=self.policy,
            epsilon_scheduler=self.epsilon_scheduler,
            cfg_training=self.cfg.training,
            device=self.device,
        )

        generator_cfg = getattr(self.cfg, "generator_training", None)
        generator_metrics: Dict[str, float] = {}
        if generator_cfg and bool(getattr(generator_cfg, "enabled", False)):
            round_val = config.get("server_round")
            try:
                round_index = int(round_val)
            except (TypeError, ValueError):
                round_index = -1
            total_rounds = int(getattr(self.cfg.server, "num_rounds", 0))
            if round_index >= total_rounds and total_rounds > 0:
                try:
                    features = self.env.all_features_s.clone()
                    labels = self.env.all_labels_a_t.clone()
                except AttributeError:
                    logger.warning(
                        "Client %s: environment does not expose raw features; skipping generator training.",
                        self.cid,
                    )
                else:
                    logger.info(
                        "Client %s: training generation network on %s local samples (final round).",
                        self.cid,
                        features.shape[0],
                    )
                    generator_metrics = self.agent.train_generation_network(
                        features, labels, generator_cfg
                    )
            else:
                logger.debug(
                    "Client %s: generator training waits for final round (%s/%s).",
                    self.cid,
                    round_index,
                    total_rounds,
                )
        else:
            logger.debug("Client %s: generator training disabled.", self.cid)

        if generator_metrics:
            metrics.update(generator_metrics)

        updated_parameters = self.get_parameters(config={})
        return updated_parameters, num_steps_trained, metrics

    def evaluate(
        self, parameters: Union[List[np.ndarray], Parameters], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, float]]:
        round_num = config.get("server_round", "?")
        logger.info("Client %s: evaluate() called for round %s", self.cid, round_num)

        param_list = (
            parameters if isinstance(parameters, list) else parameters_to_ndarrays(parameters)
        )
        self.set_parameters(param_list)

        try:
            round_index = int(round_num)
        except (TypeError, ValueError):
            round_index = 0

        if not self.eval_enabled or self.eval_loader is None:
            logger.info(
                "Client %s: closed-set evaluation disabled; returning placeholder metrics.", self.cid
            )
            return 0.0, 0, {}

        loss, num_examples, metrics = self._evaluate_closed_set(round_index)
        return loss, num_examples, metrics

    # ------------------------------------------------------------------
    # Closed-set evaluation helpers
    # ------------------------------------------------------------------

    def _init_closed_set_evaluation(self) -> None:
        """Prepare dataloader and output folder for closed-set evaluation."""
        paths_cfg = getattr(self.cfg, "paths", None)
        if paths_cfg is None:
            logger.warning("Client %s: cfg.paths missing; closed-set evaluation disabled.", self.cid)
            return

        test_data_rel = getattr(paths_cfg, "closed_set_test_data", None)
        class_names_rel = getattr(paths_cfg, "class_names", None)
        figures_dir_rel = getattr(paths_cfg, "figures_dir", None)
        if not (test_data_rel and class_names_rel and figures_dir_rel):
            logger.warning(
                "Client %s: closed-set evaluation paths not fully specified; disabled.", self.cid
            )
            return

        test_data_path = _resolve_project_path(test_data_rel)
        class_names_path = _resolve_project_path(class_names_rel)
        figures_dir = _resolve_project_path(figures_dir_rel) / "clients" / f"client_{self.cid}"

        try:
            data = torch.load(test_data_path, map_location="cpu")
            features = data["features"].float()
            labels = data["labels"].long()
        except FileNotFoundError:
            logger.warning(
                "Client %s: closed-set test data not found at %s; evaluation disabled.",
                self.cid,
                test_data_path,
            )
            return
        except Exception as exc:
            logger.error(
                "Client %s: failed to load closed-set test data: %s", self.cid, exc, exc_info=True
            )
            return

        dataset = TensorDataset(features, labels)
        self.eval_loader = DataLoader(
            dataset, batch_size=self.cfg.training.batch_size, shuffle=False
        )
        self.eval_class_names = self._load_class_names(
            class_names_path, int(self.cfg.model.num_actions)
        )

        figures_dir.mkdir(parents=True, exist_ok=True)
        self.eval_output_dir = figures_dir
        self.eval_enabled = True
        logger.info(
            "Client %s: closed-set evaluation enabled. Saving artifacts to %s",
            self.cid,
            figures_dir,
        )

    @staticmethod
    def _load_class_names(path: Path, num_actions: int) -> List[str]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    raw = json.load(fp)
                sorted_items = sorted(((int(k), v) for k, v in raw.items()), key=lambda x: x[0])
                return [name for _, name in sorted_items]
            except Exception as exc:
                logger.warning("Client class-names load failed (%s): %s", path, exc)
        return [f"class_{idx}" for idx in range(num_actions)]

    def _run_closed_set_inference(self) -> Tuple[float, np.ndarray, np.ndarray]:
        assert self.eval_loader is not None
        self.agent.prior_net.eval()
        self.agent.value_net_main.eval()

        total_loss = 0.0
        total_samples = 0
        all_true: List[int] = []
        all_pred: List[int] = []

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

    def _save_client_report(self, round_index: int, report: str) -> None:
        if not self.eval_output_dir:
            return
        path = self.eval_output_dir / f"client_{self.cid}_report_round_{round_index:03d}.txt"
        try:
            path.write_text(report)
        except Exception as exc:  # pragma: no cover - filesystem errors
            logger.warning("Client %s: failed to write evaluation report: %s", self.cid, exc)

    def _plot_client_confusion_matrix(
        self, round_index: int, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        if not self.eval_output_dir:
            return

        labels = list(range(len(self.eval_class_names)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(cm_norm, cmap=plt.get_cmap("Blues"), vmin=0, vmax=1)
        ax.set_xticks(range(len(self.eval_class_names)))
        ax.set_yticks(range(len(self.eval_class_names)))
        ax.set_xticklabels(self.eval_class_names, rotation=45, ha="right")
        ax.set_yticklabels(self.eval_class_names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Client {self.cid} Confusion Matrix (Round {round_index})")

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

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Normalized Count", rotation=270, labelpad=12)

        fig.tight_layout()
        fig_path = self.eval_output_dir / f"client_{self.cid}_cm_round_{round_index:03d}.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)

    def _evaluate_closed_set(self, round_index: int) -> Tuple[float, int, Dict[str, float]]:
        loss, y_true, y_pred = self._run_closed_set_inference()
        num_examples = int(y_true.size)
        accuracy = float((y_true == y_pred).mean()) if num_examples else 0.0

        if num_examples == 0:
            logger.warning("Client %s: closed-set evaluation has no samples.", self.cid)
            return loss, 0, {"accuracy": accuracy}

        report = classification_report(
            y_true, y_pred, target_names=self.eval_class_names, digits=4, zero_division=0
        )
        self._save_client_report(round_index, report)
        self._plot_client_confusion_matrix(round_index, y_true, y_pred)

        logger.info(
            "Client %s: closed-set evaluation round %s | loss=%.4f | accuracy=%.4f | samples=%s",
            self.cid,
            round_index,
            loss,
            accuracy,
            num_examples,
        )
        logger.debug("Client %s evaluation report (round %s):\n%s", self.cid, round_index, report)

        return loss, num_examples, {"accuracy": accuracy}
