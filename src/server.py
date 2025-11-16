import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from flwr.server.strategy import FedProx, Strategy
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from .models import OpenSetQChainModelFactory
from .utils import get_device

logger = logging.getLogger('Server')

REWARD_HISTORY: List[Tuple[int, float]] = []


def _project_root() -> Path:
    try:
        return Path(get_original_cwd())
    except ValueError:
        return Path(os.getcwd())


def _resolve_path(path_like) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (_project_root() / path)


def reset_reward_history() -> None:
    """Clear stored reward metrics before a new training session."""
    REWARD_HISTORY.clear()


def get_reward_history() -> List[Tuple[int, float]]:
    """Return a shallow copy of the reward history."""
    return list(REWARD_HISTORY)


def fit_config_fn(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Pass the server round number to the client's fit method."""
    return {"server_round": server_round}


def aggregate_fit_metrics(
    fit_metrics: List[Tuple[int, Dict[str, fl.common.Scalar]]]
) -> Dict[str, float]:
    """Aggregate client-side metrics returned by Flower's FedAvg strategy."""
    if not fit_metrics:
        logger.warning("No fit metrics to aggregate; skipping.")
        return {}

    total_examples = sum(num_examples for num_examples, _ in fit_metrics)
    if total_examples == 0:
        logger.warning("Fit metrics reported zero examples; skipping aggregation.")
        return {}

    aggregated: Dict[str, float] = {}
    metric_keys = ["avg_reward_per_episode", "avg_td_loss", "avg_kl_loss", "avg_q_value"]

    generator_sample_total = 0.0
    generator_loss_weighted = 0.0
    generator_correct_weighted = 0.0

    for key in metric_keys:
        weighted_sum = 0.0
        present = False
        for num_examples, metrics in fit_metrics:
            metric_val = metrics.get(key)
            if metric_val is None:
                continue
            weighted_sum += float(metric_val) * num_examples
            present = True
        if present:
            aggregated[key] = weighted_sum / total_examples

    for _, metrics in fit_metrics:
        gen_samples = float(metrics.get("generator_samples", 0.0))
        if gen_samples <= 0:
            continue
        generator_sample_total += gen_samples
        gen_loss = metrics.get("generator_loss")
        if gen_loss is not None:
            generator_loss_weighted += float(gen_loss) * gen_samples
        gen_correct = metrics.get("generator_correct_frac")
        if gen_correct is not None:
            generator_correct_weighted += float(gen_correct) * gen_samples

    if generator_sample_total > 0:
        aggregated["generator_samples"] = generator_sample_total
        aggregated["generator_loss"] = generator_loss_weighted / generator_sample_total
        aggregated["generator_correct_frac"] = (
            generator_correct_weighted / generator_sample_total
            if generator_sample_total
            else 0.0
        )
        logger.info(
            "Aggregated generator metrics | samples=%s | loss=%.6f | correct_frac=%.4f",
            generator_sample_total,
            aggregated["generator_loss"],
            aggregated["generator_correct_frac"],
        )
    else:
        logger.info(
            "No generator metrics received from clients this round; skip aggregated generator stats."
        )

    avg_reward = aggregated.get("avg_reward_per_episode")
    if avg_reward is not None:
        REWARD_HISTORY.append((len(REWARD_HISTORY) + 1, avg_reward))

    return aggregated


class ClosedSetEvaluator:
    """Evaluate the aggregated global model on held-out closed-set data."""

    def __init__(self, cfg: DictConfig, device: Optional[torch.device] = None):
        self.device = device if device is not None else get_device()
        self.cfg = cfg
        self.enabled = True
        device_cfg = getattr(cfg, "device", None)
        move_data_to_device = bool(getattr(device_cfg, "move_data_to_device", False)) if device_cfg else False

        data_path = _resolve_path(cfg.paths.closed_set_test_data)
        class_names_path = _resolve_path(cfg.paths.class_names)
        self.figure_dir = _resolve_path(cfg.paths.figures_dir)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        try:
            data_device = self.device if move_data_to_device else torch.device("cpu")
            data = torch.load(data_path, map_location="cpu", weights_only=True)
            features = data["features"].to(device=data_device).float()
            labels = data["labels"].to(device=data_device).long()
            self.loader = DataLoader(
                TensorDataset(features, labels),
                batch_size=cfg.training.batch_size,
                shuffle=False,
            )
        except FileNotFoundError:
            logger.warning("Closed-set test data not found at %s; evaluation disabled.", data_path)
            self.enabled = False
            self.loader = None
        except Exception as exc:
            logger.error("Failed to load closed-set test data: %s", exc, exc_info=True)
            self.enabled = False
            self.loader = None

        self.class_names = self._load_class_names(class_names_path, cfg.model.num_actions)

        model_factory = OpenSetQChainModelFactory(cfg.model)
        self.value_network = model_factory.create_value_network().to(self.device)
        self.prior_net = self.value_network.encoder.prior
        self.recognition_net = self.value_network.encoder.recognition
        self.main_q_net = self.value_network.decoder.main_q
        self.generation_net = model_factory.create_generation_network().to(self.device)
        self.prior_keys = list(self.prior_net.state_dict().keys())
        self.recog_keys = list(self.recognition_net.state_dict().keys())
        self.main_keys = list(self.main_q_net.state_dict().keys())
        self.generator_keys = list(self.generation_net.state_dict().keys())

    @staticmethod
    def _load_class_names(path: Path, num_actions: int) -> List[str]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                sorted_items = sorted(((int(k), v) for k, v in raw.items()), key=lambda x: x[0])
                return [name for _, name in sorted_items]
            except Exception as exc:
                logger.warning("Failed to read class names from %s: %s", path, exc)
        return [f"class_{idx}" for idx in range(num_actions)]

    def _load_federated_parameters(
        self, parameters: Union[List[np.ndarray], fl.common.Parameters]
    ) -> None:
        if isinstance(parameters, list):
            ndarrays = parameters
        else:
            ndarrays = fl.common.parameters_to_ndarrays(parameters)

        expected_min = len(self.prior_keys) + len(self.recog_keys) + len(self.main_keys)
        expected_with_gen = expected_min + len(self.generator_keys)
        if len(ndarrays) not in (expected_min, expected_with_gen):
            raise ValueError(
                f"Parameter length mismatch. Expected {expected_min} or {expected_with_gen}, "
                f"received {len(ndarrays)}"
            )

        cursor = 0
        prior_slice = ndarrays[cursor : cursor + len(self.prior_keys)]
        cursor += len(self.prior_keys)
        recog_slice = ndarrays[cursor : cursor + len(self.recog_keys)]
        cursor += len(self.recog_keys)
        main_slice = ndarrays[cursor : cursor + len(self.main_keys)]
        cursor += len(self.main_keys)

        prior_tensors = [
            torch.tensor(arr, device=self.device, dtype=self.prior_net.state_dict()[key].dtype)
            for arr, key in zip(prior_slice, self.prior_keys)
        ]
        recog_tensors = [
            torch.tensor(arr, device=self.device, dtype=self.recognition_net.state_dict()[key].dtype)
            for arr, key in zip(recog_slice, self.recog_keys)
        ]
        main_tensors = [
            torch.tensor(arr, device=self.device, dtype=self.main_q_net.state_dict()[key].dtype)
            for arr, key in zip(main_slice, self.main_keys)
        ]

        self.prior_net.load_state_dict(dict(zip(self.prior_keys, prior_tensors)))
        self.recognition_net.load_state_dict(dict(zip(self.recog_keys, recog_tensors)))
        self.main_q_net.load_state_dict(dict(zip(self.main_keys, main_tensors)))

        if len(ndarrays) == expected_with_gen:
            gen_slice = ndarrays[cursor: cursor + len(self.generator_keys)]
            gen_tensors = [
                torch.tensor(arr, device=self.device, dtype=self.generation_net.state_dict()[key].dtype)
                for arr, key in zip(gen_slice, self.generator_keys)
            ]
            self.generation_net.load_state_dict(dict(zip(self.generator_keys, gen_tensors)))

    def _run_inference(self) -> Tuple[float, np.ndarray, np.ndarray]:
        assert self.loader is not None
        self.prior_net.eval()
        self.main_q_net.eval()

        total_loss = 0.0
        total_samples = 0
        all_true: List[int] = []
        all_pred: List[int] = []

        with torch.no_grad():
            for features, labels in self.loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                mu_p, _ = self.prior_net(features)
                q_values = self.main_q_net(mu_p, features)
                loss = F.cross_entropy(q_values, labels, reduction="mean")

                preds = q_values.argmax(dim=1)
                all_true.extend(labels.cpu().numpy().tolist())
                all_pred.extend(preds.cpu().numpy().tolist())

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples else 0.0
        return avg_loss, np.array(all_true), np.array(all_pred)

    def _save_classification_report(self, server_round: int, report: str) -> None:
        report_path = self.figure_dir / f"closed_set_report_round_{server_round:03d}.txt"
        try:
            report_path.write_text(report)
        except Exception as exc:
            logger.warning("Failed to save classification report: %s", exc)

    def _plot_confusion_matrix(self, server_round: int, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        labels = list(range(len(self.class_names)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(cm_norm, cmap=plt.get_cmap("Blues"), vmin=0, vmax=1)
        ax.set_xticks(range(len(self.class_names)))
        ax.set_yticks(range(len(self.class_names)))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_yticklabels(self.class_names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Closed-set Confusion Matrix (Round {server_round})")

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
        fig_path = self.figure_dir / f"closed_set_cm_round_{server_round:03d}.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)

    def __call__(
        self,
        server_round: int,
        parameters: Union[List[np.ndarray], fl.common.Parameters],
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[float, Dict[str, float]]:
        if not self.enabled or self.loader is None:
            return 0.0, {}

        self._load_federated_parameters(parameters)
        loss, y_true, y_pred = self._run_inference()
        accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0

        report = classification_report(
            y_true, y_pred, target_names=self.class_names, digits=4, zero_division=0
        )
        logger.info("Closed-set evaluation (round %s):\n%s", server_round, report)
        self._save_classification_report(server_round, report)
        self._plot_confusion_matrix(server_round, y_true, y_pred)

        return float(loss), {"accuracy": accuracy}


def get_strategy(cfg: DictConfig, evaluator: Optional[ClosedSetEvaluator]) -> Strategy:
    """Create the server's federated learning strategy."""
    proximal_mu = float(cfg.server.get("proximal_mu", 0.1))
    strategy = FedProx(
        fraction_fit=cfg.server.fraction_fit,
        fraction_evaluate=cfg.server.fraction_evaluate,
        min_fit_clients=cfg.server.min_fit_clients,
        min_evaluate_clients=cfg.server.min_evaluate_clients,
        min_available_clients=cfg.server.min_available_clients,
        on_fit_config_fn=fit_config_fn,
        evaluate_fn=evaluator if (evaluator and evaluator.enabled) else None,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        proximal_mu=proximal_mu,
    )

    logger.info(
        "Server strategy: FedProx(mu=%s) with closed-set evaluation=%s",
        proximal_mu,
        bool(evaluator and evaluator.enabled),
    )
    return strategy


def run_server(cfg: DictConfig, device: Optional[torch.device] = None) -> None:
    """Start the Flower server."""
    reset_reward_history()
    evaluator = ClosedSetEvaluator(cfg, device=device)
    strategy = get_strategy(cfg, evaluator)

    logger.info("Starting server at %s for %s rounds.", cfg.server.address, cfg.server.num_rounds)
    try:
        fl.server.start_server(
            server_address=cfg.server.address,
            config=fl.server.ServerConfig(num_rounds=cfg.server.num_rounds),
            strategy=strategy,
        )
    except RuntimeError as exc:
        if "Failed to bind to address" in str(exc):
            logger.error(
                "Unable to bind to %s. The port is already in use. "
                "Update 'cfg.server.address' or free the port before retrying.",
                cfg.server.address,
            )
            raise SystemExit(1) from exc
        raise


def plot_reward_history(cfg: DictConfig) -> None:
    """Plot the aggregated reward per round and save it to the figures directory."""
    history = get_reward_history()
    if not history:
        logger.warning("No reward history collected; skipping reward plot.")
        return

    figure_dir = _resolve_path(cfg.paths.figures_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rounds, rewards = zip(*sorted(history, key=lambda item: item[0]))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, rewards, marker="o", color="#2a9d8f", linewidth=2)
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Average Reward per Episode")
    ax.set_title("Client Reward Across Federated Rounds")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_path = figure_dir / "federated_rewards.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    logger.info("Saved reward history plot to %s", plot_path)
