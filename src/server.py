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
from flwr.server.strategy import FedProx, Strategy
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

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

    metric_keys = [
        # RL Metrics
        "avg_reward_per_episode",
        "avg_td_loss",
        "avg_kl_loss",
        "avg_q_value",
        # Local (Pre-Aggregation) Closed-Set
        "local_loss",
        "local_accuracy",
        "local_f1_macro",
        # Local (Pre-Aggregation) Open-Set
        "openset_f1_macro",
        "openset_auroc",
        "openset_unknown_recall",
        "openset_known_acc",
        "openset_overall_acc",
        # Generator
        "generator_loss",
        "generator_correct_frac",
    ]

    # Log individual client metrics before aggregation
    logger.info("-" * 60)
    logger.info("PER-CLIENT METRICS")
    logger.info("-" * 60)
    for num_examples, metrics in fit_metrics:
        client_id = metrics.get("cid", "unknown")
        msg_parts = [f"client={client_id}", f"samples={num_examples}"]
        for key in ("local_accuracy", "local_f1_macro", "local_loss"):
            if key in metrics:
                msg_parts.append(f"{key}={metrics[key]:.4f}")
        for key in ("openset_auroc", "openset_unknown_recall", "openset_known_acc"):
            if key in metrics:
                msg_parts.append(f"{key}={metrics[key]:.4f}")
        logger.info(" | ".join(msg_parts))
    logger.info("-" * 60)

    for key in metric_keys:
        weighted_sum = 0.0
        present = False
        for num_examples, metrics in fit_metrics:
            metric_val = metrics.get(key)
            if metric_val is None:
                continue
            try:
                metric_float = float(metric_val)
            except (TypeError, ValueError):
                continue
            weighted_sum += metric_float * num_examples
            present = True
        if present:
            aggregated[key] = weighted_sum / total_examples

    logger.info("=" * 60)
    logger.info("AGGREGATED METRICS (Weighted Avg)")
    logger.info("=" * 60)
    if "local_accuracy" in aggregated:
        logger.info("Local (Pre-Agg) Accuracy    : %.4f", aggregated["local_accuracy"])
    if "local_f1_macro" in aggregated:
        logger.info("Local (Pre-Agg) F1 Macro    : %.4f", aggregated["local_f1_macro"])
    if "local_loss" in aggregated:
        logger.info("Local (Pre-Agg) Loss        : %.4f", aggregated["local_loss"])

    if "openset_auroc" in aggregated:
        logger.info("-" * 46)
        logger.info("Open-Set AUROC              : %.4f", aggregated["openset_auroc"])
        if "openset_f1_macro" in aggregated:
            logger.info("Open-Set F1 Macro           : %.4f", aggregated["openset_f1_macro"])
        if "openset_unknown_recall" in aggregated:
            logger.info(
                "Open-Set Unknown Recall     : %.4f",
                aggregated["openset_unknown_recall"],
            )
        if "openset_known_acc" in aggregated:
            logger.info("Open-Set Known Accuracy     : %.4f", aggregated["openset_known_acc"])
        if "openset_overall_acc" in aggregated:
            logger.info("Open-Set Overall Accuracy   : %.4f", aggregated["openset_overall_acc"])

    if "generator_loss" in aggregated or "generator_correct_frac" in aggregated:
        logger.info("-" * 46)
        if "generator_loss" in aggregated:
            logger.info("Generator Loss              : %.4f", aggregated["generator_loss"])
        if "generator_correct_frac" in aggregated:
            logger.info(
                "Generator Correct Fraction  : %.4f",
                aggregated["generator_correct_frac"],
            )
    logger.info("=" * 60)

    avg_reward = aggregated.get("avg_reward_per_episode")
    if avg_reward is not None:
        REWARD_HISTORY.append((len(REWARD_HISTORY) + 1, avg_reward))

    return aggregated


def get_strategy(cfg: DictConfig) -> Strategy:
    """Create the server's federated learning strategy."""
    proximal_mu = float(cfg.server.get("proximal_mu", 0.1))
    strategy = FedProx(
        fraction_fit=cfg.server.fraction_fit,
        fraction_evaluate=cfg.server.fraction_evaluate,
        min_fit_clients=cfg.server.min_fit_clients,
        min_evaluate_clients=cfg.server.min_evaluate_clients,
        min_available_clients=cfg.server.min_available_clients,
        on_fit_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        proximal_mu=proximal_mu,
    )

    logger.info("Server strategy: FedProx(mu=%s)", proximal_mu)
    return strategy


def run_server(cfg: DictConfig, device: Optional[torch.device] = None) -> None:
    """Start the Flower server."""
    reset_reward_history()
    strategy = get_strategy(cfg)

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
