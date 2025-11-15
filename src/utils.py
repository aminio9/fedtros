import logging
import logging.config
import logging.handlers
import sys
import random
import os
from pathlib import Path
from typing import Optional, List, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- VAE Constants ---
LOGVAR_MIN, LOGVAR_MAX = -6.0, 2.0
EPS = 1e-12


def setup_logging(
    log_file_path: str,
    log_level: str = "INFO",
    max_bytes: int = 10_485_760,  # 10 MB
    backup_count: int = 5,
) -> None:
    """
    Set up advanced, production-ready logging using dictConfig.

    Features:
    - Separate formats for console (simple) and file (detailed).
    - Rotating file handler to manage log file size.
    - Configurable log level.
    """
    log_file_path = str(Path(log_file_path).resolve())
    log_dir = str(Path(log_file_path).parent)
    os.makedirs(log_dir, exist_ok=True)

    # Ensure log_level is a valid string
    if log_level.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        print(f"Invalid log level '{log_level}', defaulting to 'INFO'")
        log_level = "INFO"

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,  # Keep 3rd-party loggers (e.g., flwr)
        "formatters": {
            "console_simple": {"format": "%(levelname)-8s | %(name)-12s | %(message)s"},
            "file_detailed": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)-20s | %(filename)s:%(lineno)d | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "console_simple",
                "level": log_level,
            },
            "file_rotating": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_file_path,
                "maxBytes": max_bytes,
                "backupCount": backup_count,
                "formatter": "file_detailed",
                "level": log_level,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file_rotating"],
                "level": log_level,
                "propagate": True,
            },
            "flwr": {  # Example: Set Flower's logger to be less noisy
                "level": "INFO",
                "propagate": True,
            },
        },
    }

    try:
        logging.config.dictConfig(LOGGING_CONFIG)
        logging.info(f"Logging configured. Level: {log_level}, File: {log_file_path}")
    except Exception as e:
        # Fallback to basic logging if dictConfig fails
        print(f"CRITICAL: Failed to configure logging: {e}", file=sys.stderr)
        logging.basicConfig(level=logging.INFO)


def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Best-effort determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Global random seed set to {seed}")


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Get the appropriate torch device, logging the choice."""
    logger = logging.getLogger(__name__)
    if prefer == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert a tensor of indices to a one-hot representation."""
    if indices.dim() > 1:
        indices = indices.squeeze(-1)
    indices = indices.to(dtype=torch.long)

    # Mask valid classes
    valid = (indices >= 0) & (indices < num_classes)
    # Replace invalid with 0 for one_hot, then zero them out
    safe_idx = indices.clone()
    safe_idx[~valid] = 0

    oh = F.one_hot(safe_idx, num_classes=num_classes).to(dtype=torch.float32)
    # Make rows for invalid indices all zeros
    if (~valid).any():
        oh[~valid] = 0.0
    return oh


def soft_update_target_network(main_net: nn.Module, target_net: nn.Module, tau: float):
    """Soft update target network parameters. θ_target = τ*θ_policy + (1 - τ)*θ_target"""
    with torch.no_grad():
        for tp, p in zip(target_net.parameters(), main_net.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def reparameterization_trick(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """z = mu + sigma * eps with log-variance clamped for stability."""
    log_var = log_var.clamp(LOGVAR_MIN, LOGVAR_MAX)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + std * eps


def calculate_kl_divergence(
    mu_q: torch.Tensor,
    log_var_q: torch.Tensor,
    mu_p: torch.Tensor,
    log_var_p: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """Calculate KL divergence KL(q||p) with log-variance clamping."""
    log_var_q = log_var_q.clamp(LOGVAR_MIN, LOGVAR_MAX)
    log_var_p = log_var_p.clamp(LOGVAR_MIN, LOGVAR_MAX)

    var_q = torch.exp(log_var_q)
    var_p = torch.exp(log_var_p)

    term_log = log_var_p - log_var_q
    term_frac = (var_q + (mu_q - mu_p) ** 2) / (var_p + EPS)
    kl = 0.5 * (term_log + term_frac - 1.0)
    kl = kl.sum(dim=1)  # [B]

    if reduce == "none":
        return kl
    if reduce == "sum":
        return kl.sum()
    return kl.mean()  # 'mean'


def calculate_kl_divergence_raw(
    mu_q: torch.Tensor,
    log_var_q: torch.Tensor,
    mu_p: torch.Tensor,
    log_var_p: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """Calculate KL divergence KL(q||p) WITHOUT log-variance clamping."""
    var_q = torch.exp(log_var_q)
    var_p = torch.exp(log_var_p)
    term_log = log_var_p - log_var_q
    term_frac = (var_q + (mu_q - mu_p) ** 2) / (var_p + EPS)
    kl = 0.5 * (term_log + term_frac - 1.0)
    kl = kl.sum(dim=1)  # [B]
    if reduce == "none":
        return kl
    if reduce == "sum":
        return kl.sum()
    return kl.mean()
