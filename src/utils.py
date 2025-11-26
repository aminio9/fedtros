import logging
import logging.config
import logging.handlers
import sys
import random
import os
from pathlib import Path
from typing import Optional, List, OrderedDict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# try:
# import torch_directml  # type: ignore
# except ImportError:  # pragma: no cover - optional dependency
torch_directml = None  # type: ignore

# --- VAE Constants ---
LOGVAR_MIN, LOGVAR_MAX = -6.0, 2.0
EPS = 1e-12

# Cached device so the whole app uses a single, consistent target.
_GLOBAL_DEVICE: Optional[torch.device] = None
_DEVICE_ENV_VARS = ("FEDOSQ_DEVICE", "DEVICE", "TORCH_DEVICE")


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


def _normalize_device_preference(prefer: Optional[str]) -> Optional[str]:
    """Map various user inputs to canonical device strings."""
    if prefer is None:
        return None
    prefer = str(prefer).strip().lower()
    if prefer in {"auto", ""}:
        return None
    if prefer in {"cuda", "gpu"}:
        return "cuda"
    if prefer == "cpu":
        return "cpu"
    if prefer in {"dml", "directml"}:
        return "directml"
    return prefer


def get_device(
    prefer: Optional[str] = None,
    *,
    allow_cpu_fallback: bool = True,
    cache: bool = True,
) -> torch.device:
    """
    Get the torch device to use across the project.

    - Honors env vars: FEDOSQ_DEVICE / DEVICE / TORCH_DEVICE
    - Caches the first resolved device so every module shares it
    - Optionally refuses to fall back to CPU if CUDA is requested
    """
    global _GLOBAL_DEVICE
    logger = logging.getLogger(__name__)

    # Resolve preference: config/arg -> env -> auto
    prefer = _normalize_device_preference(prefer)
    if prefer is None:
        for env_key in _DEVICE_ENV_VARS:
            env_val = os.getenv(env_key)
            if env_val:
                prefer = _normalize_device_preference(env_val)
                break

    # Reuse cached device if it matches the preference
    if cache and _GLOBAL_DEVICE is not None:
        if _device_matches_preference(_GLOBAL_DEVICE, prefer):
            return _GLOBAL_DEVICE

    # Choose device
    device: torch.device
    if prefer == "cpu":
        device = torch.device("cpu")
    elif prefer == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA device (torch %s)", torch.__version__)
        elif torch_directml is not None:
            logger.warning("CUDA requested but unavailable; trying DirectML as fallback.")
            device = _resolve_directml_device(logger, allow_cpu_fallback)
        elif allow_cpu_fallback:
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            device = torch.device("cpu")
        else:
            raise RuntimeError("CUDA requested but not available. Set allow_cpu_fallback=True to fall back to CPU.")
    elif prefer == "directml":
        device = _resolve_directml_device(logger, allow_cpu_fallback)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Auto-selected CUDA device (torch %s)", torch.__version__)
        elif torch_directml is not None:
            device = _resolve_directml_device(logger, allow_cpu_fallback)
        else:
            device = torch.device("cpu")

    if cache:
        _GLOBAL_DEVICE = device

    logger.info("Using device: %s", device)
    return device


def _resolve_directml_device(logger: logging.Logger, allow_cpu_fallback: bool) -> torch.device:
    """Try to return a DirectML device if torch-directml is installed."""
    if torch_directml is None:
        if allow_cpu_fallback:
            logger.warning("torch-directml not installed; falling back to CPU.")
            return torch.device("cpu")
        raise RuntimeError("DirectML requested but torch-directml is not installed.")

    try:
        device = torch_directml.device()
        name = None
        # torch-directml exposes adapter introspection on some builds; guard calls.
        try:
            idx = device.index if getattr(device, "index", None) is not None else 0
        except Exception:
            idx = 0
        try:
            if hasattr(torch_directml, "device_name"):
                name = torch_directml.device_name(idx)  # type: ignore[attr-defined,arg-type]
            elif hasattr(torch_directml, "get_device_name"):
                name = torch_directml.get_device_name(idx)  # type: ignore[attr-defined,arg-type]
        except Exception:
            name = None
        logger.info(
            "Using DirectML device: %s%s (torch-directml %s, torch %s)",
            device,
            f" / {name}" if name else "",
            getattr(torch_directml, "__version__", "unknown"),
            torch.__version__,
        )
        return device
    except Exception as exc:
        if allow_cpu_fallback:
            logger.warning("Failed to initialize DirectML (%s); falling back to CPU.", exc)
            return torch.device("cpu")
        raise RuntimeError(f"DirectML requested but failed to initialize: {exc}") from exc


def set_device(device: Union[str, torch.device]) -> torch.device:
    """Force-set the global device cache (useful for entrypoints/tests)."""
    global _GLOBAL_DEVICE
    _GLOBAL_DEVICE = torch.device(device)
    logging.getLogger(__name__).info("Global device override set to %s", _GLOBAL_DEVICE)
    return _GLOBAL_DEVICE


def resolve_device_from_config(cfg: Optional[object]) -> torch.device:
    """
    Resolve the device using a config object (expects cfg.device.prefer/allow_cpu_fallback).
    Falls back to env vars and auto-detection.
    """
    device_cfg = getattr(cfg, "device", None)
    prefer = None
    allow_cpu_fallback = True
    if device_cfg is not None:
        prefer = getattr(device_cfg, "prefer", None) or getattr(device_cfg, "preference", None)
        allow_cpu_fallback = bool(getattr(device_cfg, "allow_cpu_fallback", True))
    return get_device(prefer=prefer, allow_cpu_fallback=allow_cpu_fallback, cache=True)


def _device_matches_preference(device: torch.device, prefer: Optional[str]) -> bool:
    if prefer is None:
        return True
    if prefer == "directml":
        dev_str = str(device).lower()
        return device.type in {"dml", "directml", "privateuseone"} or dev_str.startswith(("dml", "directml"))
    return device.type == prefer


def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert a tensor of indices to a one-hot representation."""
    target_device = indices.device
    if indices.dim() > 1:
        indices = indices.squeeze(-1)
    safe_idx = indices.to(dtype=torch.long)

    # Mask valid classes
    valid = (safe_idx >= 0) & (safe_idx < num_classes)
    safe_idx = safe_idx.clamp(min=0, max=max(num_classes - 1, 0))

    # Build one-hot on CPU to avoid unsupported scatter on DirectML, then move back.
    flat = safe_idx.view(-1).cpu()
    oh_cpu = torch.zeros(flat.numel(), num_classes, dtype=torch.float32, device="cpu")
    if flat.numel() > 0 and num_classes > 0:
        oh_cpu[torch.arange(flat.numel()), flat] = 1.0
    oh_cpu[~valid.view(-1).cpu()] = 0.0
    oh = oh_cpu.view(*safe_idx.shape, num_classes).to(device=target_device)
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
