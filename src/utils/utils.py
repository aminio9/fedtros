import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --- VAE Constants ---
LOGVAR_MIN, LOGVAR_MAX = -6.0, 2.0
EPS = 1e-12

# Cached device so the whole app uses a single, consistent target.
_GLOBAL_DEVICE: torch.device | None = None
_DEVICE_ENV_VARS = ("FEDOSQ_DEVICE", "DEVICE", "TORCH_DEVICE")


def project_root() -> Path:
    """Return the repository root, preserving Hydra's original cwd when available."""
    try:
        from hydra.utils import get_original_cwd

        return Path(get_original_cwd()).resolve()
    except Exception:
        return Path(__file__).resolve().parents[2]


def set_seed(
    seed: int,
    deterministic: bool = True,
    *,
    benchmark: bool = False,
    use_deterministic_algorithms: bool = False,
) -> dict:
    """Set Python, NumPy, PyTorch, and CUDA seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = bool(benchmark)
        if use_deterministic_algorithms:
            torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = bool(benchmark)
    logging.info(f"Global random seed set to {seed}")
    return {
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "benchmark": bool(benchmark),
        "use_deterministic_algorithms": bool(use_deterministic_algorithms),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's initial seed."""
    _ = worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """Create a seeded torch.Generator for deterministic DataLoader shuffling."""
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    return generator


def _normalize_device_preference(prefer: str | None) -> str | None:
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
    prefer: str | None = None,
    *,
    allow_cpu_fallback: bool = False,
    cache: bool = True,
    logger: logging.Logger | None = None,
) -> torch.device:
    """
    Get the torch device to use across the project.

    - Honors env vars: FEDOSQ_DEVICE / DEVICE / TORCH_DEVICE
    - Caches the first resolved device so every module shares it
    - Fails fast instead of silently falling back from CUDA to CPU
    """
    global _GLOBAL_DEVICE
    active_logger = logger or logging.getLogger(__name__)

    # Resolve preference: config/arg -> env -> auto
    prefer = _normalize_device_preference(prefer)
    if prefer is None:
        for env_key in _DEVICE_ENV_VARS:
            env_val = os.getenv(env_key)
            if env_val:
                prefer = _normalize_device_preference(env_val)
                break

    # Reuse cached device if it matches the preference
    if cache and _GLOBAL_DEVICE is not None and _device_matches_preference(_GLOBAL_DEVICE, prefer):
        return _GLOBAL_DEVICE

    # Choose device
    device: torch.device
    if prefer == "cpu":
        device = torch.device("cpu")
    elif prefer == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            _log_cuda_device(active_logger, device)
        # elif torch_directml is not None:
        #     logger.warning("CUDA requested but unavailable; trying DirectML as fallback.")
        #     device = _resolve_directml_device(logger, allow_cpu_fallback)
        elif allow_cpu_fallback:
            active_logger.warning("CUDA requested but unavailable; falling back to CPU.")
            device = torch.device("cpu")
        else:
            raise RuntimeError(
                "CUDA requested but not available. Install a CUDA-enabled PyTorch build, "
                "check NVIDIA drivers, or run with an explicit CPU runtime only for non-training jobs."
            )
    elif prefer == "directml":
        device = _resolve_directml_device(active_logger, allow_cpu_fallback)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            _log_cuda_device(active_logger, device)
        # elif torch_directml is not None:
        #     device = _resolve_directml_device(logger, allow_cpu_fallback)
        else:
            if allow_cpu_fallback:
                active_logger.warning("No accelerator detected; falling back to CPU.")
                device = torch.device("cpu")
            else:
                raise RuntimeError(
                    "No CUDA device detected and no explicit CPU runtime was requested."
                )

    if cache:
        _GLOBAL_DEVICE = device

    active_logger.info("Using device: %s", device)
    return device


def _resolve_directml_device(logger: logging.Logger, allow_cpu_fallback: bool) -> torch.device:
    """Try to return a DirectML device if torch-directml is installed."""
    try:
        import torch_directml  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        torch_directml = None  # type: ignore

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


def _log_cuda_device(logger: logging.Logger, device: torch.device) -> None:
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    logger.info(
        "Using CUDA device %d: %s | torch=%s | cuda=%s | total_memory_gb=%.2f",
        index,
        props.name,
        torch.__version__,
        torch.version.cuda,
        props.total_memory / (1024**3),
    )


def set_device(device: str | torch.device) -> torch.device:
    """Force-set the global device cache (useful for entrypoints/tests)."""
    global _GLOBAL_DEVICE
    _GLOBAL_DEVICE = torch.device(device)
    logging.getLogger(__name__).info("Global device override set to %s", _GLOBAL_DEVICE)
    return _GLOBAL_DEVICE


def resolve_device_from_config(cfg: object | None) -> torch.device:
    """
    Resolve the device using a config object (expects cfg.device.prefer/allow_cpu_fallback).
    Falls back to env vars and auto-detection.
    """
    device_cfg = getattr(cfg, "device", None)
    prefer = None
    allow_cpu_fallback = False
    if device_cfg is not None:
        prefer = getattr(device_cfg, "prefer", None) or getattr(device_cfg, "preference", None)
        allow_cpu_fallback = bool(getattr(device_cfg, "allow_cpu_fallback", True))
    return get_device(prefer=prefer, allow_cpu_fallback=allow_cpu_fallback, cache=True)


def _device_matches_preference(device: torch.device, prefer: str | None) -> bool:
    if prefer is None:
        return True
    if prefer == "directml":
        dev_str = str(device).lower()
        return device.type in {"dml", "directml", "privateuseone"} or dev_str.startswith(
            ("dml", "directml")
        )
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
    return oh_cpu.view(*safe_idx.shape, num_classes).to(device=target_device)


def soft_update_target_network(main_net: nn.Module, target_net: nn.Module, tau: float):
    """Soft update target network parameters with a tau-weighted moving average."""
    with torch.no_grad():
        for tp, p in zip(target_net.parameters(), main_net.parameters(), strict=True):
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
