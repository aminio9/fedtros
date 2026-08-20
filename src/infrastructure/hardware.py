"""Hardware, host, and software environment capture utilities."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import socket
from typing import Any

import torch


def get_gpu_info() -> dict[str, Any]:
    """Capture detailed GPU and CUDA device information."""
    cuda_available = bool(torch.cuda.is_available())
    gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
    devices: list[dict[str, Any]] = []

    if cuda_available:
        for idx in range(gpu_count):
            dev_info: dict[str, Any] = {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
            }
            try:
                props = torch.cuda.get_device_properties(idx)
                dev_info["total_memory_gb"] = round(props.total_memory / (1024**3), 2)
                dev_info["major"] = props.major
                dev_info["minor"] = props.minor
                dev_info["multi_processor_count"] = getattr(props, "multi_processor_count", None)
                # Attempt to get GPU UUID if supported by PyTorch / driver
                if hasattr(props, "uuid"):
                    dev_info["uuid"] = str(props.uuid)
                else:
                    dev_info["uuid"] = f"GPU-{idx:04d}"
            except Exception:
                dev_info["uuid"] = f"GPU-{idx:04d}"
            devices.append(dev_info)

    return {
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available and hasattr(torch.backends, "cudnn") else None,
        "device_count": gpu_count,
        "devices": devices,
    }


def get_cpu_and_ram_info() -> dict[str, Any]:
    """Capture CPU architecture, core counts, and system RAM."""
    cpu_info: dict[str, Any] = {
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count() or 1,
        "cpu_count_physical": None,
        "ram_total_gb": None,
        "ram_available_gb": None,
    }
    try:
        import psutil

        cpu_info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        cpu_info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
        cpu_info["ram_total_gb"] = round(mem.total / (1024**3), 2)
        cpu_info["ram_available_gb"] = round(mem.available / (1024**3), 2)
    except Exception:
        pass

    return cpu_info


def get_installed_packages(package_names: tuple[str, ...] | None = None) -> dict[str, str]:
    """Return dictionary of installed package versions."""
    if package_names is None:
        package_names = (
            "torch",
            "flwr",
            "hydra-core",
            "omegaconf",
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
            "wandb",
            "joblib",
            "psutil",
        )

    versions: dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except Exception:
            versions[name] = "not_installed"
    return versions


def get_full_environment_provenance() -> dict[str, Any]:
    """Capture complete host, hardware, and software environment manifest."""
    return {
        "hostname": socket.gethostname(),
        "os_platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
        },
        "cpu_ram": get_cpu_and_ram_info(),
        "gpu": get_gpu_info(),
        "packages": get_installed_packages(),
    }
