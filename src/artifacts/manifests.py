"""Provenance and run manifest generation utilities for FedTROS-PR."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def compute_file_hash(path: Path) -> str:
    """Compute sha256 checksum for a file."""
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dict_hash(data: dict[str, Any] | DictConfig) -> str:
    """Compute sha256 hash of JSON-serialized dictionary."""
    try:
        if isinstance(data, DictConfig):
            container = OmegaConf.to_container(data, resolve=True)
        else:
            container = data
        payload = json.dumps(container, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    except Exception:
        return ""


def get_git_commit_hash(project_root: Path) -> str:
    """Retrieve current git commit hash if in a git repository."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return out
    except Exception:
        return "unknown_commit"


def get_hardware_info() -> dict[str, Any]:
    """Capture hardware environment metadata."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "processor": platform.processor(),
    }


def create_run_manifest(
    cfg: DictConfig,
    *,
    output_dir: Path,
    project_root: Path,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create and write canonical run_manifest.json and environment provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    git_commit = get_git_commit_hash(project_root)
    config_container = OmegaConf.to_container(cfg, resolve=True)
    config_hash = compute_dict_hash(config_container)

    split_manifest_path = output_dir / "split_manifest.csv"
    split_hash = compute_file_hash(split_manifest_path) if split_manifest_path.exists() else ""

    manifest = {
        "schema_version": 2,
        "method": str(OmegaConf.select(cfg, "experiment.method", default="FedTROS-PR")),
        "experiment_id": str(OmegaConf.select(cfg, "experiment.name", default="exp")),
        "dataset": str(OmegaConf.select(cfg, "dataset.name", default="b_nat")),
        "known_classes": list(OmegaConf.select(cfg, "dataset.preprocessing.known_labels", default=[])),
        "unknown_labels": list(OmegaConf.select(cfg, "dataset.preprocessing.unknown_labels", default=[])),
        "num_clients": int(OmegaConf.select(cfg, "federated.num_clients", default=10)),
        "alpha": float(OmegaConf.select(cfg, "dataset.preprocessing.alpha", default=0.1)),
        "seed": int(OmegaConf.select(cfg, "seed", default=42)),
        "partition_seed": int(OmegaConf.select(cfg, "seed", default=42)),
        "git_commit": git_commit,
        "config_hash": config_hash,
        "split_hash": split_hash,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": get_hardware_info(),
        "metrics_summary": metrics or {},
    }

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "git_commit.txt").write_text(git_commit + "\n", encoding="utf-8")

    logger.info("Created run manifest: %s (commit=%s, config_hash=%s)", manifest_path, git_commit[:8], config_hash[:8])
    return manifest
