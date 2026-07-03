from __future__ import annotations

import csv
<<<<<<< HEAD
=======
import hashlib
>>>>>>> ea28efe (Initial commit with updated source code)
import json
import logging
import logging.config
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.config import resolve_path


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def configure_run_logging(
    cfg: DictConfig,
    project_root: Path,
    run_dir: Path,
    *,
    log_dir_override: Path | None = None,
) -> None:
    """Configure console, run log, and debug log handlers exactly once per process."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_level = str(cfg.logging.level).upper()
    debug_level = str(cfg.logging.debug_level).upper()
    third_party_level = str(cfg.logging.third_party_level).upper()

    if log_dir_override is None:
        run_log = resolve_path(project_root, cfg.logging.run_log)
        debug_log = resolve_path(project_root, cfg.logging.debug_log)
    else:
        run_log = log_dir_override / "run.log"
        debug_log = log_dir_override / "debug.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    debug_log.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": str(cfg.logging.format),
                    "datefmt": str(cfg.logging.datefmt),
                },
                "console": {"format": "%(levelname)-8s | %(name)-20s | %(message)s"},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "console",
                    "stream": "ext://sys.stdout",
                },
                "run_file": {
                    "class": "logging.FileHandler",
                    "level": log_level,
                    "formatter": "standard",
                    "filename": str(run_log),
                    "encoding": "utf-8",
                },
                "debug_file": {
                    "class": "logging.FileHandler",
                    "level": debug_level,
                    "formatter": "standard",
                    "filename": str(debug_log),
                    "encoding": "utf-8",
                },
            },
            "root": {
                "handlers": ["console", "run_file", "debug_file"],
                "level": debug_level,
            },
            "loggers": {
                "flwr": {"level": third_party_level, "propagate": True},
                "matplotlib": {"level": "WARNING", "propagate": True},
                "PIL": {"level": "WARNING", "propagate": True},
            },
        }
    )


@dataclass
class LocalRunTracker:
    """Small local-only experiment tracker using JSONL, CSV, logs, and metadata."""

    cfg: DictConfig
    project_root: Path
    run_dir: Path
    script_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def metrics_jsonl_path(self) -> Path:
        return resolve_path(self.project_root, self.cfg.tracking.metrics_jsonl)

    @property
    def metrics_csv_path(self) -> Path:
        return resolve_path(self.project_root, self.cfg.tracking.metrics_csv)

    def start(self, device: torch.device | str | None = None) -> LocalRunTracker:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        configure_run_logging(self.cfg, self.project_root, self.run_dir)

        timestamp = datetime.now(timezone.utc).isoformat()
<<<<<<< HEAD
=======
        resolved_config_yaml = OmegaConf.to_yaml(self.cfg, resolve=True)
        config_hash = hashlib.sha256(resolved_config_yaml.encode("utf-8")).hexdigest()
>>>>>>> ea28efe (Initial commit with updated source code)
        self.metadata.update(
            {
                "experiment_name": str(self.cfg.experiment.name),
                "run_id": str(self.cfg.tracking.run_id),
                "timestamp_utc": timestamp,
<<<<<<< HEAD
=======
                "config_sha256": config_hash,
                "config_sha256_short": config_hash[:12],
>>>>>>> ea28efe (Initial commit with updated source code)
                "seed": int(self.cfg.seed),
                "device": str(device) if device is not None else str(self.cfg.device.prefer),
                "script": self.script_name,
                "git_commit": _git_commit(self.project_root),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "cuda_device_name": (
                    torch.cuda.get_device_name(torch.cuda.current_device())
                    if torch.cuda.is_available()
                    else None
                ),
                "dataset": str(self.cfg.dataset.name),
                "model": str(self.cfg.model.name),
                "method": str(self.cfg.experiment.method),
<<<<<<< HEAD
=======
                "known_labels": list(self.cfg.dataset.preprocessing.known_labels),
                "unknown_label_id": int(self.cfg.dataset.preprocessing.unknown_label_id),
                "state_dim": int(self.cfg.model.state_dim),
                "latent_dim": int(self.cfg.model.latent_dim),
                "num_actions": int(self.cfg.model.num_actions),
                "optimizer": OmegaConf.to_container(self.cfg.optimizer, resolve=True),
                "scheduler": OmegaConf.to_container(self.cfg.scheduler, resolve=True),
                "loss_weights": OmegaConf.to_container(
                    self.cfg.training.loss_weights,
                    resolve=True,
                ),
                "reward": OmegaConf.to_container(self.cfg.training.reward, resolve=True),
                "evt": OmegaConf.to_container(self.cfg.open_set.evt, resolve=True),
>>>>>>> ea28efe (Initial commit with updated source code)
            }
        )
        self.save_config()
        self.save_metadata()
        logging.getLogger(__name__).info(
            "Initialized local run | experiment=%s | run_id=%s | dir=%s",
            self.metadata["experiment_name"],
            self.metadata["run_id"],
            self.run_dir,
        )
        return self

    def save_config(self) -> None:
        raw_path = resolve_path(self.project_root, self.cfg.tracking.config_yaml)
        resolved_path = resolve_path(self.project_root, self.cfg.tracking.resolved_config_yaml)
        raw_path.write_text(OmegaConf.to_yaml(self.cfg, resolve=False), encoding="utf-8")
        resolved_path.write_text(OmegaConf.to_yaml(self.cfg, resolve=True), encoding="utf-8")

    def save_metadata(self) -> None:
        path = resolve_path(self.project_root, self.cfg.tracking.metadata_json)
        path.write_text(json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8")

    def log_metrics(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        record: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **({"step": int(step)} if step is not None else {}),
            **metrics,
        }
        self.metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._rewrite_metrics_csv()

    def write_json(self, filename: str, payload: dict[str, Any]) -> Path:
        path = self.run_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        return path

    def _rewrite_metrics_csv(self) -> None:
        records = []
        with open(self.metrics_jsonl_path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        if not records:
            return
        fieldnames: list[str] = []
        for record in records:
            for key in record:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(self.metrics_csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)


def initialize_run(
    cfg: DictConfig,
    *,
    project_root: Path,
    script_name: str,
    device: torch.device | str | None = None,
) -> LocalRunTracker:
    run_dir = resolve_path(project_root, cfg.tracking.run_dir)
    tracker = LocalRunTracker(
        cfg=cfg, project_root=project_root, run_dir=run_dir, script_name=script_name
    )
    tracker.start(device=device)
    return tracker


def attach_to_existing_run(
    cfg: DictConfig,
    *,
    project_root: Path,
    run_dir: str | Path,
    script_name: str,
) -> Path:
    """Attach logging to a run directory without rewriting its config or metadata."""
    target_run_dir = resolve_path(project_root, run_dir)
    target_run_dir.mkdir(parents=True, exist_ok=True)
    configure_run_logging(
        cfg,
        project_root,
        target_run_dir,
        log_dir_override=target_run_dir,
    )
    logging.getLogger(__name__).info(
        "Attached %s to existing run directory %s",
        script_name,
        target_run_dir,
    )
    return target_run_dir
