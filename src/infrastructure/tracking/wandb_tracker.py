"""Weights & Biases experiment tracker for FedTROS-PR."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.infrastructure.tracking.base import ExperimentTracker

logger = logging.getLogger(__name__)


def _to_container(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        value = OmegaConf.to_container(config, resolve=True)
        return dict(value) if isinstance(value, dict) else {"config": value}
    return dict(config)


def _clean_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class WandBTracker(ExperimentTracker):
    """Single interactive tracking backend using the W&B Python SDK.

    W&B is not the canonical scientific result store.  All required result files are
    persisted independently by ``ResultStore`` before/alongside calls to this class.
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        canonical_run_id: str,
        project: str = "FedTROS-PR",
        entity: str | None = None,
        name: str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
        mode: str = "online",
        resume: str | bool | None = "never",
        notes: str | None = None,
        save_code: bool = False,
    ) -> None:
        run_dir = Path(run_dir).expanduser().resolve()
        wandb_dir = run_dir / "wandb"
        wandb_data_dir = wandb_dir / "data"
        wandb_cache_dir = wandb_dir / "cache"
        wandb_config_dir = wandb_dir / "config"
        for directory in (wandb_dir, wandb_data_dir, wandb_cache_dir, wandb_config_dir):
            directory.mkdir(parents=True, exist_ok=True)

        # W&B otherwise stages artifacts and core logs under the user-local profile,
        # which is frequently read-only on managed servers/containers.  Each run is
        # an independent process, so run-local SDK state is parallel-safe and fully
        # portable with the canonical output directory.
        os.environ["WANDB_DATA_DIR"] = str(wandb_data_dir)
        os.environ["WANDB_CACHE_DIR"] = str(wandb_cache_dir)
        os.environ["WANDB_CONFIG_DIR"] = str(wandb_config_dir)
        os.environ["WANDB_DIR"] = str(wandb_dir)

        try:
            import wandb
        except ImportError as exc:  # make the configuration error explicit
            raise RuntimeError(
                "tracking.backend=wandb requires the 'wandb' package. "
                "Install project dependencies or use tracking.mode=disabled."
            ) from exc

        mode = str(mode).lower()
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError(f"Unsupported W&B mode: {mode!r}")
        # W&B run IDs may not contain /\\#?%: . Generated FedTROS IDs already avoid these;
        # enforce the restriction here to fail before network use.
        forbidden = set("/\\#?%:")
        if any(char in canonical_run_id for char in forbidden):
            raise ValueError(f"Invalid W&B run id {canonical_run_id!r}; contains a forbidden character.")

        self._wandb = wandb
        self._canonical_run_id = canonical_run_id
        self._run = wandb.init(
            project=project,
            entity=entity or None,
            id=canonical_run_id,
            name=name or canonical_run_id,
            group=group or None,
            job_type=job_type or None,
            tags=list(tags or []),
            dir=str(wandb_dir.resolve()),
            mode=mode,
            resume=resume,
            notes=notes,
            save_code=save_code,
            reinit="create_new",
        )
        if self._run is None:
            raise RuntimeError("wandb.init() returned no Run object")
        logger.info(
            "Initialized W&B tracking | mode=%s | project=%s | group=%s | id=%s",
            mode,
            project,
            group,
            self._run.id,
        )

    @property
    def tracker_id(self) -> str:
        return "wandb"

    @property
    def run_id(self) -> str:
        return str(self._run.id)

    def log_config(self, config: DictConfig | dict[str, Any]) -> None:
        payload = _to_container(config)
        # W&B handles nested dictionaries; this avoids dot-separated flat keys.
        self._run.config.update(payload, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        payload: dict[str, Any] = {}
        for key, value in metrics.items():
            clean = _clean_value(value)
            if isinstance(clean, bool):
                payload[str(key)] = int(clean)
            elif isinstance(clean, (int, float)):
                numeric = float(clean)
                if math.isfinite(numeric):
                    payload[str(key)] = numeric
            elif isinstance(clean, str):
                payload[str(key)] = clean
        if payload:
            self._run.log(payload, step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        path = Path(local_path)
        if not path.exists():
            logger.warning("W&B artifact path does not exist: %s", path)
            return
        artifact_name = artifact_path or f"{self._canonical_run_id}-{path.name}"
        artifact_type = "model" if path.suffix.lower() in {".pt", ".pth", ".ckpt"} else "run-artifact"
        self._run.log_artifact(str(path), name=artifact_name, type=artifact_type)

    def set_summary(self, summary: dict[str, Any]) -> None:
        for key, value in summary.items():
            clean = _clean_value(value)
            if isinstance(clean, (int, float, str, bool)) or clean is None:
                self._run.summary[str(key)] = clean

    def set_status(self, status: str) -> None:
        self._run.summary["run_status"] = str(status)

    def finish(self, status: str = "COMPLETED") -> None:
        self.set_status(status)
        exit_code = 0 if str(status).upper() == "COMPLETED" else 1
        self._run.finish(exit_code=exit_code)
