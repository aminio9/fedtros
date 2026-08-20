"""No-op experiment tracker used for tests and tracking-disabled runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.infrastructure.tracking.base import ExperimentTracker


class NullTracker(ExperimentTracker):
    def __init__(self, run_id: str = "null_run") -> None:
        self._run_id = str(run_id)
        self._status = "CREATED"

    @property
    def tracker_id(self) -> str:
        return "null"

    @property
    def run_id(self) -> str:
        return self._run_id

    def log_config(self, config: DictConfig | dict[str, Any]) -> None:
        _ = config

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        _ = metrics, step

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        _ = local_path, artifact_path

    def set_summary(self, summary: dict[str, Any]) -> None:
        _ = summary

    def set_status(self, status: str) -> None:
        self._status = str(status)

    def finish(self, status: str = "COMPLETED") -> None:
        self._status = str(status)
