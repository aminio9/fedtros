"""Experiment-tracker abstraction for FedTROS-PR.

The tracker is intentionally restricted to interactive observability. Durable scientific
results live in :mod:`src.experiment.result_store`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


class ExperimentTracker(ABC):
    """Abstract interface implemented by W&B and a no-op tracker."""

    @property
    @abstractmethod
    def tracker_id(self) -> str: ...

    @property
    @abstractmethod
    def run_id(self) -> str: ...

    @abstractmethod
    def log_config(self, config: DictConfig | dict[str, Any]) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None: ...

    @abstractmethod
    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None: ...

    @abstractmethod
    def set_summary(self, summary: dict[str, Any]) -> None: ...

    @abstractmethod
    def set_status(self, status: str) -> None: ...

    @abstractmethod
    def finish(self, status: str = "COMPLETED") -> None: ...
