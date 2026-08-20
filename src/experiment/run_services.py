"""Bridge between scientific code, durable ResultStore, and interactive tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

from omegaconf import DictConfig

from src.experiment.result_store import ResultStore
from src.infrastructure.tracking import ExperimentTracker, create_tracker

logger = logging.getLogger(__name__)


class MetricsSink(Protocol):
    """Minimal interface scientific routines may use without knowing W&B."""

    run_dir: Path

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None: ...
    def write_json(self, filename: str | Path, payload: dict[str, Any] | list[Any]) -> Path: ...


class RunServices:
    """Run-scoped services combining local persistence and a single external tracker."""

    def __init__(self, result_store: ResultStore, tracker: ExperimentTracker) -> None:
        self.result_store = result_store
        self.tracker = tracker
        self.run_dir = result_store.run_dir
        self.run_id = result_store.run_id
        self._summary: dict[str, Any] = {}

    @property
    def tracker_run_id(self) -> str:
        return self.tracker.run_id

    @property
    def tracker_id(self) -> str:
        return self.tracker.tracker_id

    def log_config(self, config: DictConfig | dict[str, Any]) -> None:
        self.result_store.save_config(config)
        self.tracker.log_config(config)

    def log_resume_config(
        self,
        config: DictConfig | dict[str, Any],
        *,
        resumed_from_round: int,
    ) -> Path:
        """Record continuation overrides while preserving the original frozen config."""
        path = self.result_store.save_resume_config(
            config,
            resumed_from_round=resumed_from_round,
        )
        try:
            self.tracker.log_config(config)
        except Exception as exc:
            logger.warning("External tracker resume-config logging failed: %s", exc, exc_info=True)
        return path

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.result_store.append_round_metrics(metrics, step=step)
        try:
            self.tracker.log_metrics(metrics, step=step)
        except Exception as exc:  # external observability must not corrupt scientific results
            logger.warning("External tracker metric logging failed: %s", exc, exc_info=True)

    def write_json(self, filename: str | Path, payload: dict[str, Any] | list[Any]) -> Path:
        return self.result_store.write_json(filename, payload)

    def set_summary(self, summary: dict[str, Any]) -> None:
        self._summary.update(summary)
        self.result_store.save_final_metrics(self._summary)
        try:
            self.tracker.set_summary(summary)
        except Exception as exc:
            logger.warning("External tracker summary logging failed: %s", exc, exc_info=True)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        try:
            self.tracker.log_artifact(local_path, artifact_path=artifact_path)
        except Exception as exc:
            logger.warning("External tracker artifact logging failed: %s", exc, exc_info=True)

    def set_status(self, status: str) -> None:
        try:
            self.tracker.set_status(status)
        except Exception as exc:
            logger.warning("External tracker status update failed: %s", exc, exc_info=True)

    def finish(self, status: str = "COMPLETED") -> None:
        self.result_store.finalize_result_manifest(
            status=status,
            final_metrics=self._summary,
            extra={"tracker": self.tracker_id, "tracker_run_id": self.tracker_run_id},
        )
        try:
            self.tracker.finish(status=status)
        except Exception as exc:
            logger.warning("External tracker finalization failed: %s", exc, exc_info=True)


def create_run_services(
    cfg: DictConfig | dict[str, Any],
    *,
    run_dir: Path,
    run_id: str,
    human_name: str | None = None,
    study_id: str | None = None,
    stage: str = "development",
    resume: bool = False,
) -> RunServices:
    result_store = ResultStore(run_dir=run_dir, run_id=run_id)
    tracker = create_tracker(
        cfg,
        run_dir=run_dir,
        run_id=run_id,
        human_name=human_name,
        study_id=study_id,
        stage=stage,
        resume=resume,
    )
    return RunServices(result_store=result_store, tracker=tracker)
