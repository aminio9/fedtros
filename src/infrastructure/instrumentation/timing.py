"""Runtime timing and profiling instrumentation for FedTROS-PR."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

logger = logging.getLogger(__name__)


class RuntimeTracker:
    """Accurate timing instrumentation for federated training, aggregation, and evaluation."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        self.round_timings: list[dict[str, Any]] = []
        self._current_round: int = 1
        self._active_round_metrics: dict[str, float] = {}

    def start_round(self, round_num: int) -> None:
        """Begin tracking a new federated round."""
        self._current_round = round_num
        self._active_round_metrics = {
            "round": round_num,
            "runtime/teacher_training_seconds": 0.0,
            "runtime/student_training_seconds": 0.0,
            "runtime/client_fit_seconds": 0.0,
            "runtime/server_aggregation_seconds": 0.0,
            "runtime/open_set_eval_seconds": 0.0,
            "runtime/orchestration_seconds": 0.0,
            "runtime/round_seconds": 0.0,
        }

    @contextmanager
    def time_stage(self, stage_name: str) -> Iterator[None]:
        """Context manager to measure runtime of a named stage."""
        key = f"runtime/{stage_name}_seconds" if not stage_name.startswith("runtime/") else stage_name
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if key not in self._active_round_metrics:
                self._active_round_metrics[key] = 0.0
            self._active_round_metrics[key] += round(elapsed, 4)

    def add_duration(self, stage_name: str, duration_seconds: float) -> None:
        """Manually add elapsed seconds to a stage."""
        key = f"runtime/{stage_name}_seconds" if not stage_name.startswith("runtime/") else stage_name
        if key not in self._active_round_metrics:
            self._active_round_metrics[key] = 0.0
        self._active_round_metrics[key] += round(duration_seconds, 4)

    def end_round(self, total_round_seconds: float | None = None) -> dict[str, Any]:
        """Finalize timing metrics for the active round."""
        if total_round_seconds is not None:
            self._active_round_metrics["runtime/round_seconds"] = round(total_round_seconds, 4)
        elif self._active_round_metrics["runtime/round_seconds"] == 0.0:
            # Estimate sum of components
            self._active_round_metrics["runtime/round_seconds"] = round(
                self._active_round_metrics.get("runtime/client_fit_seconds", 0.0)
                + self._active_round_metrics.get("runtime/server_aggregation_seconds", 0.0)
                + self._active_round_metrics.get("runtime/open_set_eval_seconds", 0.0),
                4,
            )

        summary = dict(self._active_round_metrics)
        self.round_timings.append(summary)

        if self.output_dir is not None:
            self.save(self.output_dir)

        return summary

    def save(self, output_dir: Path) -> None:
        """Write timing data to metrics/timing_round.csv."""
        out = Path(output_dir)
        metrics_dir = out / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        if self.round_timings:
            df = pd.DataFrame(self.round_timings)
            df.to_csv(metrics_dir / "timing_round.csv", index=False)
