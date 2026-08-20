#!/usr/bin/env python3
"""Create the dedicated two-point W&B offline tracker validation run."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from omegaconf import OmegaConf

from src.infrastructure.tracking import create_tracker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tiny_validation/tracker_smoke"))
    args = parser.parse_args()
    run_id = "tracker_smoke_fedtros_pr_s42"
    run_dir = args.output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create({
        "seed": 42,
        "stage": "smoke",
        "experiment": {"method": "FedTROS-PR"},
        "dataset": {"name": "B-NAT", "preprocessing": {"alpha": 0.5, "unknown_labels": ["FoT"]}},
        "federated": {"num_clients": 2},
        "tracking": {"backend": "wandb", "mode": "offline", "project": "FedTROS-PR", "group": "TRACKER-SMOKE"},
    })
    tracker = create_tracker(
        cfg,
        run_dir=run_dir,
        run_id=run_id,
        human_name="FedTROS-PR | tracker smoke | s=42",
        study_id="TRACKER-SMOKE",
        stage="smoke",
    )
    tracker.log_config(cfg)
    tracker.log_metrics({"validation/test_metric": 1}, step=1)
    tracker.log_metrics({"validation/test_metric": 2}, step=2)
    tracker.set_summary({"validation/test_metric": 2, "stage": "smoke"})
    tracker.finish(status="COMPLETED")
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
