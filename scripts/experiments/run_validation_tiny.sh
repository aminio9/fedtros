#!/usr/bin/env bash
set -euo pipefail

invoke_hydra_run() {
  poetry run python run.py "$@"
}

run_id="tiny_validation_seed42"
invoke_hydra_run "experiment=validation" "seed=42" "tracking.run_id=$run_id"
poetry run python scripts/plot.py "run_dir=outputs/$run_id" "tracking.run_id=$run_id"
