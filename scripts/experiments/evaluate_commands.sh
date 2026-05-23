#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 CHECKPOINT [EXPERIMENT] [RUN_ID]" >&2
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit 2
fi

checkpoint=$1
experiment=${2:-exp4}
run_id=${3:-eval_from_checkpoint}

poetry run python run.py "experiment=$experiment" "experiment.pipeline=evaluate" "checkpoint.path=$checkpoint" "evaluation.checkpoint_path=$checkpoint" "tracking.run_id=$run_id"
poetry run python run.py "experiment=$experiment" "experiment.pipeline=plot" "run_dir=outputs/$run_id" "tracking.run_id=$run_id"
