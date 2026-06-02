#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 CHECKPOINT [RUN_ID]" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

checkpoint=$1
run_id=${2:-resumed_fmrl_seed42}

poetry run python run.py "experiment=exp3" "+method=fmrl_ava" "federated.resume_from=$checkpoint" "tracking.run_id=$run_id"
# poetry run python run.py "experiment=exp6" "+method=centralized_osr" "training.resume_from=$checkpoint" "tracking.run_id=${run_id}_central"
