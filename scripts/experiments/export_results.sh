#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [-r RUN_ID] RUN_DIR [RUN_DIR ...]" >&2
}

run_id="suite_export"
while getopts ":r:h" opt; do
  case "$opt" in
    r) run_id=$OPTARG ;;
    h)
      usage
      exit 0
      ;;
    \?)
      usage
      exit 2
      ;;
  esac
done
shift $((OPTIND - 1))

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

IFS=,
hydra_runs="[$*]"
unset IFS

poetry run python run.py "experiment.pipeline=export" "runs=$hydra_runs" "tracking.run_id=$run_id"
poetry run python run.py "experiment.pipeline=plot" "run_dir=outputs/$run_id" "tracking.run_id=$run_id"
