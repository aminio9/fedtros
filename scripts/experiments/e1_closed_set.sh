#!/usr/bin/env bash
set -euo pipefail

invoke_hydra_run() {
  poetry run python run.py "$@"
}

invoke_hydra_run "experiment=exp1" "+method=fmrl_la" "seed=42"
invoke_hydra_run "experiment=exp1" "+method=fedavg" "seed=42"
invoke_hydra_run "experiment=exp1" "+method=fedprox" "seed=42"
invoke_hydra_run "experiment=exp1" "+method=centralized_no_osr" "seed=42"
