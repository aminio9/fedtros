#!/usr/bin/env bash
set -euo pipefail

invoke_hydra_run() {
  poetry run python run.py "$@"
}

# invoke_hydra_run "experiment=exp1" "+method=centralized_no_osr" "seed=42"
invoke_hydra_run "experiment=exp1" "+method=fmrl_ava" "seed=42"
invoke_hydra_run "experiment=exp1" "+method=fedgpa" "seed=42"
invoke_hydra_run "experiment=exp1" "+method=fedavg" "seed=42"
invoke_hydra_run "experiment=exp1" "+method=fedprox" "seed=42"
