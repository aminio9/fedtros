#!/usr/bin/env bash
set -euo pipefail

invoke_hydra_run() {
  poetry run python run.py "$@"
}

for alpha in "0.1" "10.0"; do
  invoke_hydra_run "experiment=exp3" "+method=fmrl_la" "seed=42" "dataset.preprocessing.alpha=$alpha"
  invoke_hydra_run "experiment=exp3" "+method=fedavg" "seed=42" "dataset.preprocessing.alpha=$alpha"
  invoke_hydra_run "experiment=exp3" "+method=fedprox" "seed=42" "dataset.preprocessing.alpha=$alpha"
done
