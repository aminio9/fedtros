#!/usr/bin/env bash
set -euo pipefail

invoke_hydra_run() {
  poetry run python run.py "$@"
}

for alpha in "0.1" "0.5" "1.0" "10.0"; do
  invoke_hydra_run "experiment=exp3" "+method=fedtros" "seed=42" "dataset.preprocessing.alpha=$alpha"
  invoke_hydra_run "experiment=exp3" "+method=fmrl_ava" "seed=42" "dataset.preprocessing.alpha=$alpha"
  invoke_hydra_run "experiment=exp3" "+method=fedgpa" "seed=42" "dataset.preprocessing.alpha=$alpha"
  invoke_hydra_run "experiment=exp3" "+method=fedavg" "seed=42" "dataset.preprocessing.alpha=$alpha"
  invoke_hydra_run "experiment=exp3" "+method=fedprox" "seed=42" "dataset.preprocessing.alpha=$alpha"
  # invoke_hydra_run "experiment=exp3" "+method=centralized_no_osr" "seed=42" "dataset.preprocessing.alpha=$alpha" "tracking.run_id=e3_central_alpha${alpha}_seed42"
done
