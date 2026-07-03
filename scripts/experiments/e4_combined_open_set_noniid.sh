#!/usr/bin/env bash
set -euo pipefail

invoke_hydra_run() {
  poetry run python run.py "$@"
}

for alpha in "0.1" "0.5" "1.0" "10.0"; do
  invoke_hydra_run "experiment=exp4" "+method=fmrl_ava" "seed=42" "dataset.preprocessing.alpha=$alpha"
  invoke_hydra_run "experiment=exp4" "+method=fmrl_ava" "seed=42" "dataset.preprocessing.alpha=$alpha" "open_set.evt.enabled=false" "experiment.method=ClosedSet_No_EVT" "tracking.run_id=e4_no_evt_alpha${alpha}_seed42"
<<<<<<< HEAD
=======
  invoke_hydra_run "experiment=exp4" "+method=fedmade" "seed=42" "dataset.preprocessing.alpha=$alpha"
>>>>>>> ea28efe (Initial commit with updated source code)
  invoke_hydra_run "experiment=exp4" "+method=fedavg" "seed=42" "dataset.preprocessing.alpha=$alpha"
  invoke_hydra_run "experiment=exp4" "+method=fedprox" "seed=42" "dataset.preprocessing.alpha=$alpha"
  # invoke_hydra_run "experiment=exp4" "+method=centralized_no_osr" "seed=42" "dataset.preprocessing.alpha=$alpha" "tracking.run_id=e4_central_no_osr_alpha${alpha}_seed42"
  # invoke_hydra_run "experiment=exp4" "+method=centralized_osr" "seed=42" "dataset.preprocessing.alpha=$alpha" "tracking.run_id=e4_central_osr_alpha${alpha}_seed42"
done
