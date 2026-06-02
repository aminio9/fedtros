#!/usr/bin/env bash
set -euo pipefail

invoke_hydra_run() {
  poetry run python run.py "$@"
}

invoke_hydra_run "experiment=exp6" "+method=fmrl_ava" "seed=42" "tracking.run_id=e6_full_seed42"
invoke_hydra_run "experiment=exp6" "+method=fmrl_ava" "seed=42" "open_set.evt.enabled=false" "experiment.method=No_EVT" "tracking.run_id=e6_no_evt_seed42"
invoke_hydra_run "experiment=exp6" "+method=fmrl_ava" "seed=42" "training.generator.enabled=false" "experiment.method=No_Generator" "tracking.run_id=e6_no_generator_seed42"
invoke_hydra_run "experiment=exp6" "+method=fmrl_ava" "seed=42" "federated.strategy.utility_threshold=-1.0" "experiment.method=No_Selection" "tracking.run_id=e6_no_selection_seed42"
invoke_hydra_run "experiment=exp6" "+method=fedavg" "seed=42" "open_set.evt.enabled=false" "tracking.run_id=e6_fedavg_no_osr_seed42"
invoke_hydra_run "experiment=exp6" "+method=fedprox" "seed=42" "open_set.evt.enabled=false" "tracking.run_id=e6_fedprox_no_osr_seed42"
# invoke_hydra_run "experiment=exp6" "+method=centralized_osr" "seed=42" "tracking.run_id=e6_central_osr_seed42"
# invoke_hydra_run "experiment=exp6" "+method=centralized_no_osr" "seed=42" "tracking.run_id=e6_central_no_osr_seed42"
