#!/usr/bin/env bash
set -euo pipefail

invoke_hydra_run() {
  poetry run python run.py "$@"
}

invoke_hydra_run "experiment=exp6" "+method=fedtros" "seed=42" "tracking.run_id=e6_fedtros_full_seed42"
invoke_hydra_run "experiment=exp6" "+method=fedtros" "seed=42" "training.fedtros_task_weight=0.0" "experiment.method=FedTROS_No_Task" "tracking.run_id=e6_fedtros_no_task_seed42"
invoke_hydra_run "experiment=exp6" "+method=fedtros" "seed=42" "training.fedtros_lambda_kd_init=0.0" "training.fedtros_lambda_kd_max=0.0" "experiment.method=FedTROS_No_KD" "tracking.run_id=e6_fedtros_no_kd_seed42"
invoke_hydra_run "experiment=exp6" "+method=fedtros" "seed=42" "training.fedtros_lambda_align_init=0.0" "training.fedtros_lambda_align_max=0.0" "experiment.method=FedTROS_No_Align" "tracking.run_id=e6_fedtros_no_align_seed42"
invoke_hydra_run "experiment=exp6" "+method=fmrl_ava" "seed=42" "tracking.run_id=e6_full_seed42"
invoke_hydra_run "experiment=exp6" "+method=fmrl_ava" "seed=42" "open_set.evt.enabled=false" "experiment.method=No_EVT" "tracking.run_id=e6_no_evt_seed42"
invoke_hydra_run "experiment=exp6" "+method=fmrl_ava" "seed=42" "training.generator.enabled=false" "experiment.method=No_Generator" "tracking.run_id=e6_no_generator_seed42"
invoke_hydra_run "experiment=exp6" "+method=fmrl_ava" "seed=42" "federated.strategy.utility_threshold=-1.0" "experiment.method=No_Selection" "tracking.run_id=e6_no_selection_seed42"
invoke_hydra_run "experiment=exp6" "+method=fedavg" "seed=42" "open_set.evt.enabled=false" "tracking.run_id=e6_fedavg_no_osr_seed42"
invoke_hydra_run "experiment=exp6" "+method=fedprox" "seed=42" "open_set.evt.enabled=false" "tracking.run_id=e6_fedprox_no_osr_seed42"
# invoke_hydra_run "experiment=exp6" "+method=centralized_osr" "seed=42" "tracking.run_id=e6_central_osr_seed42"
# invoke_hydra_run "experiment=exp6" "+method=centralized_no_osr" "seed=42" "tracking.run_id=e6_central_no_osr_seed42"
