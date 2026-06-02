#!/usr/bin/env bash
set -euo pipefail

invoke_hydra_run() {
  poetry run python run.py "$@"
}

for rounds in 50 100 200; do
  for clients in 3 10 20 50 100; do
    invoke_hydra_run "experiment=exp7" "+method=fmrl_ava" "seed=42" "federated.num_clients=$clients" "federated.num_rounds=$rounds" "tracking.run_id=e7_fmrl_ava_clients${clients}_rounds${rounds}_seed42"
    invoke_hydra_run "experiment=exp7" "+method=fedavg" "seed=42" "federated.num_clients=$clients" "federated.num_rounds=$rounds" "tracking.run_id=e7_fedavg_clients${clients}_rounds${rounds}_seed42"
  done
done
