#!/usr/bin/env bash
set -euo pipefail

invoke_hydra_run() {
  poetry run python run.py "$@"
}

for clients in 3 10 20 50 100; do
  invoke_hydra_run "experiment=efficiency" "+method=fmrl_la" "seed=42" "federated.num_clients=$clients"
  invoke_hydra_run "experiment=efficiency" "+method=fedavg" "seed=42" "federated.num_clients=$clients"
done
