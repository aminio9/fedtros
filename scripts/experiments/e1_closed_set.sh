#!/usr/bin/env bash
set -euo pipefail

invoke_hydra_run() {
  poetry run python run.py "$@"
}

# invoke_hydra_run "experiment=exp1" "+method=centralized_no_osr" "seed=42"
invoke_hydra_run "experiment=exp1" "+method=fmrl_ava" "seed=42"
invoke_hydra_run "experiment=exp1" "+method=fedavg" "seed=42"
invoke_hydra_run "experiment=exp1" "+method=fedprox" "seed=42"
poetry run python run.py "experiment=exp1" "+method=fmrl_ava_glow" "seed=42"



poetry run python run.py experiment=exp3 +method=fmrl_ava_glow_twa seed=42 \
  federated.num_clients=3 \
  dataset.preprocessing.num_clients=3 \
  dataset.preprocessing.iid=true \
  dataset.preprocessing.alpha=1.0 \
  federated.strategy.local_proximal_mu=0.0 \
  tracking.run_id=e3_FMRL_AVA_GLOW_TWA_iid_c3_seed42