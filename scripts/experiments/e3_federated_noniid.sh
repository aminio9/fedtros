#!/usr/bin/env bash
set -euo pipefail

invoke_hydra_run() {
  poetry run python run.py "$@"
}

for alpha in "0.1" "0.5" "1.0" "10.0"; do
  invoke_hydra_run "experiment=exp3" "+method=fmrl_ava_glow" "seed=42" "dataset.preprocessing.alpha=$alpha"
  invoke_hydra_run "experiment=exp3" "+method=fmrl_ava" "seed=42" "dataset.preprocessing.alpha=$alpha"
  invoke_hydra_run "experiment=exp3" "+method=fedmade" "seed=42" "dataset.preprocessing.alpha=$alpha"
  invoke_hydra_run "experiment=exp3" "+method=fedavg" "seed=42" "dataset.preprocessing.alpha=$alpha"
  invoke_hydra_run "experiment=exp3" "+method=fedprox" "seed=42" "dataset.preprocessing.alpha=$alpha"
  # invoke_hydra_run "experiment=exp3" "+method=centralized_no_osr" "seed=42" "dataset.preprocessing.alpha=$alpha" "tracking.run_id=e3_central_alpha${alpha}_seed42"
done


python run.py experiment=exp1 +method=fedavg seed=42 federated.num_clients=3 dataset.preprocessing.num_clients=3 dataset.preprocessing.iid=true dataset.preprocessing.alpha=1.0

python run.py experiment=exp1 +method=fedgpa_glow seed=42 federated.num_clients=3 dataset.preprocessing.num_clients=3 dataset.preprocessing.iid=true dataset.preprocessing.alpha=1.0

python run.py experiment=exp3 +method=fedavg seed=42 dataset.preprocessing.alpha=0.1

python run.py experiment=exp3 +method=fedgpa_glow seed=42 dataset.preprocessing.alpha=0.1