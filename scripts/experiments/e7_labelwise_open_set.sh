#!/usr/bin/env bash
set -euo pipefail

invoke_hydra_run() {
  poetry run python run.py "$@"
}

label_names=("MitM" "FoT")
label_knowns=("Normal,BP,DoS,FoT" "Normal,BP,DoS,MitM")

for idx in "${!label_names[@]}"; do
  unknown_label="${label_names[$idx]}"
  known_labels="${label_knowns[$idx]}"

  invoke_hydra_run \
    "experiment=exp7" \
    "+method=fmrl_la" \
    "seed=42" \
    "dataset.known_labels=[$known_labels]" \
    "tracking.run_id=e7_${unknown_label,,}_fmrl_la_seed42"

  invoke_hydra_run \
    "experiment=exp7" \
    "+method=fedavg" \
    "seed=42" \
    "dataset.known_labels=[$known_labels]" \
    "tracking.run_id=e7_${unknown_label,,}_fedavg_seed42"
done
