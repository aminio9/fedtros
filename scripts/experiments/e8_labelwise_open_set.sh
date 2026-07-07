#!/usr/bin/env bash
set -euo pipefail

invoke_hydra_run() {
  poetry run python run.py "$@"
}

# Exp8 label-wise open-set runs. Normal is always kept as known.
# Each command trains on the known labels and evaluates the held-out label only in open_set_test.

invoke_hydra_run \
  "experiment=exp8" \
  "+method=dkd_fedos" \
  "seed=42" \
  "dataset.known_labels=[Normal,DoS,MitM,FoT]" \
  "tracking.run_id=e8_bp_dkd_fedos_seed42"

invoke_hydra_run \
  "experiment=exp8" \
  "+method=dkd_fedos" \
  "seed=42" \
  "dataset.known_labels=[Normal,BP,MitM,FoT]" \
  "tracking.run_id=e8_dos_dkd_fedos_seed42"

invoke_hydra_run \
  "experiment=exp8" \
  "+method=dkd_fedos" \
  "seed=42" \
  "dataset.known_labels=[Normal,BP,DoS,FoT]" \
  "tracking.run_id=e8_mitm_dkd_fedos_seed42"

invoke_hydra_run \
  "experiment=exp8" \
  "+method=dkd_fedos" \
  "seed=42" \
  "dataset.known_labels=[Normal,BP,DoS,MitM]" \
  "tracking.run_id=e8_fot_dkd_fedos_seed42"
