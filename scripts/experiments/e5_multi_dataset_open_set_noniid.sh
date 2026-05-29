#!/usr/bin/env bash
set -euo pipefail

invoke_hydra_run() {
  poetry run python run.py "$@"
}

run_dataset() {
  local dataset_name="$1"
  local raw_file="$2"
  local source_labels="$3"
  local known_labels="$4"
  local num_actions="$5"
  local run_slug="$6"

  if [[ -z "$raw_file" || -z "$source_labels" || -z "$known_labels" || -z "$num_actions" ]]; then
    echo "Skipping $dataset_name: finalize raw_file, source_labels, known_labels, and num_actions first."
    return 0
  fi

  invoke_hydra_run \
    "experiment=exp5" \
    "+method=fmrl_la" \
    "seed=42" \
    "dataset.name=$dataset_name" \
    "dataset.raw_path=$raw_file" \
    "dataset.source_labels=$source_labels" \
    "dataset.known_labels=$known_labels" \
    "dataset.preprocessing.raw_file=$raw_file" \
    "model.num_actions=$num_actions" \
    "dataset.preprocessing.output_dir=outputs/e5_${run_slug}/processed" \
    "tracking.run_id=e5_${run_slug}_fmrl_la_seed42"

  invoke_hydra_run \
    "experiment=exp5" \
    "+method=fedavg" \
    "seed=42" \
    "dataset.name=$dataset_name" \
    "dataset.raw_path=$raw_file" \
    "dataset.source_labels=$source_labels" \
    "dataset.known_labels=$known_labels" \
    "dataset.preprocessing.raw_file=$raw_file" \
    "model.num_actions=$num_actions" \
    "dataset.preprocessing.output_dir=outputs/e5_${run_slug}_fedavg/processed" \
    "tracking.run_id=e5_${run_slug}_fedavg_seed42"

  invoke_hydra_run \
    "experiment=exp5" \
    "+method=fedprox" \
    "seed=42" \
    "dataset.name=$dataset_name" \
    "dataset.raw_path=$raw_file" \
    "dataset.source_labels=$source_labels" \
    "dataset.known_labels=$known_labels" \
    "dataset.preprocessing.raw_file=$raw_file" \
    "model.num_actions=$num_actions" \
    "dataset.preprocessing.output_dir=outputs/e5_${run_slug}_fedprox/processed" \
    "tracking.run_id=e5_${run_slug}_fedprox_seed42"
}

ran_any=0

# Finalize these mappings before execution.
# B-TAT has a local conversion path in data/raw/convert_btat.py, but the final
# known/unknown mapping still needs to be fixed before the experiment is run.
BTAT_RAW_FILE="${BTAT_RAW_FILE:-data/raw/BTAT_dataset_final.csv}"
BTAT_SOURCE_LABELS="${BTAT_SOURCE_LABELS:-}"
BTAT_KNOWN_LABELS="${BTAT_KNOWN_LABELS:-}"
BTAT_NUM_ACTIONS="${BTAT_NUM_ACTIONS:-}"

TONIOT_RAW_FILE="${TONIOT_RAW_FILE:-}"
TONIOT_SOURCE_LABELS="${TONIOT_SOURCE_LABELS:-}"
TONIOT_KNOWN_LABELS="${TONIOT_KNOWN_LABELS:-}"
TONIOT_NUM_ACTIONS="${TONIOT_NUM_ACTIONS:-}"

CICIDS_RAW_FILE="${CICIDS_RAW_FILE:-}"
CICIDS_SOURCE_LABELS="${CICIDS_SOURCE_LABELS:-}"
CICIDS_KNOWN_LABELS="${CICIDS_KNOWN_LABELS:-}"
CICIDS_NUM_ACTIONS="${CICIDS_NUM_ACTIONS:-}"

run_dataset "B-TAT" "$BTAT_RAW_FILE" "$BTAT_SOURCE_LABELS" "$BTAT_KNOWN_LABELS" "$BTAT_NUM_ACTIONS" "btat"
run_dataset "ToN-IoT" "$TONIOT_RAW_FILE" "$TONIOT_SOURCE_LABELS" "$TONIOT_KNOWN_LABELS" "$TONIOT_NUM_ACTIONS" "toniot"
run_dataset "CIC-IDS2017" "$CICIDS_RAW_FILE" "$CICIDS_SOURCE_LABELS" "$CICIDS_KNOWN_LABELS" "$CICIDS_NUM_ACTIONS" "cicids2017"

if [[ "$ran_any" -eq 0 ]]; then
  echo "E5 scaffold is installed, but no external dataset mappings are finalized yet."
fi
