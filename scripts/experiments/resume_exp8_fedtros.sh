#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 CHECKPOINT PREVIOUS_ROUND UNKNOWN_LABEL [RUN_ID] [TARGET_ROUND]" >&2
  echo "Example: $0 outputs/e8_mitm_unknown_fedtros_seed42/fedtros_student_latest.pt 52 MitM e8_mitm_resume_from52_seed42 100" >&2
  exit 2
fi

checkpoint=$1
previous_round=$2
unknown_label=$3
run_id=${4:-e8_${unknown_label,,}_resume_from${previous_round}_seed42}
target_round=${5:-100}
remaining_rounds=$(( target_round - previous_round ))

if [[ $remaining_rounds -le 0 ]]; then
  echo "Nothing to resume: target_round=$target_round previous_round=$previous_round" >&2
  exit 2
fi

case "$unknown_label" in
  BP) known='[Normal,DoS,MitM,FoT]' ;;
  DoS) known='[Normal,BP,MitM,FoT]' ;;
  MitM) known='[Normal,BP,DoS,FoT]' ;;
  FoT) known='[Normal,BP,DoS,MitM]' ;;
  *) echo "UNKNOWN_LABEL must be one of BP, DoS, MitM, FoT" >&2; exit 2 ;;
esac

poetry run python run.py experiment=exp8 +method=fedtros seed=42 \
  "dataset.known_labels=${known}" \
  "federated.resume_from=${checkpoint}" \
  "federated.resume_round_offset=${previous_round}" \
  "federated.num_rounds=${remaining_rounds}" \
  training.generator.enabled=false \
  training.fedtros_student_osr_enabled=true \
  training.fedtros_student_open_set_enabled=true \
  open_set.evt.backend=fedtros_osr \
  open_set.fedtros_osr.enabled=true \
  open_set.fedtros_osr.score_fusion.method=prototype_rank \
  tracking.run_id="${run_id}" \
  2>&1 | tee "${run_id}.log"
