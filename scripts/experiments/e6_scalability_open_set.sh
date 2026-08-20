#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
: > run_status.txt

run_exp6_scalability() {
  local clients="$1"
  local run_id="e6_scalability_${clients}clients_alpha05_openset_fot_seed42"

  echo "================================================================================"
  echo "EXP6: Scalability open-set, ${clients} clients, moderate non-IID alpha=0.5, FoT unknown"
  echo "================================================================================"

  if poetry run python run.py experiment=exp6 \
    +method=fedtros \
    seed=42 \
    federated.num_rounds=100 \
    training.local_episodes_per_round=10 \
    open_set.evt.backend=fedtros_osr \
    open_set.evt.evaluate_each_round=true \
    open_set.evt.evaluate_every_n_rounds=1 \
    open_set.evt.save_round_scores=false \
    open_set.fedtros_osr.enabled=true \
    open_set.fedtros_osr.score_fusion.method=prototype_rank \
    open_set.fedtros_osr.proser.enabled=false \
    training.fedtros_student_osr_enabled=true \
    training.fedtros_student_open_set_enabled=true \
    training.generator.enabled=false \
    training.evaluate_after_local_fit=true \
    evaluation.save_client_reports=false \
    evaluation.log_client_reports=false \
    evaluation.save_client_confusion_matrices=false \
    federated.server.fraction_evaluate=0.0 \
    federated.num_clients="${clients}" \
    dataset.preprocessing.alpha=0.5 \
    'dataset.known_labels=[Normal,BP,DoS,MitM]' \
    tracking.run_id="${run_id}" \
    2>&1 | tee "logs/${run_id}.log"; then
    echo "DONE ${run_id}" | tee -a run_status.txt
  else
    echo "FAILED ${run_id}" | tee -a run_status.txt
    return 1
  fi
}

run_exp6_scalability 50
run_exp6_scalability 100

if poetry run python scripts/scalability_report.py \
  --runs \
    outputs/e6_scalability_50clients_alpha05_openset_fot_seed42 \
    outputs/e6_scalability_100clients_alpha05_openset_fot_seed42 \
  --output outputs/e6_scalability_report \
  2>&1 | tee logs/e6_scalability_report.log; then
  echo "DONE e6_scalability_report" | tee -a run_status.txt
else
  echo "FAILED e6_scalability_report" | tee -a run_status.txt
  exit 1
fi
