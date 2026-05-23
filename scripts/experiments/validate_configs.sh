#!/usr/bin/env bash
set -euo pipefail

experiments=("exp1" "exp2" "exp3" "exp4" "ablation" "efficiency" "validation" "all")
for experiment in "${experiments[@]}"; do
  poetry run python run.py "experiment=$experiment" --cfg job --resolve >/dev/null
done

poetry run python run.py "experiment=exp3" "+method=fedavg" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp3" "+method=fedprox" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=ablation" "runtime=gpu" --cfg job --resolve >/dev/null
