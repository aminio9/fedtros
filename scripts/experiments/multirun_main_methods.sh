#!/usr/bin/env bash
set -euo pipefail

poetry run python run.py --multirun "experiment=exp3" "+method=fmrl_ava,fedavg,fedprox" "seed=42" "dataset.preprocessing.alpha=0.1"
