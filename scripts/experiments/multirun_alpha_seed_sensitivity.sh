#!/usr/bin/env bash
set -euo pipefail

poetry run python run.py --multirun "experiment=exp4" "+method=fmrl_la" "+sweep=seeds" "dataset.preprocessing.alpha=0.01,0.5,1.0"
