#!/usr/bin/env bash
set -euo pipefail

poetry run python run.py --multirun "experiment=exp4" "+method=fmrl_ava" "+sweep=seeds" "dataset.preprocessing.alpha=0.1,0.5,1.0,10.0"
