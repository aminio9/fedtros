#!/usr/bin/env bash
set -euo pipefail

<<<<<<< HEAD
poetry run python run.py --multirun "experiment=exp3" "+method=fmrl_ava,fedavg,fedprox" "seed=42" "dataset.preprocessing.alpha=0.1"
=======
poetry run python run.py --multirun "experiment=exp3" "+method=fmrl_ava_glow,fmrl_ava,fedmade,fedavg,fedprox" "seed=42" "dataset.preprocessing.alpha=0.1"
>>>>>>> ea28efe (Initial commit with updated source code)
