#!/usr/bin/env bash
set -euo pipefail

experiments=("baseline" "exp1" "exp2" "exp3" "exp4" "exp5" "exp6" "exp7" "exp8" "validation" "all")
for experiment in "${experiments[@]}"; do
  poetry run python run.py "experiment=$experiment" --cfg job --resolve >/dev/null
done

poetry run python run.py "experiment=validation" "runtime=tiny" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=smoke" "runtime=tiny" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp1" "+method=fedprox" --cfg job --resolve >/dev/null
# poetry run python run.py "experiment=exp1" "+method=centralized_no_osr" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp2" "+method=fmrl_ava" "open_set.evt.enabled=false" "experiment.method=ClosedSet_No_EVT" --cfg job --resolve >/dev/null
# poetry run python run.py "experiment=exp2" "+method=centralized_osr" --cfg job --resolve >/dev/null
# poetry run python run.py "experiment=exp2" "+method=centralized_no_osr" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp5" "+method=fmrl_ava" "dataset.name=B-TAT" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp6" "+method=fmrl_ava" "open_set.evt.enabled=false" "experiment.method=No_EVT" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp6" "+method=fmrl_ava" "training.generator.enabled=false" "experiment.method=No_Generator" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp6" "+method=fmrl_ava" "federated.strategy.utility_threshold=-1.0" "experiment.method=No_Selection" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp7" "federated.num_clients=20" "federated.num_rounds=50" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp7" "federated.num_clients=20" "federated.num_rounds=100" --cfg job --resolve >/dev/null
<<<<<<< HEAD
=======
poetry run python run.py "experiment=exp3" "+method=fmrl_ava_glow" --cfg job --resolve >/dev/null
>>>>>>> ea28efe (Initial commit with updated source code)
poetry run python run.py "experiment=exp3" "+method=fedavg" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp3" "+method=fedprox" --cfg job --resolve >/dev/null
# poetry run python run.py "experiment=exp3" "+method=centralized_no_osr" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp4" "+method=fmrl_ava" "open_set.evt.enabled=false" "experiment.method=ClosedSet_No_EVT" --cfg job --resolve >/dev/null
# poetry run python run.py "experiment=exp4" "+method=centralized_osr" --cfg job --resolve >/dev/null
# poetry run python run.py "experiment=exp4" "+method=centralized_no_osr" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp8" "+method=fmrl_ava" "dataset.known_labels=[Normal,BP,DoS,FoT]" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp8" "+method=fmrl_ava" "dataset.known_labels=[Normal,BP,DoS,MitM]" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp8" "+method=fedavg" "dataset.known_labels=[Normal,BP,DoS,FoT]" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp8" "+method=fedavg" "dataset.known_labels=[Normal,BP,DoS,MitM]" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp6" "runtime=gpu" --cfg job --resolve >/dev/null
poetry run python run.py "experiment=exp7" "federated.num_clients=20" "federated.num_rounds=200" --cfg job --resolve >/dev/null
