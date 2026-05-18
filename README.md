# Cooperative Federated MARL for Open-Set Blockchain Traffics Intrusion Detection

This repository implements a local, reproducible research pipeline for federated Blockchain Traffics intrusion detection with cooperative multi-agent reinforcement learning, Double Q-learning, EVT open-set recognition, and Flower-based federated training.

## Project Layout

```text
src/
  agents/          Double Q-learning agent and policies
  checkpointing/   best/latest/final checkpoint helpers
  configs/         Hydra config groups
  data/            preprocessing and tensor loading
  evaluation/      closed-set, open-set, and run comparison
  federated/       Flower client/server/simulation orchestration
  models/          CVAE-DQN and optional tabular Transformer encoder
  openset/         EVT implementation
  plotting/        Q1 figure registry and per-plot rendering
  rl/              Gymnasium environment and replay training loop
  tracking/        local-only run tracker
  training/        centralized/debug training and smoke test
  utils/           config, device, logging, seeding helpers
scripts/           thin Hydra entry points
docs/              experiment, architecture, evaluation, and reproducibility notes
```

## Setup

```bash
poetry install
poetry install -E dev
```

## Main Commands

Preprocess:

```bash
poetry run python scripts/preprocess.py dataset.preprocessing.raw_file=data/raw/BNaT.csv federated.num_clients=10
```

Central/local training:

```bash
poetry run python scripts/train.py experiment=baseline seed=42 training.epochs=100
```

Federated simulation:

```bash
poetry run python scripts/federated_train.py federated.num_clients=10 federated.num_rounds=50
```

Manual Flower server/client:

```bash
poetry run python scripts/federated_server.py
poetry run python scripts/federated_client.py federated.client_id=1 federated.client_data_path=data/processed/client_1_train.pt
```

Evaluation:

```bash
poetry run python scripts/evaluate.py checkpoint.path=outputs/run_id/best_model.pt
```

Regenerate plots without rerunning training:

```bash
poetry run python scripts/plot.py run_dir=outputs/run_id
```

This writes one image per experiment into `outputs/run_id/plots/`, plus a `plot_manifest.json` file that records the generated artifacts.

Compare runs:

```bash
poetry run python scripts/compare_runs.py runs='[outputs/run1,outputs/run2]'
```

Smoke test:

```bash
poetry run python scripts/smoke_test.py experiment=smoke device.prefer=cpu
```

## Outputs

Each run writes local artifacts under `outputs/<run_id>/`:

```text
run.log
debug.log
metrics.jsonl
metrics.csv
metadata.json
config.yaml
resolved_config.yaml
best_model.pt
latest_checkpoint.pt
final_model.pt
test_metrics.json
open_set_metrics.json
federated_round_metrics.csv
federated_history.csv
open_set_scores.csv
before_osr_confusion_matrix.csv
after_osr_confusion_matrix.csv
plots/
plots/plot_manifest.json
evt/
```

No W&B or online tracking service is required.

## Configuration

Hydra config groups live in `src/configs/`:

```text
dataset, model, agent, optimizer, scheduler, training, evaluation,
federated, open_set, plotting, tracking, checkpointing, logging, experiment
```

All important experiment parameters are controlled by config files or explicit Hydra overrides. Missing required values fail during script startup through `src.utils.config.validate_config`.

See `docs/` for the experiment protocol, plotting contract, checkpoint format, and reproducibility workflow.
