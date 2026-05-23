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

Unified Hydra runner:

```bash
poetry run python run.py experiment=exp1 +method=fmrl_la seed=42
poetry run python run.py experiment=all
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

Evaluation now also writes `latent_embeddings.csv` by default when `evaluation.export_latent_embeddings=true`.

Compare runs:

```bash
poetry run python scripts/compare_runs.py runs='[outputs/run1,outputs/run2]'
```

`compare_runs.py` writes `comparison_metrics.csv` into `outputs/<suite_run>/` for the convergence plots. Other suite-level plot inputs must be staged into a dedicated suite run directory with the following schemas before plotting:

- `scalability.csv`: `num_clients,final_accuracy`
- `openness_metrics.csv`: `method,openness,auroc`
- `cross_dataset_metrics.csv`: `dataset,metric,metric_value`
- `seed_robustness.csv`: `seed,heterogeneity,accuracy`
- `latent_embeddings.csv`: `x,y,label`
- `communication_metrics.csv`: `method,cumulative_mb,accuracy`
- `ablation_metrics.csv`: `configuration,macro_f1`

Full suite aggregation:

```bash
poetry run python scripts/build_suite_artifacts.py runs='[outputs/run1,outputs/run2,outputs/run3]'
```

That command writes the suite CSVs above into `outputs/<suite_run>/` and records a `suite_artifacts_manifest.json` alongside them.

Experiment command files and batch scripts live in `scripts/experiments/`.

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
latent_embeddings.csv
communication_metrics.csv
plots/
plots/plot_manifest.json
evt/
suite_artifacts_manifest.json
```

No W&B or online tracking service is required.

## Configuration

Hydra config groups live in `src/configs/`:

```text
dataset, model, agent, optimizer, scheduler, training, evaluation,
federated, open_set, plotting, tracking, checkpointing, logging, runtime,
output, sweep, method, experiment
```

All important experiment parameters are controlled by config files or explicit Hydra overrides. Missing required values fail during script startup through `src.utils.config.validate_config`.

See `docs/` for the experiment protocol, plotting contract, checkpoint format, and reproducibility workflow.
