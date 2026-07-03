# CF-MARLOS-AVA
<<<<<<< HEAD
Adaptive Vector-Aligned Federated Multi-Agent Reinforcement Learning for Open-Set Blockchain Intrusion Detection

This repository implements the `cf_marlos` research pipeline for cooperative federated multi-agent reinforcement learning, Double Q-learning, EVT open-set recognition, and Flower-based federated training on B-NAT blockchain traffic.

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
=======

Federated CVAE-DQN training with EVT open-set rejection for blockchain intrusion-detection experiments. The repository is a research codebase, not a packaged product: commands are Hydra-driven, artifacts are written locally, and full experiments are expensive.

## Directory Structure

```text
src/
  agents/          Agent training step and epsilon-greedy policy
  checkpointing/   checkpoint payloads, metadata, and validation-only selection
  configs/         Hydra config groups for data, experiment, method, model, runtime, training, OSR
  data/            preprocessing, tensor loading, and split helpers
  evaluation/      closed-set metrics, open-set EVT evaluation, evaluation runner
  federated/       Flower client/server strategies, including FMRL-AVA and FedMADE-style weighting
  models/          CVAE-DQN stack and shared prediction adapter
  openset/         EVT models, scorer utilities, validation thresholding
  plotting/        plot registry and local renderers
  rl/              tabular classification environment and local replay training loop
  tracking/        local run tracker
  training/        centralized runner, reusable losses, smoke runner
  utils/           config validation, imbalance, device, and reproducibility helpers
scripts/           entry points, cheap validation, experiment shell scripts
tests/             unit and smoke tests for configs, training, OSR, aggregation, losses
docs/              method, training, evaluation, logging, and reproducibility notes
```

## Install
>>>>>>> ea28efe (Initial commit with updated source code)

```bash
poetry install
poetry install -E dev
```

<<<<<<< HEAD
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
poetry run python run.py experiment=exp1 +method=fmrl_ava seed=42
poetry run python run.py experiment=all
```

Proposed aggregation method: `FMRL-AVA` (`Federated Multi-Agent Reinforcement Learning with Adaptive Vector-Aligned Aggregation`); Hydra alias: `+method=fmrl_ava`.

Federated simulation:

```bash
poetry run python scripts/federated_train.py runtime=cpu federated.num_clients=10 federated.num_rounds=50
```

Manual Flower server/client:

```bash
poetry run python scripts/federated_server.py runtime=cpu
poetry run python scripts/federated_client.py runtime=cpu federated.client_id=1 federated.client_data_path=data/processed/client_1_train.pt
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

Evaluation now also writes `latent_embeddings.csv` by default when `evaluation.export_latent_embeddings=true`; open-set runs export the active evaluation tensor only, which keeps label-wise latent plots from duplicating closed-set rows.

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
poetry run python scripts/smoke_test.py experiment=smoke runtime=tiny
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
latent_embeddings.json
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
=======
If Poetry extras are unavailable in the local environment, install the project dependencies from `pyproject.toml` and ensure `pytest` is available.

## Quick Validation

Run these before any full experiment:

```bash
python scripts/cheap_validation.py
python -m pytest
python run.py experiment=validation runtime=tiny seed=42 tracking.run_id=tiny_validation_seed42
```

`experiment=validation runtime=tiny` uses CPU settings, caps preprocessing rows, and runs one tiny federated round. It is a preflight, not a benchmark result.

For a source-level map of what actually runs, see `docs/source_code_map.md`.

## Main Commands

```bash
python run.py experiment=validation runtime=tiny seed=42
python run.py experiment=exp1 +method=fmrl_ava seed=42
python run.py experiment=exp3 +method=fmrl_ava_glow seed=42 dataset.preprocessing.alpha=0.1 runtime=tiny output=tiny
python run.py experiment=exp3 +method=fedmade seed=42 dataset.preprocessing.alpha=0.1
bash scripts/experiments/validate_fmrl_ava_glow_tiny.sh
python scripts/cheap_validation.py
```

Shell scripts live in `scripts/experiments/`. `run_full_suite.sh` and `e7_efficiency_scalability.sh` are intentionally expensive; run the cheap validation commands first.

## Config Groups

Hydra configs live under `src/configs/`:

```text
checkpointing/ data via dataset/ experiment/ federated/ method/ model/
open_set/ runtime/ training/ evaluation/ optimizer/ output/ plotting/
```

Important boundaries:

- model configs define architecture fields.
- method configs select training/aggregation/loss behavior.
- open-set configs define EVT/scorer/threshold settings.
- runtime configs define scale and device behavior.
- experiment configs combine dataset split, pipeline, tracking, and expected outputs.

`src.utils.config.validate_config` checks required keys, model/action dimensions, EVT tail semantics, threshold mode, scorer names, strategy names, non-negative loss weights, validation-only checkpoint monitor metrics, and unsupported model/scorer/pipeline combinations.

## Outputs

Runs write under `outputs/<run_id>/` unless overridden:

```text
config.yaml
resolved_config.yaml
metadata.json
metrics.jsonl
metrics.csv
run.log
debug.log
latest_checkpoint.pt
last_model.pt
best_model.pt
final_model.pt
checkpoint_metadata.json
federated_history.csv
federated_round_metrics.csv
open_set_metrics.json
open_set_scores.csv
open_set_roc_curve.csv
open_set_pr_curve.csv
open_set_oscr_curve.csv
before_osr_confusion_matrix.csv
after_osr_confusion_matrix.csv
evt/
plots/
```

No online tracker is required.

## Status

Implemented: CVAE-DQN agent training, centralized/federated runners, FedAvg, FedProx, FMRL-AVA, FMRL-AVA-GLOW, FedMADE-style class-aware aggregation, EVT reconstruction open-set evaluation, checkpoint metadata, local plotting, and cheap validation.

Implemented as tested building blocks but not yet first-class experiment pipelines: MSP, energy, prototype-distance, and Mahalanobis scorer baselines.

Do not report state-of-the-art, cross-dataset, or RL-superiority claims without the matched multi-seed baseline suite and confidence intervals.
>>>>>>> ea28efe (Initial commit with updated source code)
