# CF-MARLOS-AVA

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

```bash
poetry install
poetry install -E dev
```

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
