# CF-MARLOS-AVA
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
poetry run python run.py experiment=exp1 +method=fmrl_ava seed=42
poetry run python run.py experiment=all
```

Proposed aggregation method: `FMRL-AVA` (`Federated Multi-Agent Reinforcement Learning with Adaptive Vector-Aligned Aggregation`); Hydra alias: `+method=fmrl_ava`.

FedGPA baseline/adaptation: `+method=fedgpa` adds prototype-personalized aggregation plus RL stabilizers for non-IID clients: class-balanced reward, slower epsilon decay, and a small auxiliary CE loss on Q-values. See `docs/fedgpa-implementation.md` and `docs/rl-stability-fixes.md`.

Example FedGPA non-IID run:

```bash
poetry run python run.py experiment=exp3 +method=fedgpa seed=42 \
  federated.num_clients=3 federated.num_rounds=10 training.local_episodes_per_round=10 \
  dataset.preprocessing.alpha=0.1
```

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


## DKD-FedOS method

This repository now includes `+method=dkd_fedos`, a Sentinel-inspired dynamic knowledge-distillation strategy for extreme non-IID blockchain traffic clients. The existing CVAE-DQN agent is used as a personalized local teacher, while a compact student classifier is shared globally. The server aggregates only student models using equal-weight normalized pseudo-gradient aggregation with momentum. See `docs/dkd-fedos-implementation.md`.

Example:

```bash
poetry run python run.py experiment=exp3 +method=dkd_fedos seed=42 \
  federated.num_clients=10 federated.num_rounds=100 \
  training.local_episodes_per_round=10 dataset.preprocessing.alpha=0.1
```


## DKD-FedOS v2

`+method=dkd_fedos` implements a Sentinel-inspired teacher/student federated IDS strategy for extreme non-IID blockchain traffic.

Key behavior:

- local CVAE-DQN teacher remains private and personalized;
- lightweight student is the only model uploaded to the server;
- server aggregates student pseudo-gradients with L2 normalization, equal weighting, and momentum;
- local teacher update, student update, and student-to-teacher KD are separated;
- absent-class local gradients are blocked, but global student KD can still teach missing classes;
- teacher and student are evaluated separately in logs.

Smoke test:

```bash
poetry run python run.py experiment=exp1 +method=dkd_fedos seed=42 \
  federated.num_clients=3 federated.num_rounds=3 \
  training.local_episodes_per_round=5 dataset.preprocessing.iid=true
```

Non-IID run:

```bash
poetry run python run.py experiment=exp3 +method=dkd_fedos seed=42 \
  federated.num_clients=10 federated.num_rounds=100 \
  training.local_episodes_per_round=10 dataset.preprocessing.alpha=0.1
```

See `docs/dkd-fedos-implementation.md` for the full method mapping and logging checklist.

### DKD-FedOS v3 smoke command

```bash
poetry run python run.py experiment=exp1 +method=dkd_fedos seed=42 \
  federated.num_clients=3 federated.num_rounds=8 \
  training.local_episodes_per_round=10 dataset.preprocessing.iid=true \
  dataset.preprocessing.validation_split=0.1
```

In the log, check that `GLOBAL_STUDENT_AFTER_SERVER_AGG` is not stuck at `0.0714` accuracy and that `prediction_max_ratio` is not `1.0`.

### DKD-FedOS v4 RL-safe dataset DKD

The DKD dataset mini-batch phase is now student-only by default. It reads `env.all_features_s` and `env.all_labels_a_t` but does not step the environment, does not add to replay buffer, and does not update the CVAE-DQN teacher unless explicitly enabled.

Safe defaults:

```yaml
training.dkd_dataset_update_teacher=false
training.dkd_update_teacher_from_student=false
training.dkd_teacher_task_weight=0.0
training.dkd_student_to_teacher_start_round=999
```

So the RL teacher is still trained by the normal replay-buffer path, while the dataset DKD phase trains only the shared student and aligner from a frozen teacher.

## DKD-FedOS v5

v5 adds global-student anchoring, reliability-weighted aggregation, a stronger student MLP, and before/after student diagnostics.  The RL teacher remains protected: dataset DKD trains the student and aligner by default, not the CVAE-DQN teacher.

Recommended debug run:

```bash
poetry run python run.py experiment=exp1 +method=dkd_fedos seed=42 \
  federated.num_clients=3 federated.num_rounds=8 \
  training.local_episodes_per_round=10 dataset.preprocessing.iid=true \
  dataset.preprocessing.validation_split=0.1 \
  2>&1 | tee dkd_fedos_v5_iid.log
```

For harsh non-IID, check `dkd_student_train_delta_norm`, `avg_dkd_global_anchor_loss`, `dkd_global_anchor_weight`, and server reliability weights before trusting accuracy.

### DKD-FedOS v7 open-set backend

The default open-set detector for E2/E4 is **global student feature-distance EVT** (`open_set.evt.backend=student_feature_evt`). The optional `dual_boundary_evt` backend adds support-gated local teacher-generator reconstruction EVT only for client-side ablation/debug. See `docs/dkd-fedos-open-set-v7-feature-dual-evt.md`.


### DKD-FedOS v7 per-round open-set evaluation

Open-set E2/E4 experiments now enable server-side global open-set evaluation after each aggregation round. The hook evaluates the aggregated global student with class-wise Feature-EVT and writes a round curve to `open_set_round_metrics.csv` plus per-round artifacts under `open_set_rounds/`. For `dual_boundary_evt`, this server-side round evaluation uses the global Feature-EVT boundary; the local generator boundary remains a client-side ablation because local teacher/generator modules are not uploaded.

### Fed-DiGOS open-set backend

E2/E4 now use **Fed-DiGOS**, a federated student-attached open-set generator branch with EVT-calibrated generator, energy, and prototype scores. The private RL teacher generator remains local and its standalone generator training is disabled for the main method. See `docs/fed-digos-implementation-plan.md`.
