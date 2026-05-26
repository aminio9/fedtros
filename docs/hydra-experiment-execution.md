# Hydra Experiment Execution

This repository keeps the existing Hydra config manager and extends it with
runtime, output, sweep, and method overlays. The execution path is still the
same code base:

`run.py` -> preprocessing -> federated or centralized training -> evaluation -> optional plot/export stages.

## Config Tree

```text
src/configs/config.yaml
  dataset: dataset/bnat.yaml
  model: model/openset_qchain.yaml | model/transformer.yaml
  agent: agent/double_q.yaml
  optimizer: optimizer/adam.yaml
  scheduler: scheduler/none.yaml
  training: training/default.yaml
  evaluation: evaluation/default.yaml | evaluation/closed_set.yaml | evaluation/open_set.yaml
  federated: federated/default.yaml
  open_set: open_set/evt.yaml
  plotting: plotting/q1_dashboard.yaml
  tracking: tracking/local.yaml
  checkpointing: checkpointing/default.yaml
  logging: logging/default.yaml
  runtime: runtime/cpu.yaml | runtime/gpu.yaml | runtime/directml.yaml | runtime/tiny.yaml
  output: output/local.yaml | output/tiny.yaml
  sweep: sweep/none.yaml | sweep/seeds.yaml | sweep/alpha.yaml | sweep/clients.yaml
  experiment: experiment/baseline.yaml | experiment/validation.yaml | experiment/exp1.yaml ... experiment/all.yaml
  optional overlays: method/*, evaluation/*, CLI overrides
```

## Inheritance Flow

1. `config.yaml` composes the base system.
2. Dataset/model/training/evaluation/federated groups set domain defaults.
3. `runtime/*` feeds `device.*` and client resource defaults.
4. `output/*` feeds `tracking.output_dir` and run-id templates.
5. `experiment/*` sets the concrete experiment objective, pipeline, and
   run-local output directory.
6. Optional `method/*` overlays set `experiment.method` and federated strategy
   names without duplicating the experiment configs.
7. CLI overrides win last.

## Standard Commands

Single run:

```powershell
python run.py experiment=exp1
python run.py experiment=exp2 dataset=bnat model=transformer
python run.py experiment=ablation runtime=gpu
python run.py experiment=validation runtime=tiny
```

Hydra multirun:

```powershell
python run.py --multirun experiment=all
python run.py --multirun experiment=exp3 +method=fmrl_la,fedavg,fedprox
python run.py --multirun experiment=exp4 +method=fmrl_la seed=42,43,44
```

Direct scripts still work and use the same config tree:

```powershell
python scripts/preprocess.py
python scripts/federated_train.py
python scripts/evaluate.py
python scripts/plot.py
```

## Override Patterns

Use overrides instead of duplicated configs:

```powershell
dataset.preprocessing.alpha=0.1
dataset.preprocessing.iid=false
federated.num_clients=10
federated.num_rounds=100
open_set.evt.enabled=true
training.generator.enabled=false
federated.resume_from=outputs/run_id/latest_checkpoint.pt
checkpoint.path=outputs/run_id/best_model.pt
evaluation.checkpoint_path=outputs/run_id/best_model.pt
tracking.run_id=my_run_name
```

## Experiment Dependency Map

| Experiment | Dependencies |
|---|---|
| E1 Closed-set | preprocessing -> federated training or centralized training -> closed-set evaluation |
| E2 Open-set | preprocessing -> federated training -> EVT calibration -> open-set evaluation |
| E3 Federated non-IID | preprocessing -> federated training -> closed-set evaluation |
| E4 Combined | preprocessing -> federated training -> EVT calibration -> open-set evaluation |
| E5 Ablation | preprocessing -> selected training variant -> evaluation |
| E6 Efficiency | preprocessing -> federated training -> communication export -> evaluation |

## Output Structure

Each run writes to `outputs/<run_id>/` and keeps preprocessing under a
run-local `processed/` directory:

```text
outputs/<run_id>/
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
  evaluation_metrics.json
  test_metrics.json
  open_set_metrics.json
  open_set_scores.csv
  open_set_roc_curve.csv
  open_set_pr_curve.csv
  before_osr_confusion_matrix.csv
  after_osr_confusion_matrix.csv
  latent_embeddings.csv
  communication_metrics.csv
  federated_history.csv
  plots/
  evt/
  processed/
```

## Data Flow

1. Raw B-NAT CSV is loaded from `dataset.preprocessing.raw_file`.
2. Known/unknown labels are split using `dataset.preprocessing.known_labels`.
3. Preprocessing writes tensor datasets and manifests into
   `dataset.preprocessing.output_dir`.
4. Training loads `known_train.pt` or the client shards.
5. Evaluation loads `validation.pt`, `shared_closed_set_test.pt`, and
   `shared_open_set_test.pt`.
6. EVT calibration uses validation data only.
7. Plots and suite exports read saved artifacts only.

## Reproducibility

- Fix `seed` and reuse it across all paired comparisons.
- Keep `dataset.preprocessing.output_dir` run-local.
- Use the same client count and alpha for all methods in a comparison.
- Save `resolved_config.yaml`, `metadata.json`, checkpoints, metrics, and
  plot manifests.
- Use `federated.resume_from` or `training.resume_from` for restartable runs.

## Result Export

Suite exports are built from saved run directories only:

```powershell
python run.py experiment.pipeline=export runs='[outputs/run1,outputs/run2]'
python scripts/build_suite_artifacts.py runs='[outputs/run1,outputs/run2]'
```
