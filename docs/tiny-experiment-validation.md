# Tiny Experiment Validation

Use this document to check the full pipeline without committing to the full
Q1 budget. The goal is to verify that preprocessing, federated training,
evaluation, checkpointing, and plotting all work together on a tiny run.

## 1. Target

Run a minimal end-to-end experiment with:

- 2 clients
- 1 logical federated round
- 1 local episode per round
- 3 steps per episode
- CPU execution
- a run-local processed-data directory

This keeps the run small enough for fast validation while still exercising the
whole stack.

## 2. Command

Run from the repository root:

```powershell
cd D:\Research\cf_marlos

# 1️⃣ Assign the configuration to a variable
BASE="dataset=bnat \
model=openset_qchain \
agent=double_q \
optimizer=adam \
scheduler=none \
experiment=baseline \
device.prefer=cpu \
dataset.known_labels=[Normal,BP,DoS,MitM] \
dataset.preprocessing.known_labels=[Normal,BP,DoS,MitM] \
model.num_actions=4 \
model.state_dim=31 \
training.batch_size=2 \
training.local_episodes_per_round=1 \
training.steps_per_episode=3 \
training.min_buffer_size=2 \
open_set.evt.enabled=true \
federated.num_clients=2 \
dataset.preprocessing.num_clients=2 \
federated.num_rounds=1 \
evaluation.batch_size=4096"

# 2️⃣ Run the experiment
poetry run python scripts/reproduce_experiment.py $BASE \
tracking.run_id=tiny_e2e_validation \
dataset.preprocessing.output_dir=outputs/tiny_e2e_validation/processed \
checkpointing.dir=outputs/tiny_e2e_validation \
checkpointing.best_model_path=outputs/tiny_e2e_validation/best_model.pt \
checkpointing.latest_checkpoint_path=outputs/tiny_e2e_validation/latest_checkpoint.pt \
checkpoint.path=outputs/tiny_e2e_validation/latest_checkpoint.pt \
evaluation.checkpoint_path=outputs/tiny_e2e_validation/best_model.pt \
dataset.preprocessing.iid=false \
dataset.preprocessing.alpha=0.1

# 3️⃣ Evaluate the model
poetry run python scripts/evaluate.py $BASE \
tracking.run_id=tiny_e2e_validation_eval \
dataset.preprocessing.output_dir=outputs/tiny_e2e_validation/processed \
checkpoint.path=outputs/tiny_e2e_validation/best_model.pt \
evaluation.checkpoint_path=outputs/tiny_e2e_validation/best_model.pt

# 4️⃣ Plot results
poetry run python scripts/plot.py \
run_dir=outputs/tiny_e2e_validation_eval
```

## 3. Expected Artifacts

Training run:

```text
outputs/tiny_e2e_validation/
  run.log
  debug.log
  metrics.csv
  metrics.jsonl
  metadata.json
  config.yaml
  resolved_config.yaml
  best_model.pt
  latest_checkpoint.pt
  global_model_round_0002.pt
```

The processed tensors should be isolated under:

```text
outputs/tiny_e2e_validation/processed/
```

That keeps the canonical `data/processed/` files intact.

Evaluation and plot run:

```text
outputs/tiny_e2e_validation_eval/
  run.log
  debug.log
  metrics.csv
  metrics.jsonl
  metadata.json
  config.yaml
  resolved_config.yaml
  evaluation_metrics.json
  test_metrics.json
  open_set_metrics.json
  open_set_scores.csv
  open_set_roc_curve.csv
  open_set_pr_curve.csv
  latent_embeddings.csv
  before_osr_confusion_matrix.csv
  after_osr_confusion_matrix.csv
  evt/
  plots/
    plot_manifest.json
```

The training run also writes `communication_metrics.csv` when federated history
rows are available.

## 4. Acceptance Checks

Confirm these after the run:

- `resolved_config.yaml` contains the tiny overrides.
- `metadata.json` records the seed, device, dataset, model, and method.
- `best_model.pt` and `latest_checkpoint.pt` exist.
- `evaluation_metrics.json` exists in the evaluation run directory and contains both closed-set and open-set values.
- `open_set_scores.csv` exists and has the unknown-score columns.
- `latent_embeddings.csv` exists when latent export is enabled.
- `plots/plot_manifest.json` exists in the evaluation run directory.
- The run log shows preprocessing, training, and evaluation in sequence.

## 5. Optional Follow-Up Plot Render

The evaluation and plotting steps can be run separately after the training run
finishes. This keeps the tiny validation practical even when the full B-NAT
evaluation takes longer than the federated phase itself.

Some figures will still require suite-level CSVs, but the run should at least
prove that the plotting entrypoint attaches to an existing run and writes a
manifest.

## 6. What This Does Not Prove

This validation does not prove the final Q1 claims. It only checks that:

- the pipeline works end to end,
- the key files are emitted,
- the configs are wired correctly,
- the open-set evaluation path runs,
- the run-local data layout is safe.
