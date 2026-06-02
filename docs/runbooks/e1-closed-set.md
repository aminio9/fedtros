# E1 Closed-Set Runbook

## Objective

Verify that the unified model preserves performance across the full B-NAT label set, with no unknown labels held out.

## Hydra Config Used

`experiment=exp1` with optional method overlays:

- `+method=fmrl_ava`
- `+method=fedavg`
- `+method=fedprox`
- `+method=centralized_no_osr` (commented reference only in the shell script)
- `dataset.known_labels=${dataset.source_labels}`

Default run-local paths:

- `dataset.preprocessing.output_dir=${tracking.run_dir}/processed`
- `dataset.preprocessing.iid=true`
- `tracking.run_id=e1_${experiment.method}_iid_seed${seed}`

`tracking.*` is derived from the root `output.*` config values rather than a separate config group.

## Override Examples

```bash
python run.py experiment=exp1 +method=fmrl_ava seed=42
python run.py experiment=exp1 +method=fedavg seed=42
python run.py experiment=exp1 +method=fedprox seed=42
# python run.py experiment=exp1 +method=centralized_no_osr seed=42
```

## Execution Commands

```bash
python run.py experiment=exp1 +method=fmrl_ava seed=42
python run.py experiment=exp1 +method=fedavg seed=42
python run.py experiment=exp1 +method=fedprox seed=42
# python run.py experiment=exp1 +method=centralized_no_osr seed=42
bash scripts/experiments/e1_closed_set.sh
```

## Expected Outputs

- `evaluation_metrics.json`
- `test_metrics.json`
- `test_confusion_matrix.csv`
- `federated_history.csv`
- `communication_metrics.csv`

## Checkpoints

- `best_model.pt`
- `latest_checkpoint.pt`
- `final_model.pt`

## Logs

- `run.log`
- `debug.log`
- `metrics.jsonl`
- `metrics.csv`
- `metadata.json`

## Metrics

- `test/accuracy`
- `test/balanced_accuracy`
- `test/macro_f1`
- `test/weighted_f1`

## Artifacts

- `resolved_config.yaml`
- `processed/`
- `plots/`
- `plot_manifest.json`

## Validation

- Confirm `resolved_config.yaml` records the chosen method overlay.
- Confirm `resolved_config.yaml` records `dataset.preprocessing.iid: true`.
- Confirm `evaluation_metrics.json` contains closed-set metrics.
- Confirm the run-local `processed/` directory exists.

## Troubleshooting

- If preprocessing fails, check `dataset.preprocessing.raw_file`.
- If the run reuses the wrong output directory, inspect `tracking.run_id`.
- If the checkpoint is missing, check `checkpointing.save_best` and
  `checkpointing.save_latest`.
