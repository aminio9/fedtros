# E1 Closed-Set Runbook

## Objective

Verify that the unified model preserves known-class performance on B-NAT.

## Hydra Config Used

`experiment=exp1` with optional method overlays:

- `+method=fmrl_la`
- `+method=fedavg`
- `+method=fedprox`
- `+method=centralized_no_osr`

Default run-local paths:

- `dataset.preprocessing.output_dir=${tracking.run_dir}/processed`
- `tracking.run_id=e1_${experiment.method}_alpha${dataset.preprocessing.alpha}_seed${seed}`

## Override Examples

```powershell
python run.py experiment=exp1 +method=fmrl_la seed=42
python run.py experiment=exp1 +method=fedavg seed=42
python run.py experiment=exp1 +method=centralized_no_osr seed=42
```

## Execution Commands

```powershell
python run.py experiment=exp1 +method=fmrl_la seed=42
python run.py experiment=exp1 +method=fedavg seed=42
python run.py experiment=exp1 +method=fedprox seed=42
python run.py experiment=exp1 +method=centralized_no_osr seed=42
python scripts/experiments/e1_closed_set.ps1
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
- Confirm `evaluation_metrics.json` contains closed-set metrics.
- Confirm the run-local `processed/` directory exists.

## Troubleshooting

- If preprocessing fails, check `dataset.preprocessing.raw_file`.
- If the run reuses the wrong output directory, inspect `tracking.run_id`.
- If the checkpoint is missing, check `checkpointing.save_best` and
  `checkpointing.save_latest`.

