# Validation Runbook

## Objective

Check the full pipeline with a tiny client/round/episode budget.

## Hydra Config Used

`experiment=validation`

## Override Examples

```powershell
python run.py experiment=validation seed=42
python run.py experiment=validation runtime=tiny seed=42
```

## Execution Commands

```powershell
python run.py experiment=validation seed=42
python scripts/experiments/validate_configs.ps1
```

## Expected Outputs

- `evaluation_metrics.json`
- `open_set_metrics.json`
- `plots/plot_manifest.json`
- `processed/`
- `latent_embeddings.csv` only if `evaluation.export_latent_embeddings=true`

## Checkpoints

- `best_model.pt`
- `latest_checkpoint.pt`

## Logs

- `run.log`
- `debug.log`
- `metrics.jsonl`
- `metrics.csv`

## Metrics

- closed-set metrics
- open-set metrics

## Artifacts

- `communication_metrics.csv`

## Validation

- Confirm the run stays in the run-local output tree.
- Confirm only two clients and one logical round are used.

## Troubleshooting

- If the tiny run writes to the shared processed tree, override
  `dataset.preprocessing.output_dir`.
- If the validation run takes too long, use `runtime=tiny`.
