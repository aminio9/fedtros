# E5 Ablation Runbook

## Objective

Isolate the contribution of EVT, generator training, and federated strategy.

## Hydra Config Used

`experiment=ablation` with overlayed toggles such as:

- `training.generator.enabled=false`
- `open_set.evt.enabled=false`
- `+method=fedavg`
- `+method=centralized_osr`
- `+method=centralized_no_osr`

## Override Examples

```powershell
python run.py experiment=ablation +method=fmrl_la seed=42
python run.py experiment=ablation +method=fmrl_la training.generator.enabled=false seed=42
python run.py experiment=ablation +method=fedavg open_set.evt.enabled=false seed=42
```

## Execution Commands

```powershell
python run.py experiment=ablation +method=fmrl_la seed=42
python run.py experiment=ablation +method=fmrl_la training.generator.enabled=false seed=42
python run.py experiment=ablation +method=fedavg open_set.evt.enabled=false seed=42
python run.py experiment=ablation +method=centralized_osr seed=42
python run.py experiment=ablation +method=centralized_no_osr seed=42
python scripts/experiments/e5_ablation.ps1
```

## Expected Outputs

- `evaluation_metrics.json`
- `open_set_metrics.json`
- `ablation_metrics.csv`
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

## Metrics

- `test/macro_f1`
- `open_set/auroc`
- `open_set/unknown_f1`
- `communication/cumulative_mb`

## Artifacts

- `resolved_config.yaml`
- `plots/`
- `plot_manifest.json`

## Validation

- Confirm each ablation changes only one mechanism at a time.
- Confirm centralized baselines use `method/centralized_*`.
- Confirm the reported comparison table reads saved artifacts only.

## Troubleshooting

- If `training.generator.enabled=false` is ignored, check the overlay order.
- If the centralized baseline still uses OSR, inspect `open_set.evt.enabled`.
- If the run id collides, override `tracking.run_id`.

