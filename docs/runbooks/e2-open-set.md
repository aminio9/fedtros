# E2 Open-Set Runbook

## Objective

Measure EVT-based unknown rejection on B-NAT with `FoT` as the held-out attack.

## Hydra Config Used

`experiment=exp2` with `open_set.evt.enabled=true` and the chosen method overlay.

## Override Examples

```powershell
python run.py experiment=exp2 +method=fmrl_la seed=42
python run.py experiment=exp2 +method=fmrl_la open_set.evt.enabled=false seed=42
python run.py experiment=exp2 +method=centralized_osr seed=42
```

## Execution Commands

```powershell
python run.py experiment=exp2 +method=fmrl_la seed=42
python run.py experiment=exp2 +method=centralized_osr seed=42
python scripts/experiments/e2_open_set.ps1
```

## Expected Outputs

- `open_set_metrics.json`
- `open_set_scores.csv`
- `open_set_roc_curve.csv`
- `open_set_pr_curve.csv`
- `before_osr_confusion_matrix.csv`
- `after_osr_confusion_matrix.csv`

## Checkpoints

- `best_model.pt`
- `latest_checkpoint.pt`
- `evt/evt_models.pkl`
- `evt/evt_meta.json`

## Logs

- `run.log`
- `debug.log`
- `metrics.jsonl`
- `metrics.csv`

## Metrics

- `open_set/auroc`
- `open_set/auprc`
- `open_set/fpr95`
- `open_set/unknown_detection_rate`
- `open_set/unknown_f1`

## Artifacts

- `latent_embeddings.csv`
- `plots/plot_manifest.json`
- `processed/`

## Validation

- Confirm validation-only EVT calibration uses `validation.pt`.
- Confirm `open_set_scores.csv` contains `y_true`, `raw_pred`, `y_pred`,
  `unknown_score`, and `is_unknown`.

## Troubleshooting

- If EVT calibration falls back to test data, check `validation.pt`.
- If unknown rejection never triggers, inspect `open_set.evt.decision_threshold`.
- If latent export is missing, check `evaluation.export_latent_embeddings`.

