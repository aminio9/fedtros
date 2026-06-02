# E4 Open-Set Under Non-IID Runbook

## Objective

Measure the full system, no-EVT control, and centralized baselines when unknown rejection and client skew are active.

## Hydra Config Used

`experiment=exp4` with `dataset.preprocessing.alpha=0.1` or `10.0`.

## Override Examples

```bash
python run.py experiment=exp4 +method=fmrl_ava seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp4 +method=fmrl_ava seed=42 dataset.preprocessing.alpha=0.1 open_set.evt.enabled=false experiment.method=ClosedSet_No_EVT tracking.run_id=e4_no_evt_alpha0.1_seed42
python run.py experiment=exp4 +method=fedavg seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp4 +method=fedprox seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp4 +method=centralized_osr seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp4 +method=centralized_no_osr seed=42 dataset.preprocessing.alpha=0.1
```

## Execution Commands

```bash
python run.py experiment=exp4 +method=fmrl_ava seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp4 +method=fmrl_ava seed=42 dataset.preprocessing.alpha=0.1 open_set.evt.enabled=false experiment.method=ClosedSet_No_EVT
python run.py experiment=exp4 +method=fedavg seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp4 +method=fedprox seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp4 +method=centralized_osr seed=42 dataset.preprocessing.alpha=0.1 tracking.run_id=e4_central_osr_alpha0.1_seed42
python run.py experiment=exp4 +method=centralized_no_osr seed=42 dataset.preprocessing.alpha=0.1 tracking.run_id=e4_central_no_osr_alpha0.1_seed42
bash scripts/experiments/e4_combined_open_set_noniid.sh
```

## Expected Outputs

- `evaluation_metrics.json`
- `open_set_metrics.json`
- `open_set_scores.csv`
- `open_set_roc_curve.csv`
- `open_set_pr_curve.csv`
- `communication_metrics.csv`

## Checkpoints

- `best_model.pt`
- `latest_checkpoint.pt`
- `evt/evt_models.pkl`

## Logs

- `run.log`
- `debug.log`
- `metrics.jsonl`
- `metrics.csv`

## Metrics

- `test/macro_f1`
- `open_set/auroc`
- `open_set/auprc`
- `open_set/fpr95`
- `open_set/unknown_f1`

## Artifacts

- `before_osr_confusion_matrix.csv`
- `after_osr_confusion_matrix.csv`
- `latent_embeddings.csv`
- `processed/`

## Validation

- Confirm EVT is enabled and calibrated on validation data only.
- Confirm the same partition seed is reused across methods.
- Confirm the open-set scores file includes unknown labels encoded as `-1`.

## Troubleshooting

- If open-set metrics are missing, check `open_set.evt.enabled`.
- If the run writes to the shared `data/processed/` folder, override
  `dataset.preprocessing.output_dir` to a run-local path.
- If plots show missing data, build the suite CSVs first.
