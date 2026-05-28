# E7 Label-Wise Open-Set Runbook

## Objective

Measure open-set separation with one held-out attack label per run, then save the latent tensor for plotting.

## Hydra Config Used

`experiment=exp7` with `open_set.evt.enabled=true`, latent export enabled, and the known-label list overridden per held-out attack.

## Override Examples

```bash
python run.py experiment=exp7 +method=fmrl_la seed=42 dataset.known_labels='[Normal,BP,DoS,FoT]' tracking.run_id=e7_mitm_fmrl_la_seed42
python run.py experiment=exp7 +method=fmrl_la seed=42 dataset.known_labels='[Normal,BP,DoS,MitM]' tracking.run_id=e7_fot_fmrl_la_seed42
```

## Execution Commands

```bash
python run.py experiment=exp7 +method=fmrl_la seed=42 dataset.known_labels='[Normal,BP,DoS,FoT]'
python run.py experiment=exp7 +method=fmrl_la seed=42 dataset.known_labels='[Normal,BP,DoS,MitM]'
python run.py experiment=exp7 +method=fedavg seed=42 dataset.known_labels='[Normal,BP,DoS,FoT]'
python run.py experiment=exp7 +method=fedavg seed=42 dataset.known_labels='[Normal,BP,DoS,MitM]'
bash scripts/experiments/e7_labelwise_open_set.sh
```

## Expected Outputs

- `open_set_metrics.json`
- `open_set_scores.csv`
- `latent_embeddings.csv`
- `latent_embeddings.json`
- `open_set_roc_curve.csv`
- `open_set_pr_curve.csv`

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

- `open_set/auroc`
- `open_set/auprc`
- `open_set/fpr95`
- `open_set/unknown_f1`

## Artifacts

- `latent_embeddings.csv`
- `latent_embeddings.json`
- `processed/`

## Validation

- Confirm each run writes its own `latent_embeddings.csv` and `latent_embeddings.json`.
- Confirm the latent export includes `source=open_set` in the metadata.
- Confirm the exported rows come from the open-set evaluation tensor only.
- Confirm unknown labels are encoded as `Unknown` in the latent plot.

## Troubleshooting

- If the latent plot shows duplicated known samples, check that the export came from the open-set tensor only.
- If the plot is missing unknown points, check `dataset.known_labels` for the held-out label and `evaluation.open_set_data`.
