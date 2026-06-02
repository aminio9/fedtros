# E8 Label-Wise Open-Set Runbook

## Objective

Measure open-set separation with one held-out attack label per run, then save the latent tensor for plotting.

## Hydra Config Used

`experiment=exp8` with `open_set.evt.enabled=true`, latent export enabled, and the known-label list overridden per held-out attack.

## Override Examples

```bash
python run.py experiment=exp8 +method=fmrl_ava seed=42 dataset.known_labels='[Normal,BP,DoS,FoT]' tracking.run_id=e8_mitm_fmrl_ava_seed42
python run.py experiment=exp8 +method=fmrl_ava seed=42 dataset.known_labels='[Normal,BP,DoS,MitM]' tracking.run_id=e8_fot_fmrl_ava_seed42
```

## Execution Commands

```bash
python run.py experiment=exp8 +method=fmrl_ava seed=42 dataset.known_labels='[Normal,BP,DoS,FoT]'
python run.py experiment=exp8 +method=fmrl_ava seed=42 dataset.known_labels='[Normal,BP,DoS,MitM]'
python run.py experiment=exp8 +method=fedavg seed=42 dataset.known_labels='[Normal,BP,DoS,FoT]'
python run.py experiment=exp8 +method=fedavg seed=42 dataset.known_labels='[Normal,BP,DoS,MitM]'
bash scripts/experiments/e8_labelwise_open_set.sh
```

## Expected Outputs

- `open_set_metrics.json`
- `open_set_scores.csv`
- `latent_embeddings.csv`
- `latent_embeddings.json`
- `open_set_roc_curve.csv`
- `open_set_pr_curve.csv`

## Validation

- Confirm each run writes its own `latent_embeddings.csv` and `latent_embeddings.json`.
- Confirm the latent export includes `source=open_set` in the metadata.
- Confirm the exported rows come from the open-set evaluation tensor only.
- Confirm unknown labels are encoded as `Unknown` in the latent plot.
