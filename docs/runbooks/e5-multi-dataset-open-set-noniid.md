# E5 Multi-Dataset Open-Set Non-IID Validation

## Objective

Validate the model independently on B-TAT, ToN-IoT, and CIC-IDS2017 under open-set and non-IID conditions.

## Hydra Config Used

`experiment=exp5` with dataset-specific raw paths and finalized known/unknown label maps.

## Override Examples

```bash
python run.py experiment=exp5 +method=fmrl_ava dataset.name=B-TAT seed=42
python run.py experiment=exp5 +method=fmrl_ava dataset.name=ToN-IoT seed=42
python run.py experiment=exp5 +method=fedavg dataset.name=CIC-IDS2017 seed=42
```

## Execution Commands

```bash
python run.py experiment=exp5 +method=fmrl_ava dataset.name=B-TAT seed=42
python run.py experiment=exp5 +method=fmrl_ava dataset.name=ToN-IoT seed=42
python run.py experiment=exp5 +method=fmrl_ava dataset.name=CIC-IDS2017 seed=42
bash scripts/experiments/e5_multi_dataset_open_set_noniid.sh
```

## Expected Outputs

- `evaluation_metrics.json`
- `open_set_metrics.json`
- `open_set_scores.csv`
- `open_set_roc_curve.csv`
- `open_set_pr_curve.csv`
- `communication_metrics.csv`
- `latent_embeddings.csv`

## Validation

- Confirm each dataset is trained, tuned, and evaluated independently.
- Confirm B-NAT is not used in this block.
- Confirm the known/unknown label maps are finalized before execution.
