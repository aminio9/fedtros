# Reproducibility

## Determinism

`src.utils.utils.set_seed` seeds Python, NumPy, PyTorch, and CUDA. Config controls:

- `seed`
- `device.deterministic`
- `device.benchmark`
- `device.use_deterministic_algorithms`

DataLoader worker seeding utilities are available as `seed_worker` and `make_torch_generator`.

## Data Leakage Controls

Preprocessing fits scaler/encoder objects on known training rows only. Unknown samples are not used for training and are only appended to open-set evaluation tensors.

## Traceability

Every figure and metric should be traceable to:

- `resolved_config.yaml`
- `metrics.jsonl` / `metrics.csv`
- checkpoint files
- evaluation JSON/CSV outputs
- plot source data files

## Recommended Reproduction

```bash
poetry run python scripts/preprocess.py seed=42
poetry run python scripts/federated_train.py seed=42 federated.num_clients=10 federated.num_rounds=50
poetry run python scripts/evaluate.py checkpoint.path=outputs/run_id/best_model.pt
poetry run python scripts/plot.py run_dir=outputs/run_id
```
