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
- `federated_history.csv` for round-level Flower metrics when federated simulation is used
- `fmrl_ava_monitoring.jsonl` for FMRL-AVA utilities, selected-client fraction, vector-alignment multipliers, final aggregation weights, support reward, and validation team reward
- `open_set_metrics.json` for EVT calibration metadata and the calibrated threshold used by plot 5
- checkpoint files
- evaluation JSON/CSV outputs
- plot source data files
- `plots/plot_manifest.json`

## Recommended Reproduction

```bash
poetry run python scripts/reproduce_experiment.py tracking.run_id=fmrl_alpha01_seed42 seed=42 runtime=cpu federated.num_clients=10 federated.num_rounds=100 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false dataset.preprocessing.output_dir=outputs/fmrl_alpha01_seed42/processed
poetry run python scripts/plot.py run_dir=outputs/fmrl_alpha01_seed42
```
