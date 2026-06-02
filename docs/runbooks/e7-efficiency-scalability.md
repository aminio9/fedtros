# E7 Efficiency Runbook

## Objective

Quantify communication cost, runtime proxy, and accuracy versus client count and round budget.

## Hydra Config Used

`experiment=exp7` with client-count and round-budget overrides or the sweep preset.

## Override Examples

```bash
python run.py experiment=exp7 +method=fmrl_ava seed=42 federated.num_clients=3
python run.py experiment=exp7 +method=fmrl_ava seed=42 federated.num_clients=10
python run.py experiment=exp7 +method=fedavg seed=42 federated.num_clients=3
python run.py experiment=exp7 +method=fedavg seed=42 federated.num_clients=3 federated.num_rounds=50
```

## Execution Commands

```bash
python run.py experiment=exp7 +method=fmrl_ava seed=42 federated.num_clients=3
python run.py experiment=exp7 +method=fedavg seed=42 federated.num_clients=3
bash scripts/experiments/e7_efficiency_scalability.sh
```

## Expected Outputs

- `communication_metrics.csv`
- `federated_history.csv`
- `evaluation_metrics.json`
- `run_comparison.csv` when run as a suite

## Validation

- Confirm the client count sweep uses the same seed.
- Confirm the round sweep uses the same seed.
- Confirm cumulative MB increases monotonically across rounds.
- For FMRL-AVA, compare selected-client fraction, alignment multipliers, and
  accuracy per MB against FedAvg; lower uploads are only useful if
  validation/open-set quality is not lost.
- Confirm the suite exporter can stage `communication_metrics.csv`.
