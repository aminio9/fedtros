# E6 Efficiency Runbook

## Objective

Quantify communication cost, runtime proxy, and accuracy versus client count and round budget.

## Hydra Config Used

`experiment=efficiency` with client-count and round-budget overrides or the sweep preset.

## Override Examples

```bash
python run.py experiment=efficiency +method=fmrl_la seed=42 federated.num_clients=3
python run.py experiment=efficiency +method=fmrl_la seed=42 federated.num_clients=10
python run.py experiment=efficiency +method=fedavg seed=42 federated.num_clients=3
python run.py experiment=efficiency +method=fedavg seed=42 federated.num_clients=3 federated.num_rounds=50
```

## Execution Commands

```bash
python run.py experiment=efficiency +method=fmrl_la seed=42 federated.num_clients=3
python run.py experiment=efficiency +method=fedavg seed=42 federated.num_clients=3
bash scripts/experiments/e6_efficiency_scalability.sh
```

## Expected Outputs

- `communication_metrics.csv`
- `federated_history.csv`
- `evaluation_metrics.json`
- `run_comparison.csv` when run as a suite

## Checkpoints

- `best_model.pt`
- `latest_checkpoint.pt`
- `global_model_latest.pt`

## Logs

- `run.log`
- `debug.log`
- `metrics.jsonl`
- `metrics.csv`

## Metrics

- `test/accuracy`
- `communication/cumulative_mb`
- `federated/rounds`
- `federated/flower_rounds`

## Artifacts

- `processed/`
- `plots/`
- `suite_artifacts_manifest.json`

## Validation

- Confirm the client count sweep uses the same seed.
- Confirm the round sweep uses the same seed.
- Confirm cumulative MB increases monotonically across rounds.
- Confirm the suite exporter can stage `communication_metrics.csv`.

## Troubleshooting

- If communication metrics are empty, confirm a checkpoint exists.
- If client counts do not change, inspect the Hydra override syntax.
- If the run takes too long on GPU, override `runtime=cpu` for validation.
