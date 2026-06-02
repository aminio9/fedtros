# E3 Federated Non-IID Runbook

## Objective

Compare FedAvg, FedProx, FMRL-AVA, and the centralized pooled baseline under matched Dirichlet partitions.

## Hydra Config Used

`experiment=exp3` with `dataset.preprocessing.alpha=0.1` or `10.0`.

## Override Examples

```bash
python run.py experiment=exp3 +method=fmrl_ava seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=fedavg seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=fedprox seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=centralized_no_osr seed=42 dataset.preprocessing.alpha=0.1
```

## Execution Commands

```bash
python run.py experiment=exp3 +method=fmrl_ava seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=fedavg seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=fedprox seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=centralized_no_osr seed=42 dataset.preprocessing.alpha=0.1 tracking.run_id=e3_central_alpha0.1_seed42
bash scripts/experiments/e3_federated_noniid.sh
```

## Expected Outputs

- `federated_history.csv`
- `federated_round_metrics.csv`
- `communication_metrics.csv`
- `evaluation_metrics.json`

## Checkpoints

- `global_model_round_*.pt`
- `global_model_latest.pt`
- `best_model.pt`
- `latest_checkpoint.pt`

## Logs

- `run.log`
- `debug.log`
- `fmrl_ava_monitoring.jsonl` for FMRL-AVA

## Metrics

- `test/accuracy`
- `test/macro_f1`
- `test/balanced_accuracy`
- `federated/rounds`
- `federated/flower_rounds`
- `fmrl_ava_validation_reward`
- `fmrl_ava_support_reward`
- `fmrl_ava_team_reward_target`
- `alignment_cosine` and `alignment_multiplier` inside `fmrl_ava_monitoring.jsonl`

## Artifacts

- `processed/`
- `plots/`
- `plot_manifest.json`

## Validation

- Confirm the same seed and alpha are reused across all methods.
- Confirm `federated.resume_from` is available for restartable rounds.
- Confirm the client count matches the preprocessing partition count.

## Troubleshooting

- If client data are missing, rerun preprocessing first.
- If FMRL-AVA selects no clients, check `strategy.utility_threshold` and
  `strategy.min_selected_clients`.
- If FMRL-AVA behaves like FedAvg under hard non-IID, inspect `fmrl_ava_monitoring.jsonl`
  for near-constant utilities, near-constant alignment multipliers, or
  `strategy.alignment_strength=0.0`.
- If communication metrics are empty, verify `federated_history.csv` exists.
