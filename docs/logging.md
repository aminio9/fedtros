# Logging And Tracking

Tracking is local-only through `src/tracking/local.py`.

Each initialized run writes under `tracking.run_dir`:

- `run.log`
- `debug.log`
- `metrics.jsonl`
- `metrics.csv`
- `metadata.json`
- `config.yaml`
- `resolved_config.yaml`

Federated runs can additionally write:

- `federated_history.csv`
- `federated_round_metrics.csv`
- `fmrl_ava_monitoring.jsonl` for FMRL-AVA
- `fedmade_monitoring.jsonl` for FedMADE-style class-aware aggregation

Checkpointing writes model payloads plus `checkpoint_metadata.json`; best checkpoint promotion also writes `best_metrics.json`.

## Metric Logs

Local training logs loss, reward, Q, KL, gradient, and learning-rate metrics. Generator training adds reconstruction and generator optimizer metrics when enabled.

Closed-set evaluation writes metrics, classification report, confusion matrix, and optional predictions. Open-set evaluation writes score CSVs, ROC/PR/OSCR curves, threshold-sensitivity CSVs, labeled confusion matrices, and EVT metadata.

No W&B, MLflow, or online service is required.
