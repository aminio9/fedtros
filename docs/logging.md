# Logging And Tracking

Tracking is local-only through `src/tracking/local.py`.

Each initialized run writes:

- `run.log`: normal execution log.
- `debug.log`: debug-level log.
- `metrics.jsonl`: append-only metrics.
- `metrics.csv`: table regenerated from JSONL.
- `metadata.json`: experiment name, run ID, timestamp, seed, device, git commit if available, dataset, model, method, Python, platform, and PyTorch version.
- `config.yaml`: raw Hydra config.
- `resolved_config.yaml`: resolved Hydra config.
- `federated_history.csv`: long-format per-round Flower metrics when federated training is run.
- `fmrl_ava_monitoring.jsonl`: per-round FMRL-AVA records when `+method=fmrl_ava` is used.

Federated simulations use per-client logger names such as `Client.1` and `Client.2`, so console output stays tagged even when clients execute in-process. Progress bars are disabled when stdout is not a tty to avoid overwriting log lines.

FMRL-AVA monitoring records include selected clients, selected fraction, per-client utilities, `base_aggregation_weight`, `alignment_cosine`, `alignment_multiplier`, final `aggregation_weight`, `support_reward`, `validation_team_reward`, raw validation reward before EMA when available, and the final mixer target used to train the server-side critics and mixer.

No online service is required.
