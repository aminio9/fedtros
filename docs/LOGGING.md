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

No online service is required.
