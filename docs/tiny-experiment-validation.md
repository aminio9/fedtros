# Tiny Experiment Validation

This note is legacy. Use `docs/runbooks/validation-tiny.md` and
`scripts/experiments/run_validation_tiny.sh` for the current tiny end-to-end
validation path.

Current tiny validation settings:

- 2 clients
- 1 logical round
- 1 local episode per round
- 3 steps per episode
- `runtime=tiny`
- run-local `processed/` output

The expected artifact set includes:

- `evaluation_metrics.json`
- `open_set_metrics.json`
- `plots/plot_manifest.json`
- `best_model.pt`
- `latest_checkpoint.pt`
- `metrics.jsonl`
- `metrics.csv`
- `metadata.json`

Keep `tracking.run_id` and `dataset.preprocessing.output_dir` aligned when you
adapt the example for another local run.
