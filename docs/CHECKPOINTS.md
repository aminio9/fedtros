# Checkpoints

Checkpoint helpers live in `src/checkpointing/checkpoints.py`.

Each checkpoint contains:

- model state for prior, recognition, main Q, target Q, and generation network.
- optimizer state for prior and Q/RL optimizers.
- epoch/round and global step.
- config snapshot.
- metrics and best metric.
- RNG state when `checkpointing.include_rng_state=true`.

Default paths:

- `best_model.pt`
- `latest_checkpoint.pt`
- `final_model.pt`

Evaluation requires an existing checkpoint:

```bash
poetry run python scripts/evaluate.py checkpoint.path=outputs/run_id/best_model.pt
```
