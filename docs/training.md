# Training

## Execution Flow

`run.py` dispatches by `experiment.pipeline`:

- `full`: preprocess, federated simulation, evaluation.
- `centralized`: preprocess, centralized local-replay training, evaluation.
- `train` or `federated`: training only, assuming preprocessed tensors exist.
- `evaluate`: evaluation from an existing checkpoint.
- `plot`, `compare`, `suite_artifacts`: artifact-only utilities.

Use `experiment=validation runtime=tiny` for a CPU preflight. It caps known/unknown preprocessing rows and runs one tiny federated round.
All training/evaluation pipelines currently instantiate the CVAE-DQN `Agent`.

## Local Round

`src/rl/local_training.py` runs the local loop:

1. `BlockchainIntrusionEnv` samples known-training rows.
2. `EpsilonGreedyPolicy` chooses a class action from prior-latent Q logits. When local class counts are known, random exploration and greedy argmax are restricted to classes present on that client.
3. Reward is `training.reward.correct` or `training.reward.incorrect`, optionally class-weighted.
4. The replay buffer stores `(s, action, reward, next_s, done, true_label)`. The next state is another sampled row; the action does not cause it.
5. `Agent.train_step` updates prior parameters, then recognition/main-Q parameters. The default `training.rl_mode=contextual_bandit` uses `gamma=0.0`, so the retained TD target collapses to the immediate reward while the full-action bandit Q loss supervises all class logits.
6. The target Q network is soft-updated every `training.target_update_freq` steps for backward-compatible DQN experiments.
7. Optional generator training runs after a local round on correctly classified known samples. Closed-set exp3 keeps generator training disabled by default.

## Loss Logging

Each train step returns an explicit scalar dictionary. Local-round metrics average these keys and expose stable names such as:

- `avg_total_loss`
- `avg_td_loss`
- `avg_bandit_q_loss` or `loss/bandit_q` in raw train-step metrics
- `avg_kl_loss`
- `avg_classification_loss`
- `avg_prox_loss`
- `gradient_norm_prior`
- `gradient_norm_q`
- `learning_rate_prior`
- `learning_rate_q_rl`

Generator training logs reconstruction, weighted, proximal, gradient, LR, sample-count, and correct-fraction metrics when enabled.

## Checkpointing

`src/checkpointing/checkpoints.py` writes `latest`, `last`, `best`, and `final` checkpoints according to `src/configs/checkpointing/default.yaml`.

Best checkpoint selection accepts only validation-prefixed metrics or `combined_validation_score`. Test metrics are intentionally ignored for checkpoint selection.

Checkpoint sidecars include config hash, seed, known/unknown label metadata, selected metric name/value, epoch/round, and timestamp.

## Cheap Validation

Run before full experiments:

```bash
python scripts/cheap_validation.py
python -m pytest
python run.py experiment=validation runtime=tiny seed=42 tracking.run_id=tiny_validation_seed42
```

Do not run 100-round, large-client, or GPU experiments until these pass.

## Contextual-Bandit Correction

`BlockchainIntrusionEnv.step()` gives an immediate reward based on whether the selected action equals the true label. The next state is another sampled traffic row, not a consequence of the current action. Therefore the stable default is contextual-bandit training: `training.rl_mode=contextual_bandit`, `gamma=0.0`, reduced epsilon exploration, focal cross-entropy, effective-number class weights, and a full-action Q loss.

This keeps the CVAE-DQN architecture intact while removing noisy bootstrapping from unrelated future rows. It also keeps the old TD machinery in place for ablations that deliberately set `gamma > 0`.
