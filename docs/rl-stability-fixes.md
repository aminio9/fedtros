# RL stability fixes for CVAE-DQN federated intrusion detection

The local training loop is not a long-horizon control problem. It is a sampled data-pool RL formulation: each state is a traffic feature vector and each action is a class prediction. That makes the task closer to a contextual bandit. DQN is still used, but it must be stabilized so the majority class does not dominate the reward signal.

## Changed files

- `src/rl/environment.py`
  - Adds configurable reward shaping.
  - Supports `reward_mode: symmetric` and `reward_mode: class_balanced`.
  - Computes local inverse-frequency class weights from each client's data.
  - Adds `reward_weight` and `reward_mode` to step `info`.

- `src/agents/agent.py`
  - Keeps the original TD loss.
  - Adds optional auxiliary cross-entropy on the Q-value vector.
  - Records `last_aux_ce_loss` for logging.

- `src/rl/local_training.py`
  - Passes auxiliary CE settings into `Agent.train_step`.
  - Logs `avg_aux_ce_loss`.
  - Tracks `balanced_policy_accuracy` and `per_class_policy_accuracy`.

- `src/configs/training/default.yaml`
  - Keeps old behavior by default: `reward_mode: symmetric`, `aux_ce_weight: 0.0`.

- `src/configs/method/fedgpa.yaml`
  - Enables class-balanced reward, small auxiliary CE, and slower epsilon decay for FedGPA.

## Loss objective

The Q-update objective is now:

```text
total_q_loss = TD_loss
             + aux_ce_weight * CrossEntropy(Q(s), true_label)
             + prototype_lambda * PrototypeLoss
             + optional FedProx penalty
```

The prior update still uses KL alignment between `recognition_net(s, true_action)` and `prior_net(s)`, plus optional FedGPA prototype loss and optional FedProx penalty.

## Why this helps non-IID

In non-IID clients, one client may see mostly Normal traffic while another sees more of a minority attack class. Plain reward encourages each client to become locally greedy. After aggregation, those local biases can damage minority-class behavior. The class-balanced reward makes rare local labels matter, while the auxiliary CE term anchors the Q-head to the known label target.

## Recommended comparison protocol

For a fair algorithm comparison, separate two questions:

1. Does the RL stabilizer help all methods?
2. Does FedGPA improve aggregation on top of the same RL stabilizer?

Run FedGPA with its default method config, then run FedAvg with the same RL overrides:

```bash
poetry run python run.py experiment=exp3 +method=fedavg seed=42 \
  federated.num_clients=3 federated.num_rounds=10 training.local_episodes_per_round=10 \
  dataset.preprocessing.alpha=0.1 \
  training.reward_mode=class_balanced training.aux_ce_weight=0.10 \
  training.aux_ce_label_smoothing=0.02 training.epsilon_decay_rate=0.98 training.epsilon_end=0.05
```

Then compare post-aggregation macro-F1, balanced accuracy, and per-class F1 for `BP`, `MitM`, and `FoT`.
