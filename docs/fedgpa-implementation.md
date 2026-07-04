# FedGPA implementation

This repository includes `+method=fedgpa`, a FedGPA-inspired strategy adapted to the CVAE-DQN intrusion-detection model stack.

## Why this adaptation exists

FedGPA splits a model into a representation component and a classifier component, then combines two ideas:

1. local-global class prototype alignment during client training;
2. personalized aggregation weights on the server.

The original paper is written for CNN-style classification. This project uses a CVAE-DQN/RL classifier, so the feature/classifier split is mapped onto the model modules instead of copied as a CNN.

## Mapping from FedGPA to this codebase

- `prior_net`: representation/prior module. It is aggregated slowly with prototype-similarity weights.
- `recognition_net`: client-sensitive latent module. It is aggregated very slowly.
- `value_net_main`: Q-value classifier/action head. It receives the strongest personalized aggregation.
- `generation_net`: generator/open-set module. It is not aggregated by default because generator averaging was unstable in earlier runs.

Clients send class prototypes built from normalized latent vectors and normalized Q-value vectors. The server computes global prototypes, client prototype distances, feature weights (`alpha`), and classifier/Q-head weights (`beta`). Each client receives its own personalized model in the next round.

## RL stability fixes added with FedGPA

The task behaves like a sampled contextual bandit: the state is a tabular traffic sample, the action is the predicted class, and the reward is tied to whether the action matches the label. Because the dataset is imbalanced, plain `+1/-1` reward can make the agent over-optimize the majority `Normal` class.

FedGPA now enables three RL stabilizers by default:

1. **Class-balanced reward** in `src/rl/environment.py`.
   - Each client computes local class weights from its own label distribution.
   - Correct and wrong rewards are multiplied by the class weight.
   - Minority classes therefore create a stronger TD signal.

2. **Auxiliary supervised CE loss** in `src/agents/agent.py`.
   - The DQN TD loss remains the main objective.
   - A small cross-entropy term is added on `value_net_main` Q-values using the true label.
   - This stabilizes the Q/classifier head under non-IID class imbalance.

3. **Slower exploration decay** in `src/configs/method/fedgpa.yaml`.
   - FedGPA overrides epsilon decay to avoid greedy local class bias too early.
   - The default floor is `epsilon_end: 0.05`, not `0.01`.

## Important config values

FedGPA method config:

```yaml
federated:
  strategy:
    prototype_lambda: 0.05
    prototype_mu: 0.5
    prototype_feature: latent_q
    value_mix: 1.0
    prior_mix: 0.25
    recognition_mix: 0.05
    generation_mix: 0.0
    classifier_self_weight: 0.25

training:
  epsilon_end: 0.05
  epsilon_decay_rate: 0.98
  reward_mode: class_balanced
  reward_weight_power: 0.5
  reward_min_weight: 0.5
  reward_max_weight: 3.0
  reward_normalize_mean: true
  aux_ce_weight: 0.10
  aux_ce_label_smoothing: 0.02
  aux_ce_use_class_weights: true
```

Global defaults in `src/configs/training/default.yaml` keep the old behavior for other baselines:

```yaml
reward_mode: symmetric
aux_ce_weight: 0.0
```

This keeps FedAvg/FedProx comparable unless you explicitly enable the same RL stabilizers for them.

## Run examples

IID sanity check:

```bash
poetry run python run.py experiment=exp1 +method=fedgpa seed=42 \
  federated.num_clients=3 federated.num_rounds=10 training.local_episodes_per_round=10 \
  dataset.preprocessing.iid=true
```

Non-IID alpha 0.1:

```bash
poetry run python run.py experiment=exp3 +method=fedgpa seed=42 \
  federated.num_clients=3 federated.num_rounds=10 training.local_episodes_per_round=10 \
  dataset.preprocessing.alpha=0.1
```

Fair FedAvg baseline with the same RL stabilizers:

```bash
poetry run python run.py experiment=exp3 +method=fedavg seed=42 \
  federated.num_clients=3 federated.num_rounds=10 training.local_episodes_per_round=10 \
  dataset.preprocessing.alpha=0.1 \
  training.reward_mode=class_balanced training.aux_ce_weight=0.10 \
  training.aux_ce_label_smoothing=0.02 training.epsilon_decay_rate=0.98 training.epsilon_end=0.05
```

## What to monitor

Use these log fields:

- `avg_td_loss`: DQN TD loss.
- `avg_aux_ce_loss`: supervised Q-head stabilizer.
- `avg_proto_loss`: FedGPA prototype regularization.
- `policy_accuracy`: raw sampled policy accuracy.
- `balanced_policy_accuracy`: class-balanced sampled policy accuracy.
- `per_class_policy_accuracy`: JSON map of per-class sampled policy accuracy.
- `mean_reward_weight`: average class weight seen in the local round.
- `alpha_self` / `beta_self` in `fedgpa_monitoring.jsonl`: whether FedGPA is sharing representation more than classifier parameters.

Expected behavior:

- Round 1 prototype loss can be zero because no global prototypes exist yet.
- Round 2+ prototype loss should usually be non-zero, but can be very small in IID.
- In non-IID, `beta_self` should usually be higher than `alpha_self` because the classifier/Q-head should remain more personalized.
- Judge by macro-F1 and minority-class F1, not only accuracy or reward.
