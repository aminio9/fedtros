# Losses

Reusable loss functions live in `src/training/losses.py`. The CVAE-DQN agent also computes KL, TD, classification, generator reconstruction, and FedProx penalties in `src/agents/agent.py`.

## Config Weights

The default local learner is now contextual-bandit RL, not a sequential MDP setup. The old TD loss remains for backward-compatible experiments, but its default weight is deliberately smaller than the full-action bandit Q loss.

```yaml
training:
  rl_mode: contextual_bandit
  gamma: 0.0
  epsilon_start: 0.30
  epsilon_end: 0.02
  epsilon_decay_rate: 0.97
  loss_weights:
    prior_kl: 0.5
    q_td: 0.25
    bandit_q: 1.0
    classification: 2.0
    generator_reconstruction: 1.0
    proximal: 1.0
  auxiliary_losses:
    supervised_contrastive_lambda: 0.02
    supervised_contrastive_temperature: 0.1
    center_loss_lambda: 0.01
  classification_loss:
    name: focal
    focal_gamma: 1.5
    use_class_weights: true
  imbalance:
    enabled: true
    weight_mode: effective_number
    effective_number_beta: 0.999
    min_weight: 0.3
    max_weight: 3.0
    normalize: mean
    class_balanced_sampling: true
    weighted_reward: true
    weight_negative_reward: false
  kl:
    free_nats: 0.25
    warmup_steps: 200
```

Validation rejects negative loss weights.

## Agent Losses

Prior KL:

```text
warmup(step) * free_bits(KL(q_phi(z | s, a_true) || p_theta(z | s)))
```

TD loss, retained for compatibility:

```text
SmoothL1(Q_main(z_now, s, action), r + gamma * Q_target(z_next, s_next, a_star))
```

With the default `gamma=0.0`, this naturally becomes immediate-reward fitting. Double-DQN target code remains available when an experiment intentionally overrides `gamma`.

Contextual-bandit full-action Q loss, the main RL signal:

```text
SmoothL1(Q_main(mu_p(s), s), target_all_actions)

target_all_actions[y_true] = reward.correct
target_all_actions[present wrong classes] = reward.incorrect
target_all_actions[absent local classes] = masked from the loss
```

Classification loss:

```text
FocalCrossEntropy(Q_main(mu_p(s), s), a_true)
```

Generator reconstruction:

```text
SmoothL1(G(z, a_true), s)
```

FedProx:

```text
0.5 * mu * ||w_local - w_reference||_2^2
```

Cross-entropy uses logits, not softmax probabilities. Auxiliary losses do not silently detach their input embeddings.

## Reusable Loss Utilities

`src/training/losses.py` implements:

- focal cross-entropy loss
- diagonal Gaussian KL with free-bits
- KL warmup scheduling
- SmoothL1 reconstruction loss
- supervised contrastive loss
- batch center compactness loss
- energy score

## Logging Names

Agent train-step metrics include:

- `loss/total`
- `loss/prior_kl`
- `loss/prior_kl_raw`
- `loss/prior_kl_warmup`
- `loss/q_td`
- `loss/bandit_q`
- `loss/bandit_q_weighted`
- `loss/classification`
- `loss/supervised_contrastive`
- `loss/center_compactness`
- `loss/proximal`
- `gradient/prior_norm`
- `gradient/q_norm`
- `q/value_mean`, `q/value_std`, `q/value_min`, `q/value_max`
- `lr/prior`, `lr/q_rl`
