# cf_marlos FMRL-AVA Adaptation

The paper's method has four server-side requirements:

1. clients upload hidden state and reward metadata before uploading model updates
2. one asynchronous critic per client estimates local utility
3. a centralized aggregator/mixer predicts system utility conditioned on global state
4. global model updates are weighted by learned client utilities

This project adapts those requirements to a federated open-set intrusion dataset:

- hidden state: mean prior latent vector over replay audit samples
- recent reward: average local episode reward
- history reward: lifetime local reward accumulated by the client process
- additional dataset signals: class entropy, label coverage, local sample count
- additional learning signals: audit macro F1, local eval F1/accuracy, TD stability, KL novelty, generator correct fraction
- local gradient proxy: model parameter delta `theta_client - theta_global`

The server computes:

```text
audit_score_i = quality(hidden_i, scalar_features_i)
critic_score_i = AsyncCritic_i(hidden_i, scalar_features_i)
z_i = (1 - beta) * audit_score_i + beta * critic_score_i
u_i = clip(1 + 2 * gamma * (z_i - mean(z)), min_u, max_u)
b_i = n_i * u_i
delta_ref = sum_i b_i * delta_i / sum_i b_i
m_i = clip(exp(kappa * cosine(delta_i, delta_ref)), min_m, max_m)
a_i = b_i * m_i
theta_next = theta_global + aggregation_lr * sum_i a_i * delta_i / sum_i a_i
```

If a client is absolutely low-quality relative to the current round, the server
clips its utility to zero so it skips Phase B upload. In IID-like rounds the
utilities stay near 1 and update directions are similar, so the normalized
update behaves like FedAvg with sample-count weights. In non-IID rounds, the
vector-alignment multiplier down-weights selected clients whose parameter delta
conflicts with the round's reference update direction.

The server-side team reward is separate from client selection and from direct
aggregation weighting. It is computed from validation/global metrics and
smoothed with an EMA:

```text
R_val = w1 * F1_macro_val
      + w2 * BalancedAccuracy_val
      + w3 * AUROC_open_set
      + w4 * F1_unknown
      + w5 * Rejection_quality

R_sup = small fallback from local_eval_F1, TD_stability,
        coverage_quality, generator_quality, and communication

U_sys = lambda_val * R_val + (1 - lambda_val) * R_sup
```

This is more useful than train-accuracy-only monitoring because it can see
closed-set quality and open-set rejection at the same time. The dataset still
does not provide a true live Dec-POMDP team reward, so this validation-aware
proxy is used as a critic/mixer training target, not as the direct aggregation
weight. In the current config,
`lambda_val` is `strategy.validation_reward_blend`, and `R_val` is smoothed by
`strategy.validation_reward_ema_decay`.
Metrics that are not produced by the current evaluation mode are omitted from
the validation-reward denominator rather than treated as zero. This keeps
closed-set/IID runs from being penalized for missing open-set metrics, while
open-set runs still use AUROC, unknown F1, and rejection quality when those
metrics exist.
