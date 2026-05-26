# cf_marlos FMRL-LA Adaptation

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
utility_i = AsyncCritic_i(hidden_i, scalar_features_i)
Q_total = CentralizedAggregator([utility_i], global_client_state)
theta_next = theta_global + aggregation_lr * sum_i normalized(utility_i) * delta_i
```

The mixer target is a configurable composite utility:

```text
reward + closed-set F1 + accuracy + TD stability + novelty + communication efficiency
```

This is a pragmatic target because the intrusion dataset does not provide a shared live Dec-POMDP team reward like the driving benchmark in the paper.
