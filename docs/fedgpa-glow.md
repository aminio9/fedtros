# FedGPA-GLOW

FedGPA-GLOW is a repository adaptation of **FedGPA: Federated Learning with Global Personalized Aggregation** (Han et al., AI Open 2025, DOI `10.1016/j.aiopen.2025.03.001`). The paper decouples each model into a feature extractor and a classifier, uses class prototypes for local-global representation alignment, and computes separate aggregation strategies for feature extractors and classifiers.

## Mapping to this codebase

The original FedGPA paper is personalized and returns one personalized model per client. This repository's evaluation pipeline expects a single global checkpoint, so the first implementation is a global-compatible FedGPA variant:

- **Feature extractor:** `prior_net` and `recognition_net`.
- **Classifier:** `value_net_main` Q/classification head.
- **Prototype feature:** `prior_net(states)` mean vector `mu_p`, because inference already feeds `mu_p` into `value_net_main`.
- **Local-global alignment:** clients receive server prototypes and add `loss_weights.prototype * MSE(mu_p, global_prototype[y])` during the prior update.
- **Server aggregation:** feature modules and classifier modules use different weights.

## Server aggregation

Each client returns local class prototypes, class counts, and intra-class variances. The server builds global prototypes as count-weighted means and computes:

- feature weights from prototype similarity plus sample count, controlled by `fedgpa.mu`;
- classifier weights from prototype distance plus intra-class variance, lightly blended with sample count.

The server writes `fedgpa_glow_monitoring.jsonl` with per-client feature/classifier weights.

## Main config

Run with:

```bash
python run.py experiment=exp3 +method=fedgpa_glow seed=42 dataset.preprocessing.alpha=0.1
```

For IID with three clients:

```bash
python run.py experiment=exp1 +method=fedgpa_glow seed=42 \
  federated.num_clients=3 \
  dataset.preprocessing.num_clients=3 \
  dataset.preprocessing.iid=true \
  dataset.preprocessing.alpha=1.0 \
  tracking.run_id=e1_FedGPA_GLOW_iid_c3_seed42
```

For non-IID alpha=0.1:

```bash
python run.py experiment=exp3 +method=fedgpa_glow seed=42 \
  federated.num_clients=10 \
  dataset.preprocessing.num_clients=10 \
  dataset.preprocessing.iid=false \
  dataset.preprocessing.alpha=0.1 \
  tracking.run_id=e3_FedGPA_GLOW_alpha0.1_c10_seed42
```

## Recommended ablations

1. `+method=fedavg`
2. `+method=fedgpa_glow training.loss_weights.prototype=0.0`
3. `+method=fedgpa_glow federated.strategy.fedgpa.mu=0.0`
4. `+method=fedgpa_glow federated.strategy.fedgpa.mu=0.5`
5. `+method=fedgpa_glow federated.strategy.fedgpa.mu=0.8`

Primary metrics: macro-F1, balanced accuracy, worst-class F1, and per-class recall. Accuracy alone is not enough for this imbalanced intrusion dataset, because Normal can hide minority-class collapse like a very well-dressed liar.
