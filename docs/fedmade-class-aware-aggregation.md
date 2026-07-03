# FedMADE-Style Class-Aware Aggregation

The implemented method is a FedMADE-inspired federated aggregation baseline, not a verbatim reproduction of the FedMADE paper.

## Code

- `src/federated/class_aware.py`: pure weighting utilities.
- `src/federated/server.py`: `FedMADEClassAwareAggregationStrategy`.
- `src/federated/client.py`: local label-profile and quality metadata.
- `src/configs/method/fedmade.yaml`: Hydra overlay.
- `tests/test_fedmade_strategy.py`: weighting and config tests.

Command:

```bash
python run.py experiment=exp3 +method=fedmade seed=42 dataset.preprocessing.alpha=0.1
```

## Aggregation Rule

The server starts from FedAvg's sample-count prior and multiplies it by bounded class, quality, and cluster terms:

```text
a_i = n_i * class_multiplier_i * quality_multiplier_i * cluster_multiplier_i
theta_next = sum_i a_i theta_i / sum_i a_i
```

Inputs are client metadata already produced by the pipeline: label histograms, local quality metrics, class entropy, label coverage, and optional per-class recall.

Default overlay parameters:

```yaml
rare_class_strength: 1.25
quality_weight_blend: 0.35
cluster_balance_strength: 0.50
min_weight_multiplier: 0.25
max_weight_multiplier: 3.00
label_smoothing: 1.0
```

## Reporting Status

Report it as `FedMADE-style` or `FedMADE-inspired`. Do not claim exact reproduction unless the original auxiliary-validation design is implemented.
