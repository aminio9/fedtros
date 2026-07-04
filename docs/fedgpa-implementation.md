# FedGPA implementation

This repository now includes `+method=fedgpa`, a FedGPA-inspired strategy adapted to the CVAE-DQN model stack.

## Mapping from FedGPA to this codebase

FedGPA decouples the model into a feature extractor and classifier, uses local/global class prototypes during local training, and computes personalized aggregation weights on the server. In this project, the mapping is:

- `prior_net`: feature extractor, slowly aggregated with prototype-similarity weights.
- `recognition_net`: personalized latent module, frozen/very slowly aggregated.
- `value_net_main`: Q classifier/head, strongly aggregated with personalized classifier weights.
- `generation_net`: frozen by default to avoid unstable open-set generator averaging.

Clients send class prototypes built from normalized latent vectors and Q logits. The server computes global prototypes, client prototype distances, feature-extractor weights (`alpha`), and classifier/Q-head weights (`beta`). Each client receives its own personalized model in the next round.

## Run examples

```bash
poetry run python run.py experiment=exp1 +method=fedgpa seed=42
poetry run python run.py experiment=exp3 +method=fedgpa seed=42 dataset.preprocessing.alpha=0.1
poetry run python run.py experiment=exp4 +method=fedgpa seed=42 dataset.preprocessing.alpha=0.1
```

For IID sanity checks, reduce or disable personalization if needed:

```bash
poetry run python run.py experiment=exp1 +method=fedgpa \
  federated.strategy.prototype_lambda=0.02 \
  federated.strategy.classifier_self_weight=0.0 \
  federated.strategy.recognition_mix=0.0
```
