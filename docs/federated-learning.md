# Federated Learning

Flower code lives under `src/federated/`.

Entry points:

```bash
poetry run python scripts/preprocess.py federated.num_clients=10
poetry run python scripts/federated_train.py federated.num_clients=10 federated.num_rounds=50
poetry run python scripts/federated_server.py
poetry run python scripts/federated_client.py federated.client_id=1 federated.client_data_path=data/processed/client_1_train.pt
```

Use the same `federated.num_clients` value for preprocessing and federated
training. Preprocessing writes `client_1_train.pt` through
`client_N_train.pt`, where `N=federated.num_clients`.

Strategies:

- `fedavg`: standard federated averaging.
- `fedprox`: FedAvg with proximal regularization through `federated.server.proximal_mu`.
- `fmrl_la`: two-phase learnable aggregation using asynchronous critics and a centralized mixer.

FMRL_LA writes monitoring records to `federated.strategy.monitor_path`.
