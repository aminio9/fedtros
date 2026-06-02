# Federated Learning

Flower code lives under `src/federated/`.

Entry points:

```bash
poetry run python scripts/preprocess.py runtime=cpu federated.num_clients=10
poetry run python scripts/federated_train.py runtime=cpu federated.num_clients=10 federated.num_rounds=50
poetry run python scripts/federated_server.py runtime=cpu
poetry run python scripts/federated_client.py runtime=cpu federated.client_id=1 federated.client_data_path=data/processed/client_1_train.pt
```

Use the same `federated.num_clients` value for preprocessing and federated
training. Preprocessing writes `client_1_train.pt` through
`client_N_train.pt`, where `N=federated.num_clients`.

Strategies:

- `fedavg`: standard federated averaging.
- `fedprox`: FedAvg with proximal regularization through `federated.server.proximal_mu`.
- `fmrl_ava`: two-phase learnable aggregation using audit metadata, bounded asynchronous critic residuals, sample-aware utility weighting, and FedAWA-style update-vector alignment.

FMRL-AVA writes selection, vector-aligned aggregation, validation-team-reward, and support-reward monitoring records to `federated.strategy.monitor_path`.

FMRL-AVA is the new method name for the combined implementation. Phase A follows the FMRL-LA paper's two-phase communication, asynchronous critics, and centralized utility aggregation; Phase B follows FedAWA's client-vector/update-direction idea. The implementation-specific mapping is documented in `docs/fmrl-ava-source-mapping-fa.md`.
