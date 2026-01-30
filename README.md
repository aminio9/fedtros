
# FMARL

Federated Multi-Agent Reinforcement Deep Q-Learning for Open-Set Recognition in Blockchain Network Traffic

## Requirements
- Python 3.12+
- Poetry

## Install
```bash
pip install poetry
poetry install
````

## Preprocess

Generate processed datasets and (optional) visualization.

```bash
poetry run python preprocess.py
poetry run python visualize_dirichlet_split.py
```

Outputs are written under `data/processed/`.

## Run (1 server + 3 clients)

Use four terminals.

```bash
# Terminal 1 (server)
poetry run python run_server.py
```

```bash
# Terminal 2 (client 1)
poetry run python run_client.py --cid 1 --data_path data/processed/client_1_train.pt
```

```bash
# Terminal 3 (client 2)
poetry run python run_client.py --cid 2 --data_path data/processed/client_2_train.pt
```

```bash
# Terminal 4 (client 3)
poetry run python run_client.py --cid 3 --data_path data/processed/client_3_train.pt
```

### Run multiple clients with a range

```bash
poetry run python run_client.py --cid_range "1-10" --data_path "./data/processed/client_{cid}_train.pt"
```

## Outputs

* Logs: `logs/`
* Global figures/reports: `figures/`
* Per-client figures/reports: `figures/clients/client_<cid>/`

## Shared test data (what preprocess creates)

`preprocess.py` also creates shared evaluation sets so all clients evaluate on identical datasets:

* `data/processed/shared_closed_set_test.pt`
* `data/processed/shared_open_set_test.pt`

Client/server evaluation artifacts are saved under `figures/` and `figures/clients/client_<cid>/`.

