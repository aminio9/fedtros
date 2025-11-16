# FedOSQ-Chain

**FedOSQ-Chain: Federated Multi-Agent Reinforcement Deep Q-Learning for Open-Set Recognition in Blockchain Network Traffic**  
Optional subtitle: *A Privacy-Preserving Intrusion Detection Framework Using Distributed Q-Learning and Conditional Variational Autoencoding*

## Overview
FedOSQ-Chain is a federated CVAE-DQN framework for intrusion detection on blockchain network traffic. Clients train locally, share weights via Flower (FedAvg/FedProx), and perform open-set recognition by reconstructing traffic and rejecting high reconstruction errors using EVT. No raw traffic leaves the node.

**Key pieces**
- CVAE-DQN prior/encoder + dueling Q-head for action selection.
- Federated training of prior, recognition, main Q, and generator weights.
- Generator trains every round on correctly classified samples.
- EVT on reconstruction errors for Unknown attack detection.

## Requirements
- Python 3.12+
- Poetry for dependency management

## Install
```bash
pip install poetry
poetry install
```

## Data prep
Place processed tensors in `data/processed/`:
- `client_1_train.pt`, `client_2_train.pt`, `client_3_train.pt`
- `closed_set_test.pt` (held-out known traffic)
- `open_set_test.pt` (known + unknown for EVT/open-set eval)

Each file should be a PyTorch dict: `{"features": Tensor, "labels": Tensor}`.

## Configure
Edit `conf/config_fl.yaml`:
```yaml
env_metadata:
  state_dim: 31    # feature dim
  num_actions: 4   # number of classes

server:
  num_rounds: 25   # federated rounds
  proximal_mu: 0.0 # FedAvg (set >0 for FedProx)
```
Adjust `training`, `generator_training`, and `evt` blocks as needed. Paths must point to your processed data and artifact directories.

## Run (1 server + 3 clients)
Use four terminals:
```bash
# Terminal 1
poetry run python run_server.py

# Terminal 2
poetry run python run_client.py --cid 1 --data_path data/processed/client_1_train.pt

# Terminal 3
poetry run python run_client.py --cid 2 --data_path data/processed/client_2_train.pt

# Terminal 4
poetry run python run_client.py --cid 3 --data_path data/processed/client_3_train.pt
```
Logs go to `logs/`. Figures and reports go to `figures/` and `figures/clients/client_<cid>/`.

## Evaluation artifacts
- Client closed-set: `client_<cid>_report_round_XXX.txt`, `client_<cid>_cm_round_XXX.png` under `figures/clients/client_<cid>/`.
- Server closed-set (global model): reports and confusion matrices under `figures/`.
- Open-set: EVT artifacts in `evt/client_<cid>/`; reports/plots in `figures/clients/client_<cid>/openset/`.

## Open-set detection logic (EVT on reconstruction error)
1. Predict class with main Q-network.
2. Reconstruct using predicted label; compute reconstruction error.
3. EVT models (per known class) score the error; if above threshold, relabel as Unknown.

## Repository naming
- Repository: **FedOSQ-Chain**
<!-- - Python package: `fedosq_chain` (code currently lives under `src/`; keep the existing import paths).

## Citation (template)
If you use FedOSQ-Chain, please cite:
```
@article{fedosqchain2025,
  title={FedOSQ-Chain: Federated Multi-Agent Reinforcement Deep Q-Learning for Open-Set Recognition in Blockchain Network Traffic},
  author={...},
  year={2025}
}
``` -->
