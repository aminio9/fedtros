# FedOSQ-Chain

**FedOSQ-Chain: Federated Multi-Agent Reinforcement Deep Q-Learning for Open-Set Recognition in Blockchain Network Traffic**  
<!-- Optional subtitle: *A Privacy-Preserving Intrusion Detection Framework Using Distributed Q-Learning and Conditional Variational Autoencoding* -->

<!-- ## Overview
FedOSQ-Chain is a federated CVAE-DQN framework for intrusion detection on blockchain network traffic. Clients train locally, share weights via Flower (FedAvg/FedProx), and perform open-set recognition by reconstructing traffic and rejecting high reconstruction errors using EVT. No raw traffic leaves the node. -->

<!-- **Key pieces**
- CVAE-DQN prior/encoder + dueling Q-head for action selection.
- Federated training of prior, recognition, main Q, and generator weights.
- Generator trains every round on correctly classified samples.
- EVT on reconstruction errors for Unknown attack detection. -->

## Requirements
- Python 3.12+
- Poetry for dependency management

## Install
```bash
pip install poetry
poetry install
```

<!-- ## Data prep
All tensors are produced by the refactored multi-source preprocessor. It reads raw CSVs,
splits them per-client, and emits train/closed-set/open-set tensors plus the class map.

1. **Point the config at your raw data**
   - Edit the `preprocess:` block in `conf/config_fl.yaml`.
   - `sources` must map each logical client (`client_1`, `client_2`, …) to a CSV path.
   - Set `label_column` and `known_labels` to match your CSVs; numerical vs. categorical
     features are inferred automatically from all other columns.

2. **Run the multi-source script**
   ```bash
   poetry run python preprocess_multisource.py
   ```
   Hydra loads `conf/config_fl.yaml`, so any overrides can be passed CLI-style
   (e.g., `poetry run python preprocess_multisource.py preprocess.sources.client_1=/path/foo.csv`).

   The script writes to `preprocess.output_dir` (default `data/processed/`):
   - `client_<cid>_train.pt`
   - `client_<cid>_test_closed.pt`
   - `client_<cid>_test_open.pt`
   - `class_names.json`

   It prints the detected `state_dim` and `num_actions`; copy those values into the
   `env_metadata` block before training.

3. **Wire the tensors into the federated config**
   - `paths.data_client_<cid>` must point to each client’s `*_train.pt`.
   - `paths.test_closed_client_<cid>` and `paths.test_open_client_<cid>` must reference the
     closed/open tensors produced for that client.
   - `paths.class_names` should point to the generated `class_names.json`.

Each `.pt` file is a PyTorch dict with `{"features": Tensor, "labels": Tensor}`. -->

<!-- ## Configure
Edit `conf/config_fl.yaml`:
```yaml
env_metadata:
  state_dim: 31    # feature dim
  num_actions: 4   # number of classes

server:
  num_rounds: 25   # federated rounds
  proximal_mu: 0.0 # FedAvg (set >0 for FedProx)
```
Adjust `training`, `generator_training`, and `evt` blocks as needed. Paths must point to your processed data and artifact directories. -->

## Preprocess
```bash
poetry run python preprocess.py

poetry run python visualize_dirichlet_split.py
```

## Run (1 server + 3 clients)
Use four terminals:
```bash
# Terminal 1
cd FedOSQ-Chain && poetry run python run_server.py

# Terminal 2
cd FedOSQ-Chain && poetry run python run_client.py --cid 1 --data_path data/processed/client_1_train.pt

# Terminal 3
cd FedOSQ-Chain && poetry run python run_client.py --cid 2 --data_path data/processed/client_2_train.pt

# Terminal 4
cd FedOSQ-Chain && poetry run python run_client.py --cid 3 --data_path data/processed/client_3_train.pt

# Run Multiple Node like clients 1 to 50
python run_client.py --cid_range "1-5" --data_path "./data/processed/client_{cid}_train.pt"
```
Logs go to `logs/`. Figures and reports go to `figures/` and `figures/clients/client_<cid>/`.

## Shared test data
- Run `poetry run python preprocess.py` to emit per-client train splits plus shared closed/open test sets (`data/processed/shared_closed_set_test.pt`, `data/processed/shared_open_set_test.pt`) and the class map.
- The defaults in `conf/config_fl.yaml` already point `paths.closed_set_test_data` and `paths.open_set_test_data` to those shared tensors so every client evaluates on the exact same datasets both before and after aggregation.
- Detailed client reports/plots are written under `figures/clients/client_<cid>/` for both stages; server logs print per-client and aggregated scores for closed-set and open-set tests using the shared files.

<!-- ## GPU usage
- Set `device.prefer: "cuda"` in `conf/config_fl.yaml` (or export `FEDOSQ_DEVICE=cuda`) to force GPU selection; set `allow_cpu_fallback: false` to fail fast if CUDA is missing.
- Enable `device.move_data_to_device: true` if you want dataset tensors kept on GPU alongside models (uses more memory but avoids repeated host/device copies).
- On AMD/Intel GPUs (e.g., Ryzen 3500U + Radeon Vega 8) install `torch-directml` and set `device.prefer: "directml"` (or `FEDOSQ_DEVICE=directml`). PyTorch will use DirectML, and logs will print the selected adapter name/version.
- Example for a single DirectML client on Windows PowerShell: `pip install torch-directml` then `$env:FEDOSQ_DEVICE="directml"; poetry run python run_client.py --cid 1 --data_path data/processed/client_1_train.pt`.
- Flower follows the device visible to each client process. Pin a GPU per client (e.g., PowerShell: `$env:CUDA_VISIBLE_DEVICES=\"0\"; $env:FEDOSQ_DEVICE=\"cuda\"; poetry run python run_client.py ...`) to prevent contention when running multiple clients on one host.
- The server and evaluation pipelines reuse the same resolved device, so closed-set/open-set checks run on GPU when configured.

## Evaluation artifacts
- Client closed-set: `client_<cid>_report_round_XXX.txt`, `client_<cid>_cm_round_XXX.png` under `figures/clients/client_<cid>/`.
- Server closed-set (global model): reports and confusion matrices under `figures/`.
- Open-set: EVT artifacts in `evt/client_<cid>/`; reports/plots in `figures/clients/client_<cid>/openset/`.

## Open-set detection logic (EVT on reconstruction error)
1. Predict class with main Q-network.
2. Reconstruct using predicted label; compute reconstruction error.
3. EVT models (per known class) score the error; if above threshold, relabel as Unknown. -->

<!-- ## Repository naming
- Repository: **FedOSQ-Chain** -->
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



<!-- note about fmrl_la
Round 2 is the "Upload Phase".

Here is exactly what is happening in your logs:
1. Round 1 (Phase A) = Training Happened Here

    Log: ROUND 1 [PHASE A]: Training & Auditing

    Action: All 3 clients trained locally. They sent their metadata (rewards) to the server.

    Result: The server selected Client 2 (3457...) and rejected the others.

2. Round 2 (Phase B) = Upload Only (No Training)

    Log: ROUND 2 [PHASE B]: Uploading & Aggregation

    Action: The server asked only the selected Client 2 to upload the weights it already trained in Round 1.

    Why no training? The client does not need to train again. It just uploads the files it cached during Round 1.

    Log: Requesting heavy weights from 1 clients → This confirms the client is uploading.

    Log: Using Standard FedAvg (w=1.0) → This confirms the server received the weights and updated the global model.

3. Round 3 (Phase A) = Training Happens Again

    Log: ROUND 3 [PHASE A]: Training & Auditing

    Action: The server sends the new global model (created in Round 2) to all clients.

    Next Step: You will see "Local training finished" logs appear for Round 3.

Summary

    Odd Rounds (1, 3, 5...): Training & Audit.

    Even Rounds (2, 4, 6...): Upload & Aggregation.

 -->