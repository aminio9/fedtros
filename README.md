## Setup
1. **Install Dependencies:**
    ```bash
    pip install poetry
    poetry install
    ```

2. **Preprocess and Partition Data:**
    This project assumes you have already run your preprocessing scripts and have partitioned your training data for each client. Place them in `data/processed/`.
    - `data/processed/client_1_train.pt`
    - `data/processed/client_2_train.pt`
    - `data/processed/client_3_train.pt`

    *Note: These files must be PyTorch-loadable and return a dictionary `{'features': torch.Tensor, 'labels': torch.Tensor}`.*

3. **Configure Environment:**
    Open `conf/config_fl.yaml` and **you must** set the `env_metadata` to match your data:
    ```yaml
    env_metadata:
      state_dim: 30  # <-- CHANGE TO YOUR FEATURE DIM
      num_actions: 8 # <-- CHANGE TO YOUR NUMBER OF CLASSES
    ```

## How to Run (1 Server + 3 Clients)

You will need **4 separate terminals**.

1. **Terminal 1: Start the Server**
    ```bash
    poetry run python run_server.py
    ```
    *The server will start and wait for 3 clients to connect.*

2. **Terminal 2: Start Client 1**
    ```bash
    poetry run python run_client.py --cid 1 --data_path data/processed/client_1_train.pt
    ```

3. **Terminal 3: Start Client 2**
    ```bash
    poetry run python run_client.py --cid 2 --data_path data/processed/client_2_train.pt
    ```

4. **Terminal 4: Start Client 3**
    ```bash
    poetry run python run_client.py --cid 3 --data_path data/processed/client_3_train.pt
    ```

Once all 3 clients connect, the server will begin the first round of training. Logs for the server and each client will be saved to the `logs/` directory.

## Evaluation Artifacts

- After every federated round, each client evaluates its personalized agent on `paths.closed_set_test_data`. The resulting classification report (`client_<cid>_report_round_XXX.txt`) and confusion matrix plot (`client_<cid>_cm_round_XXX.png`) are stored under `figures/clients/client_<cid>/`.
- The server still runs the global closed-set evaluation; its confusion matrices and reports remain in `figures/`.
- Make sure `conf/config_fl.yaml` points `paths.closed_set_test_data`, `paths.class_names`, and `paths.figures_dir` at locations that exist on disk so the artifacts can be written.

## Generator Training (Unknown Attack Reconstruction)

- Each client now trains the decoder/generation network **once**, right after completing its final federated round. Training uses only the locally **correctly classified** samples, ensuring the generator learns from the final global model snapshot.
- The training loop mirrors the standalone script (`train_generator.py` in the original OpenSetQ-Chain project) and now supports multiple generator rounds with per-round epochs. Each round/epoch logs its reconstruction MSE so you can inspect convergence for every agent.
- Configure the behavior via the `generator_training` block in `conf/config_fl.yaml` (toggle `enabled`, tweak `batch_size`, learning rate, the minimum number of correct samples, plus `rounds` and `epochs_per_round` to control how many generator passes run).
- Generator weights are still federated with the prior and Q-network parameters, but they are only updated in the last round before training concludes, keeping the communication cost the same as the RL phase.
