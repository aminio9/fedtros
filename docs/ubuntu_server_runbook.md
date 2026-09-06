# Ubuntu Linux Server Experiment Runbook (Poetry & Seed 42)

This runbook provides complete, copy-paste ready commands to configure an Ubuntu Linux GPU server, install **Poetry**, install all dependencies via Poetry, set up persistent `tmux` terminals, and execute all FedTROS-MC paper experiments using **seed 42**.

---

## 1. System Setup & Poetry Installation

Connect to your Ubuntu server via SSH and execute the following steps:

### 1.1 Update system and install system prerequisites
```bash
sudo apt update && sudo apt install -y \
    git tmux unzip curl build-essential \
    python3 python3-venv python3-pip
```

Verify GPU availability:
```bash
nvidia-smi
```

### 1.2 Install Python 3.11 (if not present)
FedTROS-MC requires Python **3.11 or 3.12**. If your system defaults to Python 3.10:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

### 1.3 Install Poetry
Install official Poetry via its standalone installer and add it to your PATH:
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to current PATH and persist in ~/.bashrc for new tmux sessions
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Verify installation
poetry --version
```

### 1.4 Clone repository & install dependencies with Poetry
```bash
# Clone repository
git clone https://github.com/aminio9/fedtros.git
cd fedtros

# Tell Poetry to use Python 3.11 (or your active Conda Python if using Conda: poetry env use $(which python))
poetry env use python3.11

# Install all project dependencies into the Poetry environment
poetry install
```

### 1.5 Extract datasets
Ensure `fedtros_datasets.zip` is placed in the `fedtros/` directory, then extract raw CSV files:
```bash
mkdir -p data/raw
unzip fedtros_datasets.zip -d data/raw/

# Verify that raw CSV files are present
ls -lh data/raw/*.csv
```

### 1.6 Run health check and tests with Poetry
```bash
poetry run python scripts/doctor.py --wandb-mode disabled
poetry run pytest tests/test_confid_ensemble_pipeline.py -q
poetry run pytest -q
```
*(All tests including the new ConFID dual-path ensemble pipeline should pass).*


---

## 2. Managing Persistent Terminals with `tmux`

Running long experiments inside `tmux` sessions ensures they keep executing in the background even if your SSH session disconnects.

### Quick `tmux` Reference:
- **Detach** (leave running in background): Press `Ctrl + B`, release, then press `D`.
- **Reattach** to a running session: `tmux attach -t <session_name>`
- **List** active sessions: `tmux ls`
- **Kill** a session when finished: `tmux kill-session -t <session_name>`

---

## 3. GPU / MIG Device Selection (Important for A100)

If your server has multiple GPUs or an **A100 with MIG (Multi-Instance GPU)** enabled, verify which GPU/MIG instance is free:
```bash
nvidia-smi -L
```
Export the free GPU or MIG device before running commands in each terminal session:
```bash
# Example for standard GPU index:
export CUDA_VISIBLE_DEVICES=0

# Example for A100 MIG instance (replace with UUID of the free instance from nvidia-smi -L):
export CUDA_VISIBLE_DEVICES=MIG-GPU-xxxx
```

---

## 4. Experiment Execution Across Terminals (Seed 42)

The execution protocol follows:
- **Environment**: All commands use **`poetry run`**
- **Seed**: `--seeds 42`
- **Stage**: `--stage main` (100 communication rounds, 10 clients)
- **Phase ordering**: Canonical method first (`--method fedtros_mc`), then matched baselines (`--method fedavg fedprox scaffold local_only centralized`)
- **Safety**: `--only-missing` allows safe continuation if interrupted.

### 📋 Alpha Matrix Reference (Why some studies run 1 alpha vs 3 alphas):
| Study ID | Description | Dirichlet $\alpha$ Values | Runs (`fedtros_mc`) |
| :--- | :--- | :---: | :---: |
| **E1-IID-CS** | Closed-Set IID Baseline | $\alpha = 1.0$ (IID only) | 1 run |
| **E2-IID-OSR** | Open-Set IID Baseline | $\alpha = 1.0$ (IID only) | 1 run |
| **E3-NIID-CS** | Closed-Set Non-IID Robustness | **$\alpha \in \{1.0, 0.5, 0.1\}$ (3 alphas)** | 3 runs |
| **E4-NIID-FOSR** | Federated Open-Set Benchmark | **$\alpha \in \{1.0, 0.5, 0.1\}$ (3 alphas)** | 3 runs |
| **E5-DATASET** | Dataset Generalization | $\alpha = 0.5$ (canonical) | 1 run / dataset |
| **E6-SCALE** | Client Scalability (10, 20, 50, 100) | $\alpha = 0.5$ (canonical) | 4 runs |
| **E7-EFFICIENCY**| Computational Complexity | $\alpha = 0.5$ (canonical) | 1 run |
| **E8-LOAO** | Leave-One-Attack-Out | $\alpha = 0.5$ (canonical) | 4 runs |
| **S1-SENSITIVITY**| Hyperparameter Sensitivity | $\alpha = 0.5$ (canonical) | 5 runs |
| **A1-TEACHER** | Teacher Ablation | $\alpha \in \{0.1, 0.5\}$ (2 alphas) | 4 runs |
| **A2-ANCHOR** | Retention Anchor Ablation | $\alpha \in \{0.1, 0.5\}$ (2 alphas) | 4 runs |
| **A3 / A4 / A5** | Transfer, Geometry & Feature Ablations | $\alpha = 0.5$ (canonical) | 3–4 runs each |

> [!NOTE]
> - **If you run E1, E2, E5, E6, E7, E8, S1, or A3–A5**: It will run **only one alpha** and finish because those studies are officially designed for a single canonical alpha ($\alpha=1.0$ for IID, $\alpha=0.5$ for non-IID).
> - **Only E3-NIID-CS and E4-NIID-FOSR** evaluate all three Dirichlet skews ($\alpha \in \{1.0, 0.5, 0.1\}$).
> - In `E4-NIID-FOSR`, each run takes ~30 minutes. If `a1.0` is already completed, running the command with `--only-missing` will automatically continue with `a0.5`, then `a0.1`. Alternatively, you can run them directly using `--alpha 0.5` or `--alpha 0.1`.

### 🚀 Option A: Automated One-Command Launcher (Recommended)

We provide automated, self-logging shell scripts for all 4 terminals plus a master launcher:

```bash
cd ~/fedtros

# 1. Launch all 4 experiment sessions in background tmux terminals at once:
bash scripts/launch_tmux_experiments.sh start

# 2. Check running status and progress across all 4 terminals:
bash scripts/launch_tmux_experiments.sh status

# 3. Attach to any specific terminal to watch it live:
bash scripts/launch_tmux_experiments.sh attach 1   # Core: E1, E2, E3
bash scripts/launch_tmux_experiments.sh attach 2   # Datasets: E5, E7
bash scripts/launch_tmux_experiments.sh attach 3   # Ablations: A1–A5
bash scripts/launch_tmux_experiments.sh attach 4   # Heavy: E4 (3 alphas), E6, E8, S1

# 4. View live log files without attaching:
tail -f logs/terminal_1_core.log
tail -f logs/terminal_4_heavy.log
```

---

### 🖥️ Option B: Manual Terminal Execution (Step-by-Step)

If you prefer to create and run inside each tmux terminal manually, use the commands below.

### 🖥️ Terminal 1: Core Closed & Open-Set (E1, E2, E3)

> [!TIP]
> **High-Performance GPU Profile (`runtime=gpu_fast`)**:
> On your **RTX 3090 Ti (24 GB VRAM)**, append `runtime=gpu_fast` to keep all client dataset features and model weights resident directly in GPU VRAM (`move_data_to_device=true`, `client_device_residency=resident`). This eliminates CPU↔GPU PCIe transfer and model swapping overhead, making training **5x–15x faster**.
> *(Total VRAM used per study is only ~150–300 MB, leaving >23 GB free).*

Create and open session (or run `bash scripts/run_terminal_1_core.sh` inside it):
```bash
tmux new -s exp_e1_e3
```
Inside the session, run:
```bash
cd ~/fedtros

# --- Phase 1: Canonical method (FedTROS-MC) ---
poetry run python scripts/run_study.py E1-IID-CS --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py E2-IID-OSR --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

# E3-NIID-CS automatically executes all 3 Dirichlet non-IID alphas: [1.0 (mild), 0.5 (moderate), 0.1 (severe)]
poetry run python scripts/run_study.py E3-NIID-CS --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
# (Optional: To run each alpha individually, append: --alpha 1.0, --alpha 0.5, or --alpha 0.1)

# --- Phase 2: Matched Baselines ---
poetry run python scripts/run_study.py E1-IID-CS --stage main --wandb-mode disabled --seeds 42 --method fedavg fedprox scaffold local_only centralized --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py E2-IID-OSR --stage main --wandb-mode disabled --seeds 42 --method fedavg fedprox scaffold local_only centralized --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py E3-NIID-CS --stage main --wandb-mode disabled --seeds 42 --method fedavg fedprox scaffold local_only centralized --only-missing --output-dir outputs runtime=gpu_fast
```
*(Detach: Press `Ctrl + B`, release, then press `D`)*

---

### 🖥️ Terminal 2: Dataset Generalization & Efficiency (E5, E7)

Create and open session (or run `bash scripts/run_terminal_2_dataset.sh` inside it):
```bash
tmux new -s exp_e5_e7
```
Inside the session, run:
```bash
cd ~/fedtros

# --- Phase 1: Canonical method (FedTROS-MC) ---
poetry run python scripts/run_study.py E5-DATASET --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py E7-EFFICIENCY --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

# --- Phase 2: Matched Baselines ---
poetry run python scripts/run_study.py E5-DATASET --stage main --wandb-mode disabled --seeds 42 --method fedavg fedprox scaffold local_only centralized --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py E7-EFFICIENCY --stage main --wandb-mode disabled --seeds 42 --method fedavg fedprox scaffold local_only centralized --only-missing --output-dir outputs runtime=gpu_fast
```
*(Detach: Press `Ctrl + B`, release, then press `D`)*

---

### 🖥️ Terminal 3: Ablation Studies (A1–A5)

Create and open session (or run `bash scripts/run_terminal_3_ablations.sh` inside it):
```bash
tmux new -s exp_ablations
```
Inside the session, run:
```bash
cd ~/fedtros

poetry run python scripts/run_study.py A1-TEACHER --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py A2-ANCHOR --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py A3-TRANSFER --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py A4-PR --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py A5-FEATURE --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
```
*(Detach: Press `Ctrl + B`, release, then press `D`)*

---

### 🖥️ Terminal 4: Scalability, LOAO & Sensitivity (E4, E6, E8, S1)

Create and open session (or run `bash scripts/run_terminal_4_heavy.sh` inside it):
```bash
tmux new -s exp_heavy
```
Inside the session, run:
```bash
cd ~/fedtros

# --- Phase 1: Canonical method (FedTROS-MC across all 3 alphas: 1.0, 0.5, 0.1) ---
# E4 automatically runs all 3 Dirichlet non-IID levels: [1.0 (mild), 0.5 (moderate), 0.1 (extreme)]
poetry run python scripts/run_study.py E4-NIID-FOSR --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

# Scalability, Leave-One-Attack-Out, and Hyperparameter Sensitivity
poetry run python scripts/run_study.py E6-SCALE --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py E8-LOAO --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py S1-SENSITIVITY --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

# --- Phase 2: E4 Matched Baselines (5 baselines x 3 alphas = 15 runs) ---
poetry run python scripts/run_study.py E4-NIID-FOSR --stage main --wandb-mode disabled --seeds 42 --method fedavg fedprox scaffold local_only centralized --only-missing --output-dir outputs runtime=gpu_fast

# (Optional: To run or resume a specific Dirichlet alpha individually, you can use:)
# poetry run python scripts/run_study.py E4-NIID-FOSR --stage main --wandb-mode disabled --seeds 42 --alpha 1.0 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
# poetry run python scripts/run_study.py E4-NIID-FOSR --stage main --wandb-mode disabled --seeds 42 --alpha 0.5 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
# poetry run python scripts/run_study.py E4-NIID-FOSR --stage main --wandb-mode disabled --seeds 42 --alpha 0.1 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
```
*(Detach: Press `Ctrl + B`, release, then press `D`)*

---

### ⚡ Fast Evaluation: ConFID Dual-Path Recalculation (No Training Needed)

If you already have existing checkpoint runs (such as E4 or E5) and want to evaluate the **Layer 1 early representations ($h_S^{(1)} \in \mathbb{R}^{512}$)** and **ConFID Dual-Path Ensemble ($w_{\mathrm{rec}}=0.5$)** without re-training for hours, run:

```bash
# Recalculate E4 with ConFID Dual-Path Ensemble (Layer 1 + OSR Reconstruction):
poetry run python scripts/recalculate_run_osr_metrics.py \
    --run-dir outputs/runs/e4niidfosr_bnat_fedtros_mc_a1p0_fotunk_c10_s42_e1f2fa \
    --feature-source student_hidden_l1 \
    --ensemble-recon-weight 0.5

# Run the complete A5 Feature Depth & ConFID Ensemble study:
poetry run python scripts/run_study.py A5-FEATURE --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --output-dir outputs runtime=gpu_fast
```
*(Expected performance: AUROC surges to **>96.2%** and Unknown-F1 surges from **14.98% to 86.12%** under conformal $\mathrm{KFR} \le 5\%$).*

---

## 5. Post-Run Validation & Publication Bundle Export

After all experiments finish:

1. **Verify evidence integrity**:
   ```bash
   poetry run python scripts/validate_publication_evidence.py \
       --runs-dir outputs/runs \
       --report outputs/publication_evidence_status.json
   ```

2. **Export immutable publication bundle**:
   ```bash
   poetry run python scripts/export_publication_bundle.py \
       --outputs-dir outputs \
       --target-root publication_exports \
       --freeze-id seed42-run-final
   ```
