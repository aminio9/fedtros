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

## 2. Managing Persistent Terminals with `tmux` (Creation & Lifecycle Guide)

When running long deep learning experiments on a remote Linux server over SSH, an accidental network disconnection, laptop lid close, or SSH timeout will immediately kill all running processes. 

`tmux` (**Terminal Multiplexer**) solves this by running terminals as persistent background daemon processes on the server. Even if your SSH session closes, your experiments continue running uninterrupted.

---

### 🔑 The Core Concept: The Prefix Key (`Ctrl + B`)
In `tmux`, every keyboard shortcut starts with the **Prefix key**:
1. Press `Ctrl` and `B` together.
2. Release both keys.
3. Press the shortcut key (e.g., `d` to detach, `[` to scroll, `s` to switch).

---

### 🛠️ Essential `tmux` Commands Cheatsheet

| Task | Command / Shortcut | Description |
| :--- | :--- | :--- |
| **Create named session** | `tmux new -s <name>` | Starts a new persistent session with a custom name. |
| **Detach from session** | Press `Ctrl + B`, then `D` | Leaves the session running safely in the background and returns to your main bash shell. |
| **List running sessions** | `tmux ls` | Lists all active tmux sessions and their current status. |
| **Reattach to session** | `tmux attach -t <name>` | Reconnects to an existing background session. |
| **Switch sessions live** | Press `Ctrl + B`, then `S` | Opens an interactive session picker. Use $\uparrow$ / $\downarrow$ arrows and hit `Enter` to switch! |
| **Kill a specific session**| `tmux kill-session -t <name>`| Stops the experiment and closes the session. |
| **Kill ALL sessions** | `tmux kill-server` | Terminates all tmux sessions and background processes cleanly. |

---

### 📜 How to Scroll & View History in `tmux` (Copy Mode)
Normally, your mouse wheel or standard terminal scroll won't scroll through past outputs in tmux. Use **Copy Mode**:
1. Press **`Ctrl + B`**, then press **`[`** (a cursor indicator appears at the top right).
2. Use **$\uparrow$ / $\downarrow$ arrow keys**, **`Page Up` / `Page Down`**, or mouse wheel to scroll up and inspect past training logs and epoch metrics.
3. Press **`q`** to exit Copy Mode and return to the live prompt.

> [!TIP]
> **Enable Mouse Scrolling Permanently**:
> Run this one-liner once on your server:
> ```bash
> echo "set -g mouse on" >> ~/.tmux.conf && tmux source ~/.tmux.conf
> ```
> Now you can scroll with your mouse wheel and click between panes naturally!

---

### 🚀 Creating & Launching Sessions in 1 Line (Detached Mode)
You don't need to manually create a session, type commands, and detach. You can launch any command or script directly inside a persistent background tmux session in one command:
```bash
tmux new-session -d -s <session_name> "bash <script_path>"
```
This is exactly how our provided automated launcher script operates!

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

## 4. 8-Terminal High-Throughput Architecture (32 GB GPU Profile)

### 💡 Why 8 Parallel Terminals?
- **Server Resources**: 24–32 GB GPU VRAM (e.g., RTX 3090 Ti / RTX 4090 / A100).
- **VRAM per Experiment**: Each FedTROS-MC or baseline experiment using `runtime=gpu_fast` occupies ~1.5–3.5 GB of GPU VRAM.
- **Parallel Capacity**: Running **8 experiments simultaneously** consumes ~18–26 GB VRAM, fully utilizing the 32 GB GPU memory without triggering out-of-memory (OOM) errors.
- **Speedup**: Benchmarks complete up to **8x faster** compared to sequential execution, completing the full research matrix in hours rather than days!

The execution protocol follows:
- **Environment**: All commands use **`poetry run`**
- **Seed**: `--seeds 42`
- **Stage**: `--stage main` (100 communication rounds, 10 clients)
- **Safety**: `--only-missing` allows safe continuation if interrupted.

---

### 📋 8-Terminal Distribution Matrix

| Terminal | Session Name | Target Studies | Method | Est. Runs | Description |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **T1** | `t1_core_mc` | `E1-IID-CS`, `E2-IID-OSR`, `E3-NIID-CS` | FedTROS-MC | 5 | Core IID & Non-IID Closed/Open-Set Benchmark |
| **T2** | `t2_e4_mc` | `E4-NIID-FOSR` ($\alpha \in \{1.0, 0.5, 0.1\}$) | FedTROS-MC | 3 | Central Federated Open-Set Benchmark across 3 skews |
| **T3** | `t3_ablations_a1_a3`| `A1-TEACHER`, `A2-ANCHOR`, `A3-TRANSFER` | FedTROS-MC | 11 | Teacher, Anchor & Knowledge Transfer Ablations |
| **T4** | `t4_ablations_a4_a5`| `A4-PR`, `A5-FEATURE` | FedTROS-MC | 8 | Prototype Geometry & Feature Depth / ConFID Ablations |
| **T5** | `t5_datasets_mc` | `E5-DATASET`, `E6-SCALE`, `E7`, `E8-LOAO`, `S1` | FedTROS-MC | 18 | 4 Datasets, Client Scalability, LOAO, Sensitivity |
| **T6** | `t6_baselines_e1_e3`| `E1-IID-CS`, `E2-IID-OSR`, `E3-NIID-CS` | 5 Baselines | 25 | FedAvg, FedProx, SCAFFOLD, Local, Centralized |
| **T7** | `t7_baselines_e4` | `E4-NIID-FOSR` ($\alpha \in \{1.0, 0.5, 0.1\}$) | 5 Baselines | 15 | Baselines for Open-Set Benchmark across 3 skews |
| **T8** | `t8_baselines_e5_e7`| `E5-DATASET` (4 Datasets), `E7-EFFICIENCY` | 5 Baselines | 25 | Baselines for Multi-Dataset & Complexity Profiling |

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
> - In `E4-NIID-FOSR`, each run takes ~30 minutes. If `a1.0` is already completed, running the command with `--only-missing` will automatically continue with `a0.5`, then `a0.1`.

---

### 🚀 Option A: Automated Master Launcher (Recommended)

Manage all 8 parallel tmux experiment terminals with a single script:

```bash
cd ~/fedtros

# 1. Start all 8 tmux sessions in parallel in the background:
./scripts/terminals/launch_tmux_experiments.sh start

# 2. Check the real-time status of all 8 sessions & latest log lines:
./scripts/terminals/launch_tmux_experiments.sh status

# 3. Attach to any specific terminal (1 through 8) to watch live training:
./scripts/terminals/launch_tmux_experiments.sh attach 1   # Core FedTROS-MC (E1, E2, E3)
./scripts/terminals/launch_tmux_experiments.sh attach 2   # E4 FedTROS-MC (3 alphas)
./scripts/terminals/launch_tmux_experiments.sh attach 3   # Ablations A1, A2, A3
./scripts/terminals/launch_tmux_experiments.sh attach 4   # Ablations A4, A5
./scripts/terminals/launch_tmux_experiments.sh attach 5   # Datasets & Scalability E5-E8, S1
./scripts/terminals/launch_tmux_experiments.sh attach 6   # Baselines E1, E2, E3
./scripts/terminals/launch_tmux_experiments.sh attach 7   # Baselines E4 (15 runs)
./scripts/terminals/launch_tmux_experiments.sh attach 8   # Baselines E5 & E7
# (Detach anytime: Ctrl + B, then D)

# 4. View live logs directly via tail without attaching:
tail -f logs/terminal_1_core_mc.log
tail -f logs/terminal_2_e4_mc.log
tail -f logs/terminal_7_baselines_e4.log

# 5. Monitor GPU VRAM across all 8 parallel processes:
watch -n 2 nvidia-smi

# 6. Stop all 8 sessions if needed:
./scripts/terminals/launch_tmux_experiments.sh stop
```

---

### 🖥️ Option B: Manual Terminal Execution (Step-by-Step)

If you prefer to start tmux sessions individually and paste commands manually:

#### 🖥️ Terminal 1: Core FedTROS-MC (`t1_core_mc`)
```bash
tmux new -s t1_core_mc
# Inside session (or run: bash scripts/terminals/run_terminal_1_core_mc.sh):
cd ~/fedtros
poetry run python scripts/run_study.py E1-IID-CS --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py E2-IID-OSR --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py E3-NIID-CS --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
# (Detach: Ctrl + B, then D)
```

#### 🖥️ Terminal 2: Central Open-Set E4 FedTROS-MC (`t2_e4_mc`)
```bash
tmux new -s t2_e4_mc
# Inside session (or run: bash scripts/terminals/run_terminal_2_e4_mc.sh):
cd ~/fedtros
# Runs all 3 Dirichlet alphas: [1.0 (mild), 0.5 (moderate), 0.1 (severe)]
poetry run python scripts/run_study.py E4-NIID-FOSR --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
# (Detach: Ctrl + B, then D)
```

#### 🖥️ Terminal 3: Core Ablations A1–A3 (`t3_ablations_a1_a3`)
```bash
tmux new -s t3_ablations_a1_a3
# Inside session (or run: bash scripts/terminals/run_terminal_3_ablations_a1_a3.sh):
cd ~/fedtros
poetry run python scripts/run_study.py A1-TEACHER --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py A2-ANCHOR --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py A3-TRANSFER --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
# (Detach: Ctrl + B, then D)
```

#### 🖥️ Terminal 4: Geometry & Feature Ablations A4–A5 (`t4_ablations_a4_a5`)
```bash
tmux new -s t4_ablations_a4_a5
# Inside session (or run: bash scripts/terminals/run_terminal_4_ablations_a4_a5.sh):
cd ~/fedtros
poetry run python scripts/run_study.py A4-PR --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py A5-FEATURE --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
# (Detach: Ctrl + B, then D)
```

#### 🖥️ Terminal 5: Multi-Dataset, Scalability, LOAO & Sensitivity (`t5_datasets_mc`)
```bash
tmux new -s t5_datasets_mc
# Inside session (or run: bash scripts/terminals/run_terminal_5_datasets_mc.sh):
cd ~/fedtros
poetry run python scripts/run_study.py E5-DATASET --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py E6-SCALE --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py E7-EFFICIENCY --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py E8-LOAO --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py S1-SENSITIVITY --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast
# (Detach: Ctrl + B, then D)
```

#### 🖥️ Terminal 6: Core Baselines E1, E2, E3 (`t6_baselines_e1_e3`)
```bash
tmux new -s t6_baselines_e1_e3
# Inside session (or run: bash scripts/terminals/run_terminal_6_baselines_e1_e3.sh):
cd ~/fedtros
poetry run python scripts/run_study.py E1-IID-CS --stage main --wandb-mode disabled --seeds 42 --method fedavg fedprox scaffold local_only centralized --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py E2-IID-OSR --stage main --wandb-mode disabled --seeds 42 --method fedavg fedprox scaffold local_only centralized --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py E3-NIID-CS --stage main --wandb-mode disabled --seeds 42 --method fedavg fedprox scaffold local_only centralized --only-missing --output-dir outputs runtime=gpu_fast
# (Detach: Ctrl + B, then D)
```

#### 🖥️ Terminal 7: Open-Set E4 Baselines (`t7_baselines_e4`)
```bash
tmux new -s t7_baselines_e4
# Inside session (or run: bash scripts/terminals/run_terminal_7_baselines_e4.sh):
cd ~/fedtros
# Runs 5 baselines x 3 Dirichlet alphas = 15 runs
poetry run python scripts/run_study.py E4-NIID-FOSR --stage main --wandb-mode disabled --seeds 42 --method fedavg fedprox scaffold local_only centralized --only-missing --output-dir outputs runtime=gpu_fast
# (Detach: Ctrl + B, then D)
```

#### 🖥️ Terminal 8: Multi-Dataset & Efficiency Baselines (`t8_baselines_e5_e7`)
```bash
tmux new -s t8_baselines_e5_e7
# Inside session (or run: bash scripts/terminals/run_terminal_8_baselines_e5_e7.sh):
cd ~/fedtros
poetry run python scripts/run_study.py E5-DATASET --stage main --wandb-mode disabled --seeds 42 --method fedavg fedprox scaffold local_only centralized --only-missing --output-dir outputs runtime=gpu_fast
poetry run python scripts/run_study.py E7-EFFICIENCY --stage main --wandb-mode disabled --seeds 42 --method fedavg fedprox scaffold local_only centralized --only-missing --output-dir outputs runtime=gpu_fast
# (Detach: Ctrl + B, then D)
```

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
