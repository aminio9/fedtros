# FedTROS-PR Full Experiment Runbook for Google Colab

**Purpose:** execute the complete post-refactor FedTROS-PR experiment contract from E0 through E8, A1 through A5, and S1 with durable outputs, exact resume behavior, and publication-ready provenance.

**Repository state used for this runbook:** 20 August 2026 workspace state.

**Canonical method:** FedTROS-PR with a private Variational Classifier Teacher (VCT), a federated student, coverage-adaptive anchoring, disagreement-gated knowledge distillation, feature alignment, and known-only Prototype-Rank rejection.

---

## 1. Read this before using Colab

### 1.1 One primary Google account only

Do **not** use four Google accounts as four workers, rotate between them when one account reaches a limit, or otherwise combine their free quotas. The official [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) explicitly prohibits using multiple accounts to work around access or resource-usage restrictions. It also states that free resources, GPU types, timeouts, and maximum VM lifetimes are dynamic and not guaranteed.

This document therefore uses four **logical experiment batches**, not four Google accounts. Run them from one primary account, or move the same frozen workflow to paid Colab, GCP, a university server, or a local GPU if free Colab cannot finish the grid.

Also do not use anti-idle JavaScript, automated reconnectors, remote shells, background distributed workers, or similar quota-evasion techniques.

### 1.2 Full-grid feasibility

The live study YAMLs expand to **337 independent runs** when A5 is included. A free Colab account is appropriate for validation, pilots, and some final cells, but completion time is not predictable and the entire grid is not guaranteed to fit free quotas.

After the first representative paper-scale run, measure:

```text
estimated GPU time = median representative run time × remaining run count
estimated storage  = median completed run size × remaining run count + raw data + 25% headroom
```

Do not start the full matrix until both estimates are acceptable.

### 1.3 Scientific ordering is mandatory

Do not run the headline grid immediately. The correct order is:

```text
tests and E0
  -> engineering pilot
  -> A1/A2/A3/A4/A5 and S1 gates
  -> record decisions
  -> freeze code/config/data
  -> E1-E8 paper-final runs
  -> completeness audit
  -> statistics and immutable export
  -> figures in the separate plots repository
```

If an architecture gate changes the canonical method, update the configuration, create a new freeze, and only then start E1-E8. Never mix pre-gate and post-gate headline runs.

---

## 2. Sources of truth

Use the live declarative files, not the retired shell scripts:

- Global composition: `src/configs/config.yaml`
- Study matrices: `src/configs/study/*.yaml`
- Stage profiles: `src/configs/stage/*.yaml`
- Method definitions: `src/configs/method/*.yaml`
- Dataset contracts: `src/configs/dataset/*.yaml`
- VCT/student training defaults: `src/configs/training/default.yaml`
- Prototype-Rank configuration: `src/configs/open_set/fedtros_pr.yaml`
- Federated defaults: `src/configs/federated/default.yaml`
- Tracking: `src/configs/tracking/wandb.yaml`
- Checkpointing: `src/configs/checkpointing/default.yaml`
- Authoritative matrix runner: `scripts/run_study.py`
- Authoritative resume runner: `scripts/resume.py`

The old files under `scripts/experiments/` are migration references only. Do not use `run_full_suite.sh`, old DQN-era commands, or old checkpoints for final evidence.

---

## 3. Exact experiment matrix

All counts below were derived from the live study definitions.

| Study | Stage | Matrix | Runs |
|---|---|---|---:|
| E0-VERIFY | smoke | 1 method × 1 seed | 1 |
| A1-TEACHER | ablation | 2 alphas × 3 seeds × 4 variants | 24 |
| A2-ANCHOR | ablation | 2 alphas × 3 seeds × 3 variants | 18 |
| A3-TRANSFER | ablation | 1 alpha × 3 seeds × 4 variants | 12 |
| A4-PR | ablation | 1 alpha × 5 seeds × 5 variants | 25 |
| A5-FEATURE | ablation | 1 alpha × 5 seeds × 2 variants | 10 |
| S1-SENSITIVITY | tuning | 1 alpha × 3 seeds × 4 variants | 12 |
| E1-IID-CS | paper_final | 2 datasets × 3 methods × 5 seeds | 30 |
| E2-IID-OSR | paper_final | 1 method × 5 seeds | 5 |
| E3-NIID-CS | paper_final | 3 methods × 3 alphas × 5 seeds | 45 |
| E4-NIID-FOSR | paper_final | 3 methods × 3 alphas × 5 seeds | 45 |
| E5-DATASET | paper_final | 4 datasets × 3 methods × 5 seeds | 60 |
| E6-SCALE | paper_final | 3 client counts × 5 seeds | 15 |
| E7-EFFICIENCY | paper_final | 3 methods × 5 seeds | 15 |
| E8-LOAO | paper_final | 4 held-out attacks × 5 seeds | 20 |
| **Total** |  |  | **337** |

Canonical headline seeds:

```text
17, 42, 73, 101, 137
```

The three-seed gate/sensitivity studies use:

```text
17, 42, 73
```

### 3.1 Study-specific conditions

| Study | Dataset(s) | Method(s) | Alpha/IID | Unknown condition |
|---|---|---|---|---|
| E1 | BNaT, BTAT | FedTROS-PR, FedAvg, FedProx | IID | none; all labels known |
| E2 | BNaT | FedTROS-PR | IID | FoT |
| E3 | BNaT | FedTROS-PR, FedAvg, FedProx | α = 1.0, 0.5, 0.1 | none |
| E4 | BNaT | FedTROS-PR, FedAvg, FedProx | α = 1.0, 0.5, 0.1 | FoT |
| E5 | BNaT, BTAT, CIC-IDS2017, ToN-IoT | FedTROS-PR, FedAvg, FedProx | α = 0.5 | dataset-specific frozen holdout |
| E6 | BNaT | FedTROS-PR | α = 0.5; clients = 10, 50, 100 | FoT |
| E7 | BNaT | FedTROS-PR, FedAvg, FedProx | α = 0.5 | FoT |
| E8 | BNaT | FedTROS-PR | α = 0.5 | BP, DoS, MitM, or FoT, one at a time |

### 3.2 Frozen E5 protocols

| Dataset | Known labels | Held-out unknown labels |
|---|---|---|
| BNaT | Normal, BP, DoS, MitM | FoT |
| BTAT | Normal, DoS, OaU, FoT, FDV | Re, DeC |
| CIC-IDS2017 | BENIGN, DDoS, DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, FTP-Patator, PortScan, SSH-Patator | Bot, Heartbleed, Infiltration, Web Attack-Brute Force, Web Attack-Sql Injection, Web Attack-XSS |
| ToN-IoT | normal, backdoor, ddos, dos, injection, password, scanning | mitm, ransomware, xss |

### 3.3 Ablation variants

- A1: `no_teacher`, `deterministic_teacher`, `vct_beta0`, `vct`
- A2: `no_anchor`, `fixed_anchor`, `adaptive_anchor`
- A3: `no_kd_no_alignment`, `ungated_kd`, `gated_kd`, `full_transfer`
- A4: `msp`, `energy`, `positive_only`, `boundary_raw`, `full_rank`
- A5: `student_embedding`, `osr_branch_embedding`
- S1: `beta0`, `beta_small`, `beta_canonical`, `beta_large`

The source audit confirms that a real optional student OSR branch exists, but it is disabled by default. A5 determines whether it earns inclusion. If it does not win reproducibly without unacceptable utility/efficiency cost, keep `training.student_osr_enabled=false` and `feature_source=student_embedding`.

---

## 4. Canonical configuration snapshot

The resolved YAML saved inside every run is authoritative. This summary is a human checklist.

### 4.1 Paper-scale training

```yaml
federated:
  num_rounds: 100
  num_clients: 10          # E6 overrides with 10, 50, 100
  strategy:
    student_aggregation_mode: support_weighted_average
    support_min_weight: 0.01

training:
  local_epochs: 2
  batch_size: 64
  label_smoothing: 0.02
  grad_clip_norm: 1.0

  teacher_enabled: true
  teacher_stochastic_training: true
  teacher_hidden_dims: [512, 256]
  teacher_latent_dim: 64
  teacher_lr: 0.001
  teacher_weight_decay: 0.0001
  teacher_beta_kl: 0.01
  teacher_epochs: 2

  fedtros_student_hidden_dims: [512, 256, 128]
  fedtros_student_activation: gelu
  fedtros_student_norm: layernorm
  fedtros_student_dropout: 0.05
  student_lr: 0.001
  student_weight_decay: 0.0001

  fedtros_global_anchor_weight: 2.0
  fedtros_global_anchor_min_weight: 0.0
  fedtros_global_anchor_coverage_power: 1.0

  kd_enabled: true
  kd_gating_enabled: true
  alignment_enabled: true
  lambda_kd: 0.20
  lambda_align: 0.08
  kd_base_temperature: 3.0
  kd_min_temperature: 1.0
  kd_decay: 0.95

  student_osr_enabled: false
  student_osr_latent_dim: 8
```

### 4.2 Prototype-Rank

```yaml
open_set:
  detector: prototype_rank
  evaluate_each_round: false
  calibration:
    prototype_fit_fraction: 0.70
    threshold_calibration_fraction: 0.30
    target_known_fpr: 0.05
    min_samples_per_class: 50
    fit_correct_only: true
    strict_disjoint: true
  prototype_rank:
    strict_no_unknown_train: true
    prototype:
      feature_source: student_embedding
      normalize: true
      radius_quantile: 0.95
      num_prototypes_per_class: 16
      min_samples_per_prototype: 25
      seed: 42
      negative:
        enabled: true
        num_prototypes: 32
        max_samples: 5000
        radius_quantile: 0.75
        weight: 0.35
        random_seed: 43
    score_fusion:
      method: prototype_rank
      calibration_scope: global
      decision_rule: selected_rank_threshold
```

### 4.3 Matched baselines

- FedAvg uses the identical student, two local epochs, batch size 64, no teacher/KD/alignment/anchor/OSR branch, standard FedAvg aggregation, and reset optimizer each round.
- FedProx uses the same matched student/budget with `proximal_mu=0.01`.
- Open-set evaluation for matched baseline runs is selected by each study's open-set protocol.

### 4.4 Runtime and tracking

```yaml
runtime:
  device_prefer: gpu
  allow_cpu_fallback: false
  deterministic: true
  benchmark: true

tracking:
  backend: wandb
  project: FedTROS-PR
  mode: online             # online | offline | disabled

checkpointing:
  save_best: true
  save_latest: true
  save_final: true
  include_rng_state: true
  strict_load: true
```

W&B is for monitoring. `outputs/runs/<run_id>/` is the scientific source of truth.

---

## 5. One-time preparation on the Windows workstation

Perform this only after the code/config is ready for a freeze. Replace `<FREEZE>` with a stable label such as `fedtros-pr-rc1`.

```powershell
cd D:\Research\Code\fedtros
git status
git rev-parse HEAD
git archive --format=zip --output="fedtros-source-<FREEZE>.zip" HEAD
tar -czf "fedtros-data-raw-<FREEZE>.tar.gz" data/raw
Get-FileHash "fedtros-source-<FREEZE>.zip" -Algorithm SHA256
Get-FileHash "fedtros-data-raw-<FREEZE>.tar.gz" -Algorithm SHA256
```

`git status` must be clean, or all intended changes must be committed before `git archive`; the archive contains `HEAD`, not uncommitted workspace edits.

Upload these files to the primary account's Drive:

```text
MyDrive/FedTROS/freeze/<FREEZE>/fedtros-source-<FREEZE>.zip
MyDrive/FedTROS/freeze/<FREEZE>/fedtros-data-raw-<FREEZE>.tar.gz
MyDrive/FedTROS/freeze/<FREEZE>/SHA256SUMS.txt
```

Do not edit the archive after recording its hash. A new code or configuration decision requires a new freeze label and archive.

The current raw canonical CSVs occupy roughly 1.33 GB before any additional source archives:

```text
BNaT.csv          ~15.7 MB
BTAT.csv         ~394.4 MB
CIC-IDS2017.csv  ~902.1 MB
ToN-IoT.csv       ~17.0 MB
```

---

## 6. Drive layout

Use one deeply nested project folder rather than thousands of items in Drive root.

```text
MyDrive/FedTROS/
  freeze/<FREEZE>/
    fedtros-source-<FREEZE>.zip
    fedtros-data-raw-<FREEZE>.tar.gz
    poetry.lock
    SHA256SUMS.txt
    FREEZE_MANIFEST.md
  run-store/<FREEZE>/
    outputs/
    paper_results/
    publication_exports/
  notebooks/
    FedTROS_Run.ipynb
```

Colab advises minimizing many small Drive reads and copying archives to the VM before extraction. The source and raw data therefore run from `/content`; only durable experiment outputs are rooted in Drive. See the [Colab FAQ Drive guidance](https://research.google.com/colaboratory/faq.html#drive-timeout).

---

## 7. Start every new Colab runtime

In Colab, select **Runtime → Change runtime type → GPU**. Do not assume a particular GPU model.

### Cell 1 — mount Drive and define the freeze

```python
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path

FREEZE = "fedtros-pr-rc1"  # change once, then keep fixed
DRIVE_ROOT = Path("/content/drive/MyDrive/FedTROS")
FREEZE_DIR = DRIVE_ROOT / "freeze" / FREEZE
RUN_STORE = DRIVE_ROOT / "run-store" / FREEZE
OUTPUT_DIR = RUN_STORE / "outputs"
REPO = Path("/content/fedtros")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(FREEZE_DIR)
print(OUTPUT_DIR)
```

### Cell 2 — verify Python and GPU

```python
import platform, subprocess, sys

print("Python:", sys.version)
assert (3, 11) <= sys.version_info[:2] < (3, 13), "FedTROS requires Python 3.11 or 3.12"
subprocess.run(["nvidia-smi"], check=True)
```

If no NVIDIA GPU is present, stop and request a GPU runtime. Paper-scale runs must not silently fall back to CPU.

### Cell 3 — copy and verify the frozen archives

Fill in the expected SHA-256 strings from `SHA256SUMS.txt`.

```python
import hashlib, shutil

SOURCE_ARCHIVE = FREEZE_DIR / f"fedtros-source-{FREEZE}.zip"
DATA_ARCHIVE = FREEZE_DIR / f"fedtros-data-raw-{FREEZE}.tar.gz"
LOCAL_SOURCE = Path("/content/fedtros-source.zip")
LOCAL_DATA = Path("/content/fedtros-data-raw.tar.gz")

EXPECTED_SOURCE_SHA256 = "PASTE_SOURCE_SHA256"
EXPECTED_DATA_SHA256 = "PASTE_DATA_SHA256"

def sha256(path, block=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()

shutil.copy2(SOURCE_ARCHIVE, LOCAL_SOURCE)
shutil.copy2(DATA_ARCHIVE, LOCAL_DATA)
assert sha256(LOCAL_SOURCE) == EXPECTED_SOURCE_SHA256
assert sha256(LOCAL_DATA) == EXPECTED_DATA_SHA256
print("Frozen archives verified")
```

### Cell 4 — extract into the ephemeral VM

```python
import shutil, subprocess

if REPO.exists():
    shutil.rmtree(REPO)
REPO.mkdir(parents=True)

subprocess.run(["unzip", "-q", str(LOCAL_SOURCE), "-d", str(REPO)], check=True)
subprocess.run(["tar", "-xzf", str(LOCAL_DATA), "-C", str(REPO)], check=True)
print(REPO)
```

Deleting `/content/fedtros` here is safe because it is the ephemeral Colab copy, not the Drive run store.

### Cell 5 — install the frozen environment

The project supports Python 3.11/3.12. Generate the lock once on the target runtime, save it with the freeze, and reuse it on later sessions.

```python
import subprocess, shutil

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "poetry>=2,<3"], check=True)
subprocess.run(["poetry", "config", "virtualenvs.create", "false"], cwd=REPO, check=True)

DRIVE_LOCK = FREEZE_DIR / "poetry.lock"
LOCAL_LOCK = REPO / "poetry.lock"
if DRIVE_LOCK.exists():
    shutil.copy2(DRIVE_LOCK, LOCAL_LOCK)
else:
    subprocess.run(["poetry", "lock"], cwd=REPO, check=True)
    shutil.copy2(LOCAL_LOCK, DRIVE_LOCK)

subprocess.run(["poetry", "install", "--only", "main", "--no-interaction"], cwd=REPO, check=True)
```

Never regenerate `poetry.lock` halfway through a freeze.

### Cell 6 — optional W&B secret

Store `WANDB_API_KEY` in Colab's Secrets panel. Do not paste it into the notebook, YAML, Drive text files, or Git.

```python
import os
from google.colab import userdata

WANDB_MODE = "online"  # use "offline" or "disabled" when appropriate
if WANDB_MODE == "online":
    os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
```

If using offline mode, remember that the complete local run directory is still required. An offline W&B directory alone is not the scientific result.

### Cell 7 — reusable safe runner

```python
import subprocess

def dry_run(study_id, stage, *filters):
    cmd = [
        "poetry", "run", "python", "scripts/run_study.py",
        study_id,
        "--stage", stage,
        "--dry-run",
        "--output-dir", str(OUTPUT_DIR),
        *map(str, filters),
    ]
    return subprocess.run(cmd, cwd=REPO, check=False)

def run_study(study_id, stage, *filters):
    cmd = [
        "poetry", "run", "python", "scripts/run_study.py",
        study_id,
        "--stage", stage,
        "--only-missing",
        "--resume",
        "--continue-on-error",
        "--max-parallel", "1",
        "--gpus", "0",
        "--wandb-mode", WANDB_MODE,
        "--output-dir", str(OUTPUT_DIR),
        *map(str, filters),
    ]
    result = subprocess.run(cmd, cwd=REPO, check=False)
    print("exit code:", result.returncode)
    return result
```

Use `max_parallel=1` on a one-GPU Colab runtime. Parallel runs on one GPU can corrupt timing evidence, cause out-of-memory failures, and make E7 invalid.

---

## 8. Pre-flight and E0 gate

### Cell 8 — tests and repository doctor

```python
subprocess.run(["poetry", "run", "pytest", "-q"], cwd=REPO, check=True)
subprocess.run([
    "poetry", "run", "python", "scripts/doctor.py",
    "--wandb-mode", WANDB_MODE,
    "--output-dir", str(OUTPUT_DIR),
], cwd=REPO, check=True)
subprocess.run(["poetry", "run", "python", "scripts/studies.py", "list"], cwd=REPO, check=True)
```

The doctor may warn that the separate plots repository is not mounted. That is acceptable during training; all dependency, CUDA, output, and study checks must otherwise pass.

### Cell 9 — E0 dry-run, then execution

```python
dry_run("E0-VERIFY", "smoke")
run_study("E0-VERIFY", "smoke")
```

Inspect the output before proceeding:

```python
subprocess.run([
    "poetry", "run", "python", "scripts/runs.py",
    "--outputs-dir", str(OUTPUT_DIR), "summary",
], cwd=REPO, check=False)
```

### E0 go/no-go checklist

- [ ] All unit/integration tests pass.
- [ ] CUDA is available and recorded.
- [ ] E0 manifest status is `COMPLETED`.
- [ ] No DQN/RL state is present.
- [ ] Unknown samples are absent from training/prototype/calibration sets.
- [ ] Prototype-fit and calibration IDs are disjoint.
- [ ] Checkpoint payload validation passes.
- [ ] Resume-integrity validation passes.
- [ ] Resolved configuration and all manifests exist.
- [ ] Local metrics exist even if W&B is offline/disabled.

Do not continue if any scientific gate fails.

---

## 9. Run one representative paper-scale pilot

Use seed 42, BNaT, FedTROS-PR, α=0.5 from E4. First dry-run it:

```python
dry_run(
    "E4-NIID-FOSR", "paper_final",
    "--method", "fedtros_pr",
    "--alpha", "0.5",
    "--seeds", "42",
)
```

Then execute:

```python
run_study(
    "E4-NIID-FOSR", "paper_final",
    "--method", "fedtros_pr",
    "--alpha", "0.5",
    "--seeds", "42",
)
```

Record:

- assigned GPU model and memory;
- wall-clock duration;
- peak GPU/system memory;
- completed run-directory size;
- checkpoint size;
- whether a runtime interruption can resume from Drive;
- W&B/local metric agreement.

This pilot is **engineering evidence only** until the architecture gates and final freeze are complete. If the freeze changes, rerun this scientific cell under the new identity.

---

## 10. Four logical batches

These batches express dependencies and make the ledger manageable. They are not account assignments and should not be used concurrently to multiply free Colab quota.

### Batch 1 — method gates

Run A1, A2, A3, A4, A5, and S1 before freezing the final method.

Safe Colab-sized slices:

```python
# A1: one alpha/seed slice = 4 variants
run_study("A1-TEACHER", "ablation", "--alpha", "0.1", "--seeds", "17")

# A2: one alpha/seed slice = 3 variants
run_study("A2-ANCHOR", "ablation", "--alpha", "0.1", "--seeds", "17")

# A3: one seed slice = 4 variants
run_study("A3-TRANSFER", "ablation", "--seeds", "17")

# A4: one seed slice = 5 detector variants
run_study("A4-PR", "ablation", "--seeds", "17")

# A5: one seed slice = 2 feature-source variants
run_study("A5-FEATURE", "ablation", "--seeds", "17")

# S1: one seed slice = 4 beta variants
run_study("S1-SENSITIVITY", "tuning", "--seeds", "17")
```

Repeat the A1/A2 alpha-seed slices for:

```text
alpha ∈ {0.1, 0.5}
seed  ∈ {17, 42, 73}
```

Repeat A3 and S1 for seeds `17, 42, 73`. Repeat A4 and A5 for all five headline seeds.

Gate decisions must be written into `FREEZE_MANIFEST.md`. At minimum record:

```text
canonical teacher variant
canonical anchor variant
canonical KD/alignment variant
canonical detector variant
canonical feature source
canonical teacher_beta_kl
decision metric(s), paired seed set, and uncertainty
```

### Batch 2 — IID and closed-set evidence

#### E1 — exact one-run slice

```python
run_study(
    "E1-IID-CS", "paper_final",
    "--dataset", "bnat",
    "--method", "fedtros_pr",
    "--seeds", "17",
)
```

Complete the Cartesian product:

```text
dataset ∈ {bnat, btat}
method  ∈ {fedtros_pr, fedavg, fedprox}
seed    ∈ {17, 42, 73, 101, 137}
```

#### E2 — one-run seed slice

```python
run_study("E2-IID-OSR", "paper_final", "--seeds", "17")
```

Repeat for all five seeds.

#### E3 — exact one-run slice

```python
run_study(
    "E3-NIID-CS", "paper_final",
    "--method", "fedtros_pr",
    "--alpha", "0.1",
    "--seeds", "17",
)
```

Complete:

```text
method ∈ {fedtros_pr, fedavg, fedprox}
alpha  ∈ {1.0, 0.5, 0.1}
seed   ∈ {17, 42, 73, 101, 137}
```

### Batch 3 — central open-set evidence

#### E4 — exact one-run slice

```python
run_study(
    "E4-NIID-FOSR", "paper_final",
    "--method", "fedtros_pr",
    "--alpha", "0.1",
    "--seeds", "17",
)
```

Complete the same method/alpha/seed Cartesian product as E3.

#### E8 — one seed slice

```python
run_study("E8-LOAO", "paper_final", "--seeds", "17")
```

One E8 seed slice expands to the four held-out attacks BP, DoS, MitM, and FoT. Repeat for all five seeds. If four runs are too long for one session, use the exact single-run interface documented in `README.md`, specifying `study=E8-LOAO`, `stage=paper_final`, `seed=<seed>`, and `unknown=<attack>`, plus the same output root and tracking mode. Always dry-run/inspect the resolved configuration first.

#### E7 — exact one-run slice

```python
run_study(
    "E7-EFFICIENCY", "paper_final",
    "--method", "fedtros_pr",
    "--seeds", "17",
)
```

Complete:

```text
method ∈ {fedtros_pr, fedavg, fedprox}
seed   ∈ {17, 42, 73, 101, 137}
```

For valid E7 timing, do not run another GPU workload simultaneously. Record the actual assigned GPU for every run; do not pool timings from materially different hardware without stratifying them.

### Batch 4 — datasets and scalability

#### E5 — exact one-run slice

```python
run_study(
    "E5-DATASET", "paper_final",
    "--dataset", "cicids2017",
    "--method", "fedtros_pr",
    "--seeds", "17",
)
```

Complete:

```text
dataset ∈ {bnat, btat, cicids2017, toniot}
method  ∈ {fedtros_pr, fedavg, fedprox}
seed    ∈ {17, 42, 73, 101, 137}
```

Before E5, validate the prepared external datasets:

```python
subprocess.run([
    "poetry", "run", "python", "scripts/prepare_external_datasets.py",
    "--dataset", "all", "--raw-root", "data/raw",
], cwd=REPO, check=True)
```

This command should validate and reuse the frozen canonical CSVs. If it tries to replace data, stop and investigate the manifest/checksum mismatch; do not silently change data inside a freeze.

#### E6 — exact one-run slice

```python
run_study(
    "E6-SCALE", "paper_final",
    "--clients", "10",
    "--seeds", "17",
)
```

Complete:

```text
clients ∈ {10, 50, 100}
seed    ∈ {17, 42, 73, 101, 137}
```

E6 is a fixed-global-data fragmentation/scalability study. Do not describe it as proving a pure causal effect of client count. The 50/100-client cells may create many per-client artifacts; monitor Drive usage closely.

---

## 11. Full-study commands for a stable server

These commands are convenient on a persistent GPU server. On free Colab, prefer the slices above.

```bash
poetry run python scripts/run_study.py A1-TEACHER --stage ablation --only-missing --resume --continue-on-error
poetry run python scripts/run_study.py A2-ANCHOR --stage ablation --only-missing --resume --continue-on-error
poetry run python scripts/run_study.py A3-TRANSFER --stage ablation --only-missing --resume --continue-on-error
poetry run python scripts/run_study.py A4-PR --stage ablation --only-missing --resume --continue-on-error
poetry run python scripts/run_study.py A5-FEATURE --stage ablation --only-missing --resume --continue-on-error
poetry run python scripts/run_study.py S1-SENSITIVITY --stage tuning --only-missing --resume --continue-on-error

poetry run python scripts/run_study.py E1-IID-CS --stage paper_final --only-missing --resume --continue-on-error
poetry run python scripts/run_study.py E2-IID-OSR --stage paper_final --only-missing --resume --continue-on-error
poetry run python scripts/run_study.py E3-NIID-CS --stage paper_final --only-missing --resume --continue-on-error
poetry run python scripts/run_study.py E4-NIID-FOSR --stage paper_final --only-missing --resume --continue-on-error
poetry run python scripts/run_study.py E5-DATASET --stage paper_final --only-missing --resume --continue-on-error
poetry run python scripts/run_study.py E6-SCALE --stage paper_final --only-missing --resume --continue-on-error
poetry run python scripts/run_study.py E7-EFFICIENCY --stage paper_final --only-missing --resume --continue-on-error
poetry run python scripts/run_study.py E8-LOAO --stage paper_final --only-missing --resume --continue-on-error
```

Add the same `--wandb-mode`, `--output-dir`, GPU, and parallelism options used by the notebook. Do not use `--force-new` to hide a failed or misconfigured run.

---

## 12. Resume and interruption recovery

After a Colab disconnect:

1. Start a new GPU runtime.
2. Repeat Section 7 using the identical freeze.
3. Confirm the same Drive `OUTPUT_DIR`.
4. Dry-run the exact slice.
5. Run the same slice with `--only-missing --resume`.

The runner will skip completed identities. It resumes a compatible interrupted/failed identity only when a valid checkpoint exists. Resume is rejected when checkpoint schema, configuration hash, teacher type, partition hash, or required private teacher state is incompatible.

Never:

- copy an old DQN-era checkpoint into a new run;
- rename run directories;
- edit `resolved_config.yaml` after a run starts;
- resume with a different freeze or dataset hash;
- delete a failed manifest and pretend the cell is new;
- use `--force-new` unless the scientific identity intentionally changed.

For one known run ID, the explicit interface is:

```bash
poetry run python scripts/resume.py RUN_ID
```

Run it from the repository with the same output-root context expected by the run. The study-level `--resume` path is safer for the Drive layout because it discovers the planned identity.

---

## 13. Run ledger and daily checks

At the end of every session record:

```text
date/time UTC
freeze ID and source/data SHA-256
study/stage/filter slice
planned run IDs
completed, failed, interrupted, resumable counts
GPU model
Colab termination or error reason
Drive free space
W&B mode
notes on any rerun decision
```

Use the repository tools:

```bash
poetry run python scripts/runs.py --outputs-dir /path/to/outputs summary
poetry run python scripts/runs.py --outputs-dir /path/to/outputs failed --study E4-NIID-FOSR
poetry run python scripts/runs.py --outputs-dir /path/to/outputs resumable
poetry run python scripts/runs.py --outputs-dir /path/to/outputs missing E8-LOAO --stage paper_final
```

Before launching a slice, always inspect it:

```python
dry_run("E3-NIID-CS", "paper_final", "--method", "fedprox", "--alpha", "0.1", "--seeds", "137")
```

The dry-run should show exactly one planned scientific cell for exact E1/E3/E4/E5/E6/E7 slices.

---

## 14. Storage policy

The persistent directory is:

```text
MyDrive/FedTROS/run-store/<FREEZE>/outputs/runs/<run_id>/
```

Keep completed run directories immutable. Do not manually merge partial directories or overwrite a completed identity.

Because free Drive storage may be insufficient for 337 full run directories:

1. measure E0 and one representative paper-run sizes;
2. separately measure E5/CIC-IDS2017 and E6/100-client output sizes;
3. project total storage with 25% headroom;
4. if it does not fit, stop before the full grid and move the unchanged freeze to larger authorized storage/compute;
5. do not distribute the run store across the user's other free accounts to evade quotas.

Do not delete run-local data, checkpoints, predictions, or artifacts unless the publication/export contract has been audited and a documented archival policy explicitly permits it.

---

## 15. Final completeness audit

Before aggregation, every required identity must have `status=COMPLETED` and consistent hashes.

Expected totals:

```text
smoke:       E0 = 1
ablation:    A1 + A2 + A3 + A4 + A5 = 89
tuning:      S1 = 12
paper_final: E1-E8 = 235
grand total: 337
```

Checklist:

- [ ] 337 expected runs are accounted for.
- [ ] No required run is `FAILED`, `INTERRUPTED`, `UNINITIALIZED`, or corrupted.
- [ ] All completed runs use the final code/config/data freeze.
- [ ] Paired methods reuse identical partition manifests per condition and seed.
- [ ] Headline studies have all five seeds.
- [ ] E8 contains all four holdouts for all five seeds.
- [ ] E5 contains all four datasets, three methods, and five seeds.
- [ ] E6 contains 10, 50, and 100 clients for all five seeds.
- [ ] A5 decision is consistent with the canonical feature-source claim.
- [ ] Communication values come from actual payload instrumentation.
- [ ] Runtime comparisons are hardware-aware.
- [ ] W&B and local metrics agree where both exist.
- [ ] Every final number can be traced to run IDs and provenance files.

If A5 is intentionally excluded after a documented pre-run decision, the total becomes 327. In the current source state A5 is applicable because the optional branch exists, so the default expectation is 337.

---

## 16. Build statistics and the immutable publication export

Run only after the completeness audit:

```python
PAPER_RESULTS = RUN_STORE / "paper_results"
EXPORT_ROOT = RUN_STORE / "publication_exports"

subprocess.run([
    "poetry", "run", "python", "scripts/build_q1_results.py",
    "--outputs-dir", str(OUTPUT_DIR),
    "--target", str(PAPER_RESULTS),
    "--stage", "paper_final", "ablation", "tuning", "reproduction",
], cwd=REPO, check=True)

subprocess.run([
    "poetry", "run", "python", "scripts/export_publication_bundle.py",
    "--outputs-dir", str(OUTPUT_DIR),
    "--target-root", str(EXPORT_ROOT),
    "--freeze-id", f"{FREEZE}-paper-final-01",
    "--include-stages", "paper_final", "ablation", "tuning", "reproduction",
], cwd=REPO, check=True)
```

The export is immutable. If the freeze ID already exists, the command should fail rather than overwrite it.

Also export the strict input contract for the existing 29-figure plots workflow if required:

```python
PLOT_DATA = RUN_STORE / "paper_data"
subprocess.run([
    "poetry", "run", "python", "scripts/export_plot_data.py",
    "--runs-dir", str(OUTPUT_DIR),
    "--stage", "paper_final", "ablation", "tuning", "reproduction",
    "--output-dir", str(PLOT_DATA),
], cwd=REPO, check=True)
```

Do not manually type table values into the paper.

---

## 17. Render figures in the separate plots repository

The `plots` repository never trains models or recomputes canonical statistics. Give it only the frozen publication bundle or strict exported plot data.

On a persistent machine with both repositories:

```bash
cd ../plots
python scripts/generate_all.py \
  --data-dir ../fedtros/paper_data \
  --figures-dir outputs/figures \
  --tables-dir outputs/tables

python scripts/verify_outputs.py \
  --figures-dir outputs/figures
```

Here, `paper_data` means the directory produced by `scripts/export_plot_data.py`, not the compact publication-bundle directory. The compact immutable bundle is a separate evidence package; do not pass it to `--data-dir` unless a consumer explicitly supports that bundle schema.

---

## 18. Failure playbook

### GPU unavailable

Stop. Change the runtime to GPU or wait for authorized availability. Do not run final cells with CPU fallback.

### CUDA out of memory

Record the failure and assigned GPU. First verify no other run is using the device. Do not change batch size or architecture for only one seed. Any scientific budget change must be global, documented, frozen, and rerun consistently.

### Drive `Input/output error`

Stop the run, verify Drive capacity and folder size, reconnect/mount Drive, then resume from the same slice. Do not create a second divergent output tree. Google recommends using archives for many small reads and waiting for Drive operation quotas to reset when necessary.

### Runtime disconnect

Rebuild the ephemeral VM from the same archives and rerun the same slice with `--only-missing --resume`.

### Failed hash/config validation

Do not override it. Compare source archive hash, data archive hash, `poetry.lock`, partition manifest, resolved config, and checkpoint schema. Resume only when they match.

### One method/seed fails repeatedly

Keep the failed evidence, diagnose the cause, fix it under a new freeze if necessary, and rerun every affected paired cell. Do not omit an inconvenient seed.

### Colab quota reached

Stop and wait for normal availability, or move the unchanged workflow to authorized paid/local/university compute. Do not switch to another personal free account to continue the quota.

---

## 19. Final paper-writing gate

Do not write or revise numerical Results/Discussion claims until:

- all mandatory cells are complete;
- the final canonical A1/A2/A3/A4/A5/S1 decisions are documented;
- five-seed statistics and paired deltas are generated automatically;
- difficult unknown classes/datasets are retained rather than hidden;
- the fixed-data nature of E6 is stated;
- E7 communication is measured from actual transmitted student payloads;
- every table/figure has a provenance record;
- historical DQN-era numbers are clearly separated from reproduced VCT evidence.

The final experimental story should proceed in this order:

1. IID known-class sanity (E1).
2. Non-IID closed-set robustness (E3).
3. Detector mechanism in IID conditions (E2/A4).
4. Joint non-IID open-set performance (E4).
5. Unknown identity dependence (E8).
6. Dataset-wise robustness, not cross-dataset transfer (E5).
7. Teacher, anchor, transfer, detector, feature-source, and sensitivity evidence (A1-A5/S1).
8. Communication/compute efficiency (E7).
9. Fixed-data fragmentation/scalability (E6).
10. Limitations and failure cases.

---

## 20. Minimal session checklist

Use this at the start of each Colab session:

- [ ] I am using the one designated primary account.
- [ ] The runtime has an NVIDIA GPU.
- [ ] `FREEZE` is unchanged.
- [ ] Source/data SHA-256 verification passed.
- [ ] The saved `poetry.lock` was reused.
- [ ] `OUTPUT_DIR` points to the same Drive run store.
- [ ] The exact slice dry-run is correct.
- [ ] `--only-missing --resume --max-parallel 1` is active.
- [ ] W&B mode is intentional and the API key is not stored in code.
- [ ] No other workload is using the GPU for E7.

Use this at the end:

- [ ] The manifest says `COMPLETED`, or the run has a valid resumable checkpoint.
- [ ] Local metrics and logs exist.
- [ ] Drive writes have finished.
- [ ] The run ledger is updated.
- [ ] Failed/interrupted runs were not deleted or renamed.
