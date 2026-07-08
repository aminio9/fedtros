# FedPROTEUS

**Federated Prototype-Ranked Teacher–Student Learning for Unknown Attack Detection in Blockchain Traffic**

FedPROTEUS is a research framework for **federated open-set intrusion detection** in blockchain traffic. The method combines a **private reinforcement-learning teacher** on each client with a **shared federated student model**, then performs unknown attack detection through **prototype-rank scoring** in the global student latent space.

The goal is to detect both known blockchain traffic classes and previously unseen/unknown attacks under realistic federated conditions, including IID and highly non-IID client distributions.

---

## Highlights

- **Federated intrusion detection** for blockchain traffic and transaction/network traces.
- **Open-set recognition** for unknown attack detection.
- **Private RL teacher per client** for local specialization.
- **Shared student model** for federated aggregation.
- **Teacher–student knowledge distillation** for transferring local teacher knowledge into the student.
- **Prototype-ranked open-set detector** using the global student latent representation.
- **Strict raw-data-free federated setting**: raw traffic, labels, and private teacher parameters remain local.
- **IID and non-IID experiments**, including Dirichlet splits.
- **Round-wise evaluation**, checkpointing, resume support, open-set metrics, and publication-ready plots.

---

## Method Name

**FedPROTEUS** stands for:

```text
Fed      → Federated learning
PRO      → Prototype-ranked open-set scoring
TE       → Teacher–student learning
US       → Unknown attack/security detection
```

Full title:

```text
FedPROTEUS: Federated Prototype-Ranked Teacher–Student Learning for Unknown Attack Detection in Blockchain Traffic
```

---

## Method Overview

FedPROTEUS separates **local specialization** from **federated generalization**.

Each client maintains a private RL-based teacher trained only on its local blockchain traffic. This teacher is not uploaded to the server. Instead, the teacher guides a compact student model through knowledge distillation and feature alignment. The student is the only model exchanged and aggregated across clients.

After federated training, the global student becomes the shared representation backbone. Unknown attacks are detected using prototype-rank scoring in the global student latent space.

```text
Client i
 ├── Local blockchain traffic
 ├── Private RL teacher T_i
 │    └── learns local traffic behavior
 ├── Shared student S_i
 │    ├── learns from local labels
 │    ├── learns from teacher logits
 │    ├── aligns with teacher features
 │    └── stays close to global anchor
 └── Uploads only student update

Server
 ├── receives student updates
 ├── aggregates global student
 └── evaluates open-set detection using prototype-rank scoring
```

---

## Core Idea

For each communication round:

1. The server broadcasts the current global student.
2. Each client trains a private RL teacher locally.
3. The local student learns from:
   - local supervised labels,
   - teacher logits,
   - teacher features,
   - frozen global-student anchor.
4. The client uploads only the student parameters.
5. The server aggregates student updates.
6. Open-set detection is performed using prototype-ranked scoring in the student latent space.

The final detector is **not reconstruction-based**. It is based on:

```text
global student latent features + positive prototypes + boundary prototypes + prototype-rank score
```

---

## Architecture

### Private RL Teacher

The teacher is a local CVAE-DQN/RL-style model. It learns client-specific traffic behavior and produces teacher logits and latent features.

Teacher components include:

```text
prior_net
recognition_net
value_net_main
value_net_target
```

The teacher remains local and is not uploaded to the server.

### Shared Student

The student is a compact MLP-based IDS classifier. It is the federated model exchanged between clients and the server.

The student produces:

```text
student_features, student_logits = student_model(x)
```

The student feature vector is later used for prototype-based open-set detection.

### Prototype-Ranked Open-Set Detector

After federated training, FedPROTEUS builds prototypes in the global student latent space:

```text
positive prototypes  → known traffic regions
boundary prototypes  → open-space / unknown regions
```

A test sample is rejected as unknown if its prototype-rank score exceeds a calibrated threshold.

---

## Learning Objective

The local student is trained using a combined objective:

```text
L_student =
    L_task
  + λ_anchor L_anchor
  + λ_KD L_teacher_to_student
  + λ_align L_feature_alignment
```

Where:

- `L_task`: supervised local classification loss.
- `L_anchor`: keeps the local student close to the received global student.
- `L_teacher_to_student`: distills teacher logits into the student.
- `L_feature_alignment`: aligns projected teacher features with student features.

This design is especially important under non-IID data, where a client may not contain all known traffic classes.

---

## Strict Federated Setting

FedPROTEUS uses a strict raw-data-free federated setup.

The server receives:

```text
student model updates
number of local examples
label-free diagnostics
```
---


## Installation

### 1. Create Poetry environment

For Flower simulation stability, Python 3.11 is recommended.

```bash
cd ~/cf_marlos

poetry env remove --all || true
poetry env use /usr/bin/python3.12

poetry lock
poetry install --with dev
```
---

## Experiments

The project supports closed-set, open-set, IID, non-IID, and label-wise unknown experiments.

### Experiment 1: Closed-set IID

```bash
poetry run python run.py experiment=exp1 +method=dkd_fedos seed=42 \
  tracking.run_id=e1_iid_closed_dkd_fedos_seed42
```

### Experiment 2: Open-set IID, FoT unknown

Known labels exclude the unknown class.

```bash
poetry run python run.py experiment=exp2 +method=dkd_fedos seed=42 \
  dataset.known_labels=[Normal,BP,DoS,MitM] \
  training.generator.enabled=false \
  training.dkd_student_osr_enabled=true \
  training.dkd_student_open_set_enabled=true \
  open_set.evt.backend=fed_digos \
  open_set.fed_digos.enabled=true \
  open_set.fed_digos.score_fusion.method=prototype_rank \
  tracking.run_id=e2_iid_openset_fot_dkd_fedos_seed42
```

### Experiment 3: Closed-set non-IID

```bash
poetry run python run.py experiment=exp3 +method=dkd_fedos seed=42 \
  dataset.preprocessing.iid=false \
  dataset.preprocessing.dirichlet_alpha=0.1 \
  tracking.run_id=e3_noniid_alpha01_closed_dkd_fedos_seed42
```

### Experiment 4: Open-set non-IID, FoT unknown

```bash
poetry run python run.py experiment=exp4 +method=dkd_fedos seed=42 \
  dataset.known_labels=[Normal,BP,DoS,MitM] \
  dataset.preprocessing.iid=false \
  dataset.preprocessing.dirichlet_alpha=0.1 \
  training.generator.enabled=false \
  training.dkd_student_osr_enabled=true \
  training.dkd_student_open_set_enabled=true \
  open_set.evt.backend=fed_digos \
  open_set.fed_digos.enabled=true \
  open_set.fed_digos.score_fusion.method=prototype_rank \
  tracking.run_id=e4_noniid_alpha01_openset_fot_dkd_fedos_seed42
```

### Experiment 8: Label-wise open-set unknown detection

Example: MitM unknown.

```bash
poetry run python run.py experiment=exp8 +method=dkd_fedos seed=42 \
  dataset.known_labels=[Normal,BP,DoS,FoT] \
  training.generator.enabled=false \
  training.dkd_student_osr_enabled=true \
  training.dkd_student_open_set_enabled=true \
  open_set.evt.backend=fed_digos \
  open_set.fed_digos.enabled=true \
  open_set.fed_digos.score_fusion.method=prototype_rank \
  tracking.run_id=e8_mitm_unknown_dkd_fedos_seed42
```

---

## Resume Interrupted Training

Student checkpoints are saved as:

```text
dkd_fedos_student_round_XXXX.pt
dkd_fedos_student_latest.pt
```

To resume, use:

```text
federated.resume_from=<path_to_latest_checkpoint>
federated.resume_round_offset=<last_completed_round>
federated.num_rounds=<remaining_rounds>
```

Example: resume Exp8 MitM from round 52 to 100.

```bash
poetry run python run.py experiment=exp8 +method=dkd_fedos seed=42 \
  dataset.known_labels=[Normal,BP,DoS,FoT] \
  federated.resume_from=outputs/e8_mitm_unknown_dkd_fedos_seed42/dkd_fedos_student_latest.pt \
  federated.resume_round_offset=52 \
  federated.num_rounds=48 \
  training.generator.enabled=false \
  training.dkd_student_osr_enabled=true \
  training.dkd_student_open_set_enabled=true \
  open_set.evt.backend=fed_digos \
  open_set.fed_digos.enabled=true \
  open_set.fed_digos.score_fusion.method=prototype_rank \
  tracking.run_id=e8_mitm_unknown_resume_from52_seed42 \
  2>&1 | tee e8_mitm_unknown_resume_from52_seed42.log
```

---

## Output Artifacts

Typical output folder:

```text
outputs/<run_id>/
├── config.yaml
├── resolved_config.yaml
├── run.log
├── debug.log
├── metadata.json
├── dkd_fedos_monitoring.jsonl
├── dkd_fedos_student_latest.pt
├── dkd_fedos_student_round_XXXX.pt
├── open_set_round_metrics.csv
├── open_set_rounds/
├── evt/
├── processed/
└── plots/
```

Important open-set artifacts:

```text
open_set_scores.csv
open_set_metrics.json
fed_digos_prototypes.json
fed_digos_rank_calibration.json
known_unknown_score_quantiles.json
score_overlap_report.json
latent_embeddings.csv
```

---

## Evaluation Metrics

Closed-set metrics:

```text
accuracy
macro precision
macro recall
macro F1
weighted F1
classification report
confusion matrix
```

Open-set metrics:

```text
AUROC
AUPRC
unknown precision
unknown recall
unknown F1
known retention
false unknown rate
score distribution
before/after open-set confusion matrix
```

Federated metrics:

```text
round-wise global accuracy
round-wise macro F1
client pre-aggregation metrics
client post-aggregation metrics
aggregation reliability weight
communication rounds
scalability across number of clients
```

---

## Plotting

Use the updated Pastel Sunset plotting script:

```bash
python scripts/updated_testplot_3alg_theme.py \
  --run-dir outputs/<run_id> \
  --output-dir outputs/<run_id>/plots/updated_testplot_3alg
```

The final comparison plots should include only:

```text
Proposed
FedAvg
FedProx
```

---

## Citation

If you use this repository, please cite the corresponding paper:

```bibtex
@article{fedproteus2026,
  title   = {FedPROTEUS: Federated Prototype-Ranked Teacher--Student Learning for Unknown Attack Detection in Blockchain Traffic},
  author  = {Amini, Mohammad and collaborators},
  journal = {To be added},
  year    = {2026}
}
```

---

## License

Add your license here.

Recommended for research code:

```text
MIT License
```

or, if dataset/license restrictions require it:

```text
Apache License 2.0
```
