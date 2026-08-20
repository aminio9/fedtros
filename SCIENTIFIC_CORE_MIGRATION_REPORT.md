# Scientific Core Migration & Architecture Report (Workstream A)

**Document:** `SCIENTIFIC_CORE_MIGRATION_REPORT.md`
**Protocol:** FedTROS-PR Canonical VCT Architecture
**Status:** COMPLETE (115/115 Unit & Integration Tests Passing)
**Date:** 2026-08-19

---

## 1. Executive Summary
This report summarizes the complete scientific core migration of the project from the legacy DQN/reinforcement learning stack into the publication-ready **FedTROS-PR** algorithm ("Federated Teacher-Regularized Open-Set Recognition with Prototype-Rank Rejection"), matching the 5 canonical project reference specifications.

---

## 2. Scientific Core Architecture

### 2.1 Private Variational Classifier Teacher (VCT)
- **Module:** [`src/models/variational_teacher.py`](file:///d:/Research/Code/fedtros/src/models/variational_teacher.py), [`src/models/variational_classifier_teacher.py`](file:///d:/Research/Code/fedtros/src/models/variational_classifier_teacher.py)
- **Mathematical Formulation:** Supervised Variational Information Bottleneck (VIB):
  $$q_\phi(z|x) = \mathcal{N}(\mu_T(x), \text{diag}(\sigma_T^2(x)))$$
- **Objective Function:**
  $$L_T = L_{\text{CBCE}}^T(\hat{y}_T, y) + \beta_T \cdot D_{\text{KL}}[q_\phi(z|x) \parallel \mathcal{N}(0, I)]$$
- **Dual Mode Operation:**
  - *Teacher Training:* Stochastic reparameterization $z = \mu_T + \sigma_T \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$.
  - *Distillation & Evaluation:* Strictly deterministic inference $\hat{y}_T = W_T \mu_T + b_T$ without sampling noise.
- **Privacy Guarantee:** Private to each client; 0 teacher weights transmitted; 0 gradients from student backpropagated to teacher.

### 2.2 Shared Student IDS Model
- **Module:** [`src/models/student.py`](file:///d:/Research/Code/fedtros/src/models/student.py)
- **Model Payload:** Compact backbone $[512, 256, 128]$ with classification head and optional 8-D OSR branch.
- **Transmission:** Only student model parameters are transmitted to/from the federated server.

### 2.3 Canonical Student Loss Formulation
- **Module:** [`src/models/bundle.py`](file:///d:/Research/Code/fedtros/src/models/bundle.py#L280-L400)
- **Multi-Task Objective:**
  $$L_S = L_{\text{CBCE}} + \lambda_{a, i} L_{\text{anchor}} + \lambda_{\text{KD}} L_{T \to S} + \lambda_{\text{align}} L_{\text{align}}$$
- **Components:**
  1. **$L_{\text{CBCE}}$:** Class-Balanced Cross-Entropy using Effective Number of Samples weights $w_c = (1 - \beta) / (1 - \beta^{N_c})$.
  2. **$L_{\text{anchor}}$:** Coverage-adaptive global anchor regularization:
     $$\lambda_{a, i} = \lambda_{\text{base}} \max\left(\lambda_{\text{min}}, (1 - q_i)^p\right), \quad q_i = \frac{|\mathcal{C}_i|}{C_{\text{known}}}$$
     Anchors student predictions against the frozen received global student $\hat{y}_{\text{anchor}}$.
  3. **$L_{T \to S}$:** Disagreement-gated directional Knowledge Distillation from VCT to Student.
  4. **$L_{\text{align}}$:** Feature alignment between detached VCT mean $\mu_T$ and student penultimate feature $h_S$ via projection layer $A_\phi(\mu_T)$.

---

## 3. Dedicated OSR Branch & Feature Source Audit

### 3.1 OSR Branch Audit Verdict
- **Verdict:** `DEDICATED_OSR_BRANCH = YES` (Detailed audit in [`docs/OSR_BRANCH_AUDIT.md`](file:///d:/Research/Code/fedtros/docs/OSR_BRANCH_AUDIT.md)).
- **Dual Feature Source Interface:**
  1. `student_embedding`: Normalized penultimate feature vector $z = h_S / (\|h_S\|_2 + \epsilon) \in \mathbb{R}^{128}$.
  2. `osr_embedding`: Compact dedicated latent branch $\mu_{\text{OSR}} \in \mathbb{R}^8$.
- Both sources are cleanly exposed and selectable via configuration.

---

## 4. Prototype-Rank Rejection (PR) Implementation

### 4.1 Modularized Architecture
- **Package:** `src/openset/`
  - [`prototype_bank.py`](file:///d:/Research/Code/fedtros/src/openset/prototype_bank.py): Positive class KMeans prototypes ($K \le 16$) and known-derived synthetic boundary prototypes ($K \le 32$) using manifold mixup.
  - [`rank_calibration.py`](file:///d:/Research/Code/fedtros/src/openset/rank_calibration.py): Stratified disjoint 70/30 calibration split with SHA-256 provenance hashes and empirical CDF rank mapping.
  - [`prototype_rank.py`](file:///d:/Research/Code/fedtros/src/openset/prototype_rank.py): Unified Prototype-Rank rejection inference, empirical threshold fitting, and open-set metric evaluation (AUROC, AUPRC, FPR95, Closed-Set Retention, Unknown Recall).

### 4.2 Stratified Disjoint 70/30 Protocol
- **70% Subsample:** Fits class prototype centers and boundary prototypes.
- **30% Subsample:** Fits empirical CDF rank distributions and selects rejection threshold $T$ at target $\text{FPR} = 0.05$.
- **Provenance & Disjointness:** Automated SHA-256 hash generation and strict assertion that $\text{Fit} \cap \text{Calibration} = \emptyset$.

---

## 5. Complete Elimination of Legacy RL & Generative Subsystems

### 5.1 Removed RL Elements
- Removed `src/agents/`, `src/configs/agent/`, `src/rl/`.
- Deleted legacy parameters: `double_q`, `reward_curve`, `epsilon_schedule`, `local_episodes_per_round`, `steps_per_episode`, `min_buffer_size`, `reward_mode`, `latent_q`, `gamma`.
- Standardized terminology to supervised distillation parameters: `local_epochs`, `student_epochs`, `teacher_epochs`, `batch_size`, `student_lr`, `teacher_lr`.

### 5.2 Removed Teacher CVAE / Generative Leftovers
- Removed teacher decoders, recognition/generation networks, and generative reconstruction losses from teacher.
- Teacher is purely a Variational Classifier with an Information Bottleneck.

---

## 6. Preprocessing & Data Isolation
- Enforced strict fitting of categorical encoders, imputers, and scalers on `known_train` only.
- Added explicit out-of-vocabulary (`__UNK__`) fallback handling.
- Validated taxonomy for BNaT benchmark dataset: `Normal`, `BP`, `DoS`, `MitM` (known) and `FoT` (open-set unknown).

---

## 7. Verification & Test Suite Status

### 7.1 Automated Scientific Tests
All tests in [`tests/test_scientific_core.py`](file:///d:/Research/Code/fedtros/tests/test_scientific_core.py) and across the repository pass with **100% success**:

| Test Group | Test Name | Status |
|---|---|---|
| **VCT** | `test_teacher_output_shapes` | PASS |
| **VCT** | `test_teacher_kl_nonnegative` | PASS |
| **VCT** | `test_teacher_stochastic_training` | PASS |
| **VCT** | `test_teacher_deterministic_distillation` | PASS |
| **VCT** | `test_teacher_gradients` | PASS |
| **RL Guard** | `test_no_rl_parameters` | PASS |
| **Gradient Isolation** | `test_student_does_not_backprop_teacher` | PASS |
| **Alignment** | `test_alignment_dimensions` | PASS |
| **Communication** | `test_federated_payload_student_only` | PASS |
| **Preprocessing** | `test_known_only_preprocessing` | PASS |
| **PR Protocol** | `test_pr_disjoint_split` | PASS |
| **PR Formula** | `test_pr_rank_formula` | PASS |
| **PR Metrics** | `test_e2_e4_prototype_rank` | PASS |

### 7.2 Full Test Suite
- Total Test Cases: **115 passed, 0 failed, 2 warnings** in 56.85s.
