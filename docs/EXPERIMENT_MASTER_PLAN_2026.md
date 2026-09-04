# FedTROS-MC (VCT) — Q1 Experiment Design, Run Plan, Table/Figure Plan, and Post-Run Audit

**Target use:** 2026 Q1-journal experimental section and executable experiment workflow
**Primary evidence boundary:** Section 4 (*Experiments*) of `main(4).pdf` plus the uploaded run-command text files.
**Carried-forward implementation premise from the immediately preceding refactor:** reinforcement learning/DQN is removed and the local private teacher is a supervised **Variational Classifier Teacher (VCT)**.
**Important:** numerical values currently reported in `main(4).pdf` are treated as **historical/provisional evidence only**. They must not be presented as results of the VCT-refactored method unless reproduced after the code refactor under the protocol defined here.

---

## 1. Executive Q1 Reviewer Decision

The current experimental *scope* is strong: it already separates IID closed-set utility, IID open-set detection, non-IID closed-set robustness, non-IID open-set detection, leave-one-attack-out behavior, dataset-wise validation, communication, and scalability. That is a much better foundation than a single “accuracy on one dataset” evaluation.

However, the existing experiment package should **not** simply be rerun unchanged after the VCT refactor. The present command files are tied to the retired `dkd_fedos` / `fed_digos` / DQN-era configuration, use a single seed (`42`), include obsolete OSR flags, and do not contain a genuine external-dataset command suite. The current PDF also mixes useful results with several presentation weaknesses: single-seed evidence, last-10-round variability being used where repeated-run uncertainty is needed, an IID/client-count mismatch in the contextual comparison, a fixed-data scalability study that confounds client count with local data scarcity, and several redundant or lower-information figures.

### Final reviewer-level decision

The post-refactor experiment package should be rebuilt around five questions:

1. **Does FedTROS preserve known-class utility under IID conditions?**
2. **Does the federated training design remain robust when client label distributions are non-IID and some known classes are locally absent?**
3. **Does prototype-rank rejection actually improve unknown-attack detection over simpler matched rejection rules?**
4. **Are the VCT teacher, coverage-adaptive anchor, disagreement gate, feature alignment, boundary prototypes, and rank calibration independently necessary?**
5. **Are the gains reproducible, computationally credible, and robust across unknown classes, datasets, and federation sizes?**

The experiments below are organized to answer those questions with the minimum amount of redundant plotting and the maximum reviewer information density.

---

# 2. Critical Decision Before Any New Experiment: Does a Student OSR Branch Actually Exist?

## 2.1 Why this must be resolved before running anything

The uploaded DQN-era commands enable:

```text
training.dkd_student_osr_enabled=true
training.dkd_student_open_set_enabled=true
```

for E2/E4/E6/E8. These configuration flags **do not prove that a real trainable student OSR branch exists**. A Q1 manuscript must describe only code that is actually instantiated, optimized, checkpointed, and used at inference.

The experiment section of `main(4).pdf` only requires an **open-set rejection mechanism built from extracted student features and prototypes**. It does not require a separate trainable 8-D variational OSR subnetwork. Therefore, if the final refactored source has no genuine OSR branch, the correct action is **not to invent or restore one just to match old documentation**.

## 2.2 Source-code acceptance test for calling something an “OSR branch”

Before creating the final experiment configs, run a source audit. A component may be called a dedicated student OSR branch only if **all** of the following are true:

1. It registers trainable parameters separate from the ordinary student classifier/backbone.
2. Its forward pass produces a distinct representation or score (for example `osr_mu`, `osr_logits`, decoder output, etc.).
3. An explicit OSR loss is backpropagated into those parameters during local training.
4. Its parameters are saved in checkpoints and, if intended, aggregated by the server.
5. The final prototype/rejection code actually consumes that branch output.

If any of items 1–5 is false, do **not** describe the canonical method as having a trainable OSR branch.

## 2.3 Preferred canonical design if no branch exists

If no genuine OSR branch is found, use the cleaner model:

```text
Private VCT teacher (training only)
          ↓ KD + feature alignment
Federated student backbone/classifier
          ↓
penultimate deterministic student embedding h_S(x)
          ↓
L2 normalization
          ↓
class-wise positive prototypes + known-derived boundary prototypes
          ↓
prototype suspiciousness
          ↓
disjoint known-only empirical-rank calibration
          ↓
Known / Unknown
```

The final rejection feature is then:

\[
 z(x)=\frac{h_S(x)}{\|h_S(x)\|_2+\epsilon}.
\]

This is scientifically coherent: the global student embedding is the representation learned collaboratively; the PR module is a post-federation detector operating on that representation. No hidden auxiliary branch is required.

### Remove from final code/config/docs if no branch exists

```text
training.dkd_student_osr_enabled
training.dkd_student_open_set_enabled
student_osr decoder / reconstruction claims
osr_mu / osr_logvar claims
special osr_* aggregation unless actual osr_* tensors exist
local OSR reconstruction/KL/boundary losses that are not in executable code
```

The *open-set detector* remains. Only the nonexistent trainable *OSR branch* is removed.

## 2.4 If a real branch does exist

Do not automatically keep it. Run a controlled feature-source ablation:

- **A5-a:** PR on the ordinary penultimate student embedding `h_S`.
- **A5-b:** PR on the dedicated OSR representation.

Use identical global checkpoints, prototype-fitting data, calibration data, unknown test data, and PR hyperparameters. If the dedicated branch does not produce a reproducible improvement in AUROC/AUPRC/Unknown-F1 without harming known utility, remove it from the canonical method.

### Q1 rule

> The final paper should contain the **simplest feature path supported by reproducible evidence**.

---

# 3. Experimental Claims and Pre-Specified Hypotheses

The paper should not run a large experiment grid without defining what each experiment is meant to establish.

## H1 — Non-IID robustness

As client class distributions become more heterogeneous, FedTROS should preserve known-class **Macro-F1 and accuracy** better than architecture-matched FedAvg and FedProx students.

Primary evidence: E3.

## H2 — Value of open-set rejection

A closed-set classifier should absorb held-out attacks into known categories, whereas PR rejection should recover a meaningful fraction of unknown samples while keeping the known false-unknown rate controlled.

Primary evidence: E2, E4, E8.

## H3 — Coverage-adaptive anchoring

The adaptive anchor should provide its largest benefit under partitions with low local class coverage, particularly at severe non-IID settings.

Primary evidence: A2, plus realized client-coverage analysis in E3/E4.

## H4 — Necessity of the VCT teacher

A VCT teacher should outperform a no-teacher student and should provide a reproducible benefit beyond an otherwise matched deterministic teacher. If it does not, the variational teacher should be simplified or removed.

Primary evidence: A1.

## H5 — Necessity of disagreement gating and feature alignment

Teacher transfer should not be credited as a single monolithic component. Gated KD and feature alignment must show separable value.

Primary evidence: A3.

## H6 — Necessity of boundary prototypes and split conformal calibration calibration

The final PR module should outperform simpler same-backbone rejection methods and its improvement should be traceable to boundary information and rank calibration.

Primary evidence: A4.

## H7 — Unknown-class dependence

Open-set difficulty should vary materially with the held-out attack class; therefore one FoT-only test is insufficient to establish general OSR performance.

Primary evidence: E8.

## H8 — Fixed-data scalability effect

Under a fixed global data budget, increasing the number of clients is expected to reduce samples/class coverage per client, increasing runtime and degrading open-set performance more strongly than closed-set classification.

Primary evidence: E6. This is a **fragmentation/scaling** claim, not proof that client count alone causes degradation.

---

# 4. Protocol Lock Before Running the Main Grid

## 4.1 Freeze the repository

Create a post-refactor release candidate before expensive runs:

```text
branch: q1-vct-experiments
release tag candidate: fedtros-pr-vct-exp-freeze-rc1
```

No method-level hyperparameter may be changed after final unknown test runs begin without creating a new freeze/tag and invalidating earlier final-test claims.

## 4.2 Canonical naming

Retire all run labels containing:

```text
DKD-FedOS
dkd_fedos
Fed-DiGOS
fed_digos
PNPFF (unless separately implemented as a baseline)
FedPROTEUS
DQN
RL
```

Canonical tokens:

```text
paper method: FedTROS-MC
config token: fedtros_pr
teacher: vct
final detector: prototype_rank
```

## 4.3 Supervised local-training terminology

After RL removal, do not retain `local_episodes_per_round` merely as a renamed historical variable. Resolve what one old “episode” actually meant in optimizer steps/passes, then expose an explicit supervised quantity:

```text
training.local_epochs
```

or, if client sizes differ strongly and you need exact optimization-budget matching:

```text
training.local_optimizer_steps
```

Every baseline and ablation must use the same student optimization budget per client/round unless the difference is an explicitly studied variable.

## 4.4 Seed policy

The current command files use only `seed=42`; this is insufficient for headline Q1 claims.

### Minimum final policy

Use five predeclared seeds for all headline experiments:

```text
SEEDS = {17, 42, 73, 101, 137}
```

These seeds jointly control:

- train/validation/test split when randomized;
- client partitioning;
- model initialization;
- VCT latent sampling;
- minibatch order;
- prototype KMeans initialization;
- known-derived boundary sampling;
- calibration split.

### Preferred stronger policy

For the two most important comparisons—E3 severe non-IID (`alpha=0.1`) and E4 moderate/severe non-IID (`alpha=0.5` and `0.1`)—expand to **10 paired seeds** if computationally feasible.

## 4.5 Pairing rule

For each seed and condition, all methods must reuse the **same data split and same client partition manifest**. Do not regenerate a different Dirichlet partition independently for each method.

Example:

```text
partitions/bnat/alpha_0.1/seed_42.json
```

is reused by FedAvg-Student, FedProx-Student, FedTROS, and the relevant ablations.

## 4.6 Known-only open-set data discipline

For every open-set experiment:

```text
known training data      → model training
known prototype-fit set  → fit prototypes only
known calibration set    → empirical-rank distribution + operating threshold only
final known test set     → final utility/rejection evaluation
held-out unknown test    → final unknown evaluation only
```

No held-out unknown sample is allowed to:

- define preprocessing categories/scaler statistics;
- train VCT or student;
- select hyperparameters;
- fit prototypes;
- select prototype count;
- calibrate rank distribution;
- choose threshold;
- select the final checkpoint.

## 4.7 Disjoint prototype fitting and threshold calibration

Adopt a deterministic stratified split of the available **known-only reference pool**:

```text
70% → prototype fitting
30% → rank/threshold calibration
```

Store sample IDs/hashes in the run manifest and assert intersection size = 0.

## 4.8 Preprocessing isolation

All learned preprocessing transformations must be fit using **known training data only**. This includes numeric scalers and any learned categorical vocabulary. Unknown categories encountered at test time require a predefined unknown/OOV handling rule.

## 4.9 Test-set freeze

The final held-out unknown test partition is opened only after:

- VCT architecture is frozen;
- anchor/KD/alignment choices are frozen;
- PR detector design is frozen;
- prototype/calibration protocol is frozen;
- sensitivity analysis is completed on development data.

---

# 5. Canonical Method and Baseline Definitions

## 5.1 FedTROS-MC (canonical)

The final post-refactor system should contain only components verified in source:

```text
Private supervised VCT teacher
Federated student
Class-balanced supervised student loss
Coverage-adaptive global-student anchor
Disagreement-gated T→S KD
Teacher→student feature alignment
Support-based student aggregation
Post-federation prototype-rank rejection
```

If there is no dedicated student OSR branch, the PR detector uses the student penultimate embedding.

## 5.2 Architecture-matched closed-set baselines

### B1 — FedAvg-Student

- Identical student architecture.
- Same preprocessing.
- Same client partitions.
- Same student optimizer, batch size, local epochs/steps, rounds.
- No VCT, anchor, KD, alignment.
- Standard FedAvg aggregation.

### B2 — FedProx-Student

Same as FedAvg-Student plus the FedProx proximal objective. The proximal coefficient must be tuned only on development data and then frozen.

### B3 — SCAFFOLD-Student (strongly recommended if implementation is reliable)

Use the same student architecture and local training budget. This strengthens the non-IID baseline set by comparing against a method designed specifically for client drift.

### B4 — Local-only Student (recommended control)

No federation. Each client trains the same student locally. Report mean and worst-client Macro-F1. This answers whether collaboration is actually useful.

### B5 — Centralized Student (upper-bound/context control; optional but valuable)

Train the same student on pooled known training data. This is not a privacy-preserving competitor; it is an empirical upper/context bound showing the cost of federation and heterogeneity.

## 5.3 Matched open-set baselines

The cleanest PR attribution is obtained by applying multiple rejection rules to the **same frozen global student checkpoint**.

1. **MSP / MaxSoftmax** — reject based on low maximum known-class softmax probability.
2. **Energy score** — same frozen classifier, post-hoc energy scoring unless explicitly training an energy objective.
3. **Positive-only prototype** — nearest normalized class-prototype distance, no boundary term.
4. **Positive + boundary raw prototype score** — full geometric score, no empirical-rank transform.
5. **Full FedTROS-MC** — positive + boundary score + disjoint empirical-rank calibration.
6. **PROSER-style baseline** — only if a faithful implementation exists and uses the same known-only protocol. It should not remain hidden inside the canonical method.

## 5.4 Training-method + detector crossed comparison

For E4, do not compare only “FedTROS with PR” against closed-set FedAvg/FedProx. Use a crossed design:

```text
FedAvg-Student + PR
FedProx-Student + PR
[optional SCAFFOLD-Student + PR]
FedTROS + PR  = FedTROS-MC
```

This isolates whether the proposed federated training produces a representation that is genuinely better for open-set rejection.

---

# 6. Master Experiment Matrix

| ID | Experiment | Primary scientific question | Datasets | Distribution | Main comparators | Seeds | Main output |
|---|---|---|---|---|---|---:|---|
| E0 | Protocol/implementation verification | Is the refactored pipeline valid and leakage-free? | BNaT subset/synthetic | IID + tiny non-IID | Internal checks | 1–2 | pass/fail + manifests |
| E1 | IID closed-set utility | Does the refactored method preserve ordinary classification? | BNaT, BTAT | IID | FedAvg-Student, FedProx-Student, contextual CoL/Co-CNN values | 5 | Acc, Macro-F1, per-class F1 |
| E2 | IID open-set isolation | Does PR reject unseen attacks without non-IID confounding? | BNaT | IID, FoT unknown + optional LOO reuse | MSP, Energy, proto variants, full PR | 5 | AUROC, AUPRC, U-F1, K-FUR |
| E3 | Non-IID closed-set | Does FedTROS resist label skew/client drift? | BNaT | α={1.0,0.5,0.1} | FedAvg-S, FedProx-S, SCAFFOLD-S | 5 (10 preferred headline) | Macro-F1, Acc, worst-client F1 |
| E4 | Non-IID open-set | Does FedTROS-MC work when heterogeneity and unknowns co-occur? | BNaT | α={1.0,0.5,0.1}, FoT unknown | training×detector crossed baselines | 5 | AUROC/AUPRC/FPR@95TPR/U-F1/K-FUR |
| E5 | Dataset-wise robustness | Does behavior persist across different datasets? | BTAT, CIC-IDS2017, ToN-IoT (+BNaT reference row) | α=0.5 | FedTROS-MC + selected matched baselines | 5 | same OSR metric set |
| E6 | Fixed-data scalability | What happens as clients increase while global data are fixed? | BNaT | α=0.5, 10/50/100 clients | FedTROS-MC | 5 for final endpoints; 3 acceptable for runtime-heavy full curves | performance + runtime + client dispersion |
| E7 | Systems/communication efficiency | What is the true cost of the method? | BNaT | α=0.5 and/or 1.0 | architecture-matched baselines | 5 endpoints | bytes, time, memory, params, accuracy/MB |
| E8 | Leave-one-attack-out OSR | Is performance robust to unknown identity? | BNaT | α=0.5 canonical; optional IID supplement | BP/DoS/MitM/FoT held out one at a time | 5 | class-wise OSR matrix |
| A1 | Teacher ablation | Is VCT needed? | BNaT | α=0.1 and 0.5 | no teacher / deterministic teacher / VCT | 3→5 | paired ΔMacro-F1, ΔAUROC |
| A2 | Anchor ablation | Is coverage adaptation needed? | BNaT | α=0.1 primary, 0.5 secondary | no anchor / fixed / adaptive | 3→5 | closed+open metrics |
| A3 | Transfer ablation | Which teacher-transfer pieces matter? | BNaT | α=0.5 | no KD / ungated KD / gated KD / no alignment / full | 3→5 | performance deltas |
| A4 | PR ablation | Which rejection components matter? | BNaT | IID + α=0.5 | MSP/Energy/positive-only/raw/full-rank | 5 | OSR metric deltas |
| A5 | Feature-source gate | Is a dedicated OSR branch useful if it exists? | BNaT | α=0.5 | student embedding vs OSR branch | 3→5 | AUROC/U-F1/K-FUR |
| S1 | Sensitivity | Are main hyperparameters stable? | BNaT | α=0.5 (+0.1 for anchor) | local sensitivity grid | 3 | stability plots |

---

# 7. Detailed Experiment Protocols

## E0 — Refactor and Protocol Verification

### Purpose

Prevent expensive invalid runs.

### Mandatory tests

1. No DQN/RL modules imported by canonical method.
2. No action/reward/replay/target-Q state in resolved config.
3. VCT forward and deterministic distillation outputs are finite.
4. Student-only payload contains no private VCT tensors.
5. Preprocessing is fit on known train only.
6. Held-out unknown sample IDs are absent from all fit/calibration sets.
7. Prototype-fit and calibration sample IDs are disjoint.
8. KMeans seed is recorded.
9. Full run can be reproduced from `resolved_config.yaml`.
10. If no dedicated OSR branch exists, no stale OSR-branch flag is accepted silently.
11. Open-set score increases in the intended “more suspicious = larger score” direction.
12. Rank threshold selected on known calibration gives approximately the requested calibration K-FUR; report realized value rather than assuming exact 5%.

### Go/no-go

Do not start E1–E8 until E0 passes.

---

## E1 — IID Closed-Set Utility Validation

### Why keep this experiment

Section 4 of the current manuscript uses IID closed-set testing as a sanity check before claiming benefit under open-set/non-IID conditions. That logic is correct and should remain.

### Canonical setup

- BNaT and BTAT.
- Open-set rejection disabled.
- 100 federated rounds unless early stopping is introduced and predeclared for all methods.
- Same student architecture and student optimization budget for FedTROS, FedAvg-Student, FedProx-Student.
- Five seeds.

### BNaT contextual-reference issue

The current PDF table lists the proposed BNaT model with **3 clients**, but the uploaded E1 command uses **10 clients**. This must not remain ambiguous.

Choose one of these transparent options:

**Preferred:** canonical FedTROS experiment uses 10 clients; the literature CoL value is presented as a **contextual external result**, not a head-to-head reproduction.

**Optional reference-aligned supplement:** add a separate 3-client BNaT run (`E1B-refalign`) if direct node-count alignment is scientifically useful.

Do not relabel a 10-client run as 3 clients.

### Primary metrics

- Macro-F1 (**primary known-class metric**)
- Accuracy
- Macro-precision
- Macro-recall
- per-class F1

### Table

A compact IID utility table is sufficient. Do not overemphasize percentage-point comparison with external papers that use different architectures/protocols.

### Plot

No dedicated main-paper plot is needed unless per-class behavior is important. A per-class F1 bar/point plot can go to the supplement.

---

## E2 — IID Open-Set Recognition (Mechanism Isolation)

### Purpose

Isolate the final unknown-rejection mechanism without client heterogeneity.

### Canonical unknown

FoT may remain the first BNaT unknown scenario because that is the current manuscript protocol, but it is **not sufficient by itself**. E8 provides the required multi-unknown defense.

### Protocol

- Known labels: Normal, BP, DoS, MitM.
- Unknown: FoT.
- IID federated partition.
- Model and all detector hyperparameters frozen before final FoT test.
- Disjoint known-only prototype fit/calibration.

### Matched rejection baselines on the same checkpoint

- MSP
- Energy
- positive-only prototype
- positive+boundary raw score
- full empirical-rank PR
- PROSER only if faithfully implemented

### Primary metrics

1. AUROC (unknown = positive)
2. AUPRC (unknown = positive; always report unknown prevalence)
3. FPR@95TPR
4. Unknown F1
5. Unknown recall
6. Known false-unknown rate (K-FUR)
7. Known accuracy before rejection
8. Known accuracy after rejection

### Additional highly informative diagnostic

Report the **realized calibration K-FUR** and **final-test K-FUR** separately. A target calibration rate is not a guarantee of identical test behavior.

### Main-paper figure choice

Use a two-panel diagnostic:

- **Panel A:** empirical score/rank distributions for known vs unknown samples with the operating threshold.
- **Panel B:** precision–recall or ROC curve (prefer PR if unknown prevalence is low).

Move the second threshold-free curve to supplement if space is tight.

### Do not use as a main figure

The current simple “before vs after rejection metric bar” is largely redundant with the table and confusion matrix.

---

## E3 — Closed-Set Classification under Non-IID Client Distributions

### Conditions

```text
Dirichlet alpha = 1.0, 0.5, 0.1
```

Important: `alpha=1.0` is still a Dirichlet non-IID condition; it is not the same as the explicit IID partition.

### Methods

- FedAvg-Student
- FedProx-Student
- SCAFFOLD-Student (recommended)
- FedTROS (VCT + anchor + gated KD + alignment; PR disabled for this closed-set experiment)

### Primary metric

**Macro-F1** should become the primary headline metric because client traffic distributions exhibit severe non-IID label skew. Accuracy remains important but should not be the only metric.

### Additional metrics

- accuracy
- balanced accuracy if implemented consistently
- per-class F1
- mean client Macro-F1
- worst-client Macro-F1
- client Macro-F1 standard deviation
- realized number of locally observed classes per client

### Current Table 5 correction

Do not use “last-10-round mean ± SD” as statistical uncertainty. It measures temporal variation of one run, not run-to-run uncertainty.

Final table must report:

```text
mean ± SD across independent seeds
[optional] 95% CI
```

Per-round temporal variability can be reported separately in the supplement.

### Main figure

Plot **Macro-F1 vs alpha** for all methods with mean and 95% CI across paired seeds. Add accuracy as a secondary panel only if it tells a different story.

### Strong companion plot

Client **class-support heatmap** for each alpha, using counts or percentages, is more informative than stacked bars because it directly shows missing classes per client.

---

## E4 — Federated Open-Set Recognition under Non-IID Distributions

### Purpose

This is the central paper experiment: unknown exposure and federated heterogeneity occur simultaneously.

### Conditions

```text
alpha = 1.0, 0.5, 0.1
FoT unknown
10 clients
```

### Crossed comparator design

At minimum:

```text
FedAvg-Student + PR
FedProx-Student + PR
FedTROS + PR
```

Then, on the FedTROS global student checkpoint, compare:

```text
MSP
Energy
positive-only prototype
positive+boundary raw score
full PR rank
```

This separates **training-method quality** from **detector quality**.

### Primary metrics

- AUROC
- AUPRC
- FPR@95TPR
- Unknown F1
- Unknown recall
- K-FUR
- known accuracy after rejection
- open-set Macro-F1

### Q1 interpretation rule

Do not call `alpha=0.5` the “optimal alpha.” Alpha is an experimental heterogeneity condition, not a model hyperparameter to optimize. If 0.5 performs better than 1.0, report the non-monotonic behavior and analyze realized partition geometry.

### Required partition diagnostics

For every seed/alpha, log:

- samples/client;
- classes/client;
- class entropy/client;
- minimum/median/maximum class coverage;
- Jensen–Shannon or similar descriptive distribution divergence if already available (optional, not required for method operation).

These diagnostics help explain why two different random alpha=0.5 partitions can differ materially.

### Main figure

Use a **known–unknown operating trade-off plot** rather than multiple redundant accuracy bars:

- x-axis: K-FUR (lower better)
- y-axis: Unknown Recall or Unknown F1 (higher better)
- points/curves: method × alpha

Add a second panel with AUROC if necessary.

---

## E5 — Dataset-Wise Open-Set Robustness

### Important correction

The current manuscript eventually describes this correctly as **dataset-wise evaluation**, not cross-dataset transfer. Training and testing occur independently within each dataset.

### Datasets

- BNaT (reference row)
- BTAT
- CIC-IDS2017
- ToN-IoT

### Canonical heterogeneity

`alpha=0.5`, 10 clients.

### Unknown class protocol

The current uploaded command package does **not** contain a valid, dedicated external-dataset run suite. The file named `05_otherdataset_run_commands - Copy(1).txt` is effectively another copy of the BNaT-era command set. Therefore E5 must be rebuilt after the source refactor.

For BTAT/CIC-IDS2017/ToN-IoT, the final unknown class(es) must be frozen from the dataset manifest **before results are inspected**. Do not invent an unknown class based on which produces the best score.

Preferred hierarchy:

1. At least one predeclared held-out attack class per external dataset with sufficient support.
2. Stronger: three held-out attacks representing different frequencies/difficulties.
3. Strongest, if compute allows: leave-one-attack-out over all attack classes with adequate sample counts.

### Metrics

Same core OSR metrics as E4. Always report unknown prevalence because AUPRC is prevalence-sensitive.

### Main table

One dataset-wise robustness table:

```text
Dataset | Known classes | Unknown protocol | AUROC | AUPRC | FPR@95TPR | U-F1 | K-FUR | Known Acc After
```

### Main plot

No mandatory main plot if the table is clear. A horizontal point/range plot of AUROC and U-F1 by dataset can be supplementary.

### Interpretation requirement

Retain difficult datasets (for example the current manuscript’s weak CIC-IDS2017 unknown detection) rather than hiding them. They provide credible limitation evidence.

---

## E6 — Scalability under a Fixed Global Data Budget

### What the current experiment actually measures

With 10, 50, and 100 clients while total data remain fixed, client count is confounded with local sample size and local class coverage. Therefore the correct claim is:

> **fixed-data federation fragmentation / scalability**

not:

> “more clients inherently reduce performance.”

### Conditions

```text
clients = {10, 50, 100}
alpha = 0.5
FoT unknown
100 rounds for every final run
same total global data budget
```

### Important correction to current manuscript

The existing paper compares runs over a common 56-round horizon because some runs were incomplete. Final publication runs should use the **same predeclared full horizon** (e.g., 100 rounds) for all three sizes, or a common early-stopping rule fixed in advance.

### Evaluation frequency

Do not recalibrate the final PR detector every training round if PR is conceptually post-federation. For convergence diagnostics, use one of these clean approaches:

- evaluate the closed-set global student every round and run PR only every 5 or 10 rounds using a fixed development calibration protocol; or
- save checkpoints every 5 rounds and perform offline PR evaluation afterward.

The second approach is cleaner and reduces runtime confounding.

### Metrics

**Performance:**
- closed-set Macro-F1
- known accuracy after rejection
- open-set Macro-F1
- AUROC
- Unknown recall
- K-FUR

**Client fragmentation:**
- median samples/client
- minimum samples/client
- median classes/client
- fraction of clients missing ≥1 known class

**Systems:**
- median round time
- total wall-clock time
- client fitting time
- server aggregation time
- PR evaluation time
- peak GPU/CPU memory if measurable

**Client dispersion:**
- mean client Macro-F1
- 10th percentile client Macro-F1
- worst-client Macro-F1
- cross-client SD

Use “client dispersion/reliability,” not “fairness,” unless a formal fairness definition is introduced.

### Optional stronger scalability control

If resources permit, add **E6B fixed-local-data scaling** where samples/client are approximately held constant while the global dataset budget increases with client count. Comparing E6A and E6B separates:

- client-count/orchestration effects;
- fixed-data fragmentation effects.

This extension is highly informative but not mandatory if compute is prohibitive.

---

## E7 — Communication and Computational Efficiency

### Why the current comparison must be rebuilt

The current manuscript compares a compact communicated student against full-model baselines and reports a very large traffic factor. That is architecture-confounded. After the VCT redesign, the main communication comparison must use **architecture-matched FedAvg-Student and FedProx-Student**.

### Communication accounting

Log actual serialized/transported model tensor bytes from the federated payload:

```text
downloaded student bytes/client/round
uploaded student bytes/client/round
total participating clients/round
cumulative bidirectional parameter payload
```

State explicitly whether protocol headers, compression, TLS, and framework metadata are excluded.

### Main metrics

- transmitted parameter bytes per round
- cumulative bytes to final round
- MB to first reach target Macro-F1/accuracy
- model parameter count
- VCT private parameter count
- student parameter count
- training wall-clock
- peak memory
- inference latency/sample
- PR fit time
- PR inference overhead/sample

### Main figure

**Performance vs cumulative communication** is more informative than total MB alone.

Use:

- x-axis: cumulative bidirectional model-parameter MB (log scale acceptable)
- y-axis: validation Macro-F1 or accuracy
- methods: FedAvg-Student, FedProx-Student, FedTROS
- mean curve across seeds if feasible; otherwise endpoint CI and clearly labeled representative trajectories

### Separate compute statement

FedTROS may communicate only a student but compute a private VCT locally. Therefore communication advantage must not be presented as total resource efficiency unless local compute/runtime is also reported.

---

## E8 — Leave-One-Attack-Out Unknown Evaluation

### Why this is essential

The current experiment section already demonstrates a key fact: unknown-class detectability varies dramatically across DoS, MitM, and BP. This is scientifically more valuable than a single fixed-unknown result and should become one of the paper’s strongest robustness analyses.

### Final BNaT holdouts

Normal always remains known. Hold out each attack separately:

```text
Unknown BP   → Known {Normal, DoS, MitM, FoT}
Unknown DoS  → Known {Normal, BP, MitM, FoT}
Unknown MitM → Known {Normal, BP, DoS, FoT}
Unknown FoT  → Known {Normal, BP, DoS, MitM}
```

### Canonical distribution

Use `alpha=0.5` for the main LOO study unless a different condition is predeclared. This avoids multiplying the already expensive holdout grid by three alpha conditions. An IID LOO version can be supplementary.

### Unknown sample-count rule

For direct class-to-class comparison, cap/stratify to an equal unknown sample count when possible, and record the exact count. The current manuscript used 15,000 unknown examples in several LOO cases; the final protocol should explicitly freeze this rather than inherit it accidentally.

### Metrics

- AUROC
- AUPRC
- FPR@95TPR
- Unknown precision
- Unknown recall
- Unknown F1
- K-FUR
- known accuracy before/after rejection
- known-class per-class F1

### Additional diagnostic

For each held-out unknown class, record where missed unknowns are sent by the underlying closed-set classifier:

```text
Unknown → Normal
Unknown → BP
Unknown → DoS
Unknown → MitM
Unknown → FoT (when known)
```

This provides direct evidence for feature/representation overlap.

### Main figure

Use a **held-out-class heatmap** with rows = unknown class and columns = AUROC, AUPRC, FPR@95TPR, Unknown Recall, U-F1, K-FUR. Normalize direction visually or annotate exact values.

Move individual confusion matrices to the supplement unless one matrix illustrates a particularly important failure mode.

---

# 8. Mandatory Ablation Suite

## A1 — Teacher Design Ablation (highest priority after refactor)

Use identical student, anchor, KD schedule where applicable, client partitions, rounds, and PR detector.

### Variants

1. **No Teacher** — student + anchor only; no teacher KD/alignment.
2. **Deterministic Teacher** — supervised private classifier teacher; no stochastic latent/KL.
3. **VCT Teacher** — same teacher backbone capacity, variational latent, KL regularization.

### Interpretation gate

- VCT > deterministic teacher reproducibly → retain variational teacher.
- VCT ≈ deterministic teacher → use deterministic teacher in final method.
- teacher variants ≈ no teacher → remove teacher from canonical method.

Do not retain complexity without evidence.

---

## A2 — Coverage-Adaptive Anchor Ablation

### Variants

1. no anchor
2. fixed anchor coefficient
3. coverage-adaptive anchor

### Conditions

Primary `alpha=0.1`; secondary `alpha=0.5`.

### Required analysis

Plot per-client performance delta from adaptive vs fixed anchor against local class coverage. This directly tests the proposed mechanism rather than only reporting a global average.

---

## A3 — Teacher Transfer Ablation

Recommended variants:

1. no KD, no alignment
2. ungated T→S KD only
3. disagreement-gated T→S KD only
4. gated KD + alignment removed
5. full gated KD + alignment

If this becomes too expensive, use a factorial subset that still isolates gating and alignment.

---

## A4 — Multicenter Conformal Detector Ablation

Use the **same frozen student checkpoint** to avoid retraining confounds.

### Variants

1. MSP
2. Energy
3. positive-only normalized prototype distance
4. positive + boundary score without rank
5. positive + boundary + split conformal calibration (full PR)

Optional:

6. PROSER if implementation is faithful and independently validated

### Required conclusion

The paper should be able to state exactly whether the gain comes from:

- prototypes at all;
- boundary centers;
- empirical-rank normalization/calibration.

---

## A5 — Student Feature Source / OSR Branch Gate

Run **only if a dedicated OSR branch truly exists in code**.

Variants:

1. penultimate student embedding
2. dedicated OSR branch embedding

If no branch exists, A5 is replaced by the source-code audit decision and all branch-specific experiment flags are removed.

---

# 9. Sensitivity Analysis (S1)

Do not perform a giant hyperparameter sweep. Use targeted one-dimensional or small factorial studies tied to claimed components.

## 9.1 VCT KL weight

Include at least:

```text
beta_KL = 0
small positive
canonical
larger positive
```

`beta_KL=0` is a critical reference because it tests whether variational regularization contributes.

## 9.2 Anchor parameters

Test a small set around the canonical values for:

- base anchor coefficient;
- coverage exponent.

Primary condition: `alpha=0.1`.

## 9.3 KD temperature / weight

Use a narrow grid. Do not jointly tune many KD parameters on the final unknown test.

## 9.4 PR hyperparameters

Test only parameters that materially change geometry:

- maximum positive prototype count;
- boundary-prototype count;
- positive-radius quantile;
- boundary contribution weight;
- target known calibration false-positive rate.

Report **realized prototype counts per class**, not only requested maxima.

---

# 10. Metric Definitions and Reporting Rules

## 10.1 Closed-set metrics

Primary:

- Macro-F1

Secondary:

- accuracy
- macro-precision
- macro-recall
- per-class F1
- balanced accuracy if consistently implemented

## 10.2 Open-set metrics

Treat unknown as the positive class for detection metrics.

Primary:

- AUROC ↑
- AUPRC ↑
- FPR@95TPR ↓
- Unknown F1 ↑
- K-FUR ↓

Secondary:

- Unknown precision
- Unknown recall
- known accuracy before rejection
- known accuracy after rejection
- open-set Macro-F1
- overall open-set accuracy

## 10.3 AUPRC prevalence rule

Always record:

```text
N_known_test
N_unknown_test
unknown prevalence
```

AUPRC is prevalence-sensitive; cross-dataset AUPRC values should not be interpreted without this context.

## 10.4 Calibration reporting

For PR:

```text
target calibration K-FUR
realized calibration K-FUR
realized final-test K-FUR
selected rank threshold
```

## 10.5 Client-level metrics

For non-IID/scalability:

- mean client Macro-F1
- median client Macro-F1
- 10th percentile client Macro-F1
- worst-client Macro-F1
- cross-client SD

Do not call these “fairness” metrics unless a fairness objective/definition is formally introduced.

---

# 11. Statistical Analysis Plan

## 11.1 Headline result format

For each method/condition:

```text
mean ± SD across independent seeds
```

and, where space allows:

```text
95% confidence interval
```

## 11.2 Paired design

All method comparisons use the same seed-specific partitions. Report paired per-seed deltas:

```text
Delta = metric(FedTROS-MC) - metric(baseline)
```

## 11.3 Significance testing

### If only 5 seeds

Treat inferential testing cautiously. Report effect sizes and CIs; do not make strong “statistically significant” claims solely from a small number of stochastic runs.

### If 10 paired seeds for headline comparisons

Use a paired permutation/sign-flip test or another clearly predeclared paired nonparametric test. Apply Holm correction when testing multiple methods/conditions. Report effect size and CI in addition to p-values.

## 11.4 Multiple comparisons

Do not test every cell of every table independently. Predeclare primary comparisons:

1. FedTROS vs FedAvg-Student at `alpha=0.1` (closed-set Macro-F1).
2. FedTROS vs FedProx-Student at `alpha=0.1`.
3. FedTROS-MC vs FedAvg-Student+PR at `alpha=0.5` (AUROC/U-F1).
4. full PR vs positive+boundary raw score (AUROC/U-F1).
5. VCT vs deterministic teacher (A1).
6. adaptive vs fixed anchor (A2).

---

# 12. Q1 Main-Paper Table Plan

The current experimental section uses many useful values, but the final paper should reduce redundancy and increase causal information.

## Table 1 — Dataset and protocol summary

Columns:

```text
Dataset
Domain
Total samples used
Raw features / processed dimension
Known classes
Unknown protocol
Clients
Partition rule
Train / prototype-fit / calibration / test counts
```

For BNaT, resolve the internal label inconsistency in the current Section 4 before publication. The experiment narrative uses `Normal, BP, DoS, MitM, FoT`; the old Table 1 lists unrelated labels. The final table must be generated from the dataset manifest, not handwritten.

## Table 2 — IID closed-set utility

```text
Dataset | Method | Clients | Accuracy | Macro-P | Macro-R | Macro-F1
```

External CoL/Co-CNN values should be labeled **contextual literature values**, not direct experimental baselines.

## Table 3 — Non-IID closed-set robustness

```text
alpha | Method | Accuracy | Macro-F1 | Worst-client F1 | Mean classes/client
```

Values: mean ± SD across seeds.

This replaces the current single-seed “last 10 rounds” emphasis.

## Table 4 — Non-IID open-set performance

One compact high-value table:

```text
alpha | Training method | Detector | AUROC | AUPRC | FPR@95TPR | U-F1 | K-FUR | Known Acc After
```

If the crossed baseline grid is large, place full grid in supplement and keep the strongest baselines in main text.

## Table 5 — Leave-one-attack-out robustness

```text
Unknown | AUROC | AUPRC | FPR@95TPR | U-Precision | U-Recall | U-F1 | K-FUR | Known Acc After
```

This should remain in the main paper because unknown identity is a central OSR reviewer concern.

## Table 6 — Dataset-wise open-set robustness

```text
Dataset | Unknown protocol | AUROC | AUPRC | U-F1 | K-FUR | Known Acc After
```

## Table 7 — Component ablation

Rows:

```text
Full FedTROS-MC
No teacher
Deterministic teacher
No anchor
Fixed anchor
Ungated KD
No alignment
Positive-only PR
Raw prototype score
No rank calibration
```

Columns:

```text
Closed Macro-F1 | AUROC | U-F1 | K-FUR | Delta vs full
```

Use one representative challenging condition (`alpha=0.5` or `0.1`) and put the full ablation grid in supplement.

## Table 8 — Efficiency and scalability summary

```text
Method/clients | Student params | Private teacher params | MB/round | Total MB | Train time | PR fit time | Inference latency | Final Macro-F1/AUROC
```

Separate “communicated” from “private local” parameters.

---

# 13. Q1 Main-Paper Figure Plan

## Figure 1 — Client heterogeneity / class-support heatmap

**Keep the concept, change the visualization.**

Prefer heatmaps over stacked percentages:

- rows: clients
- columns: known classes
- cell: sample count or client-normalized proportion
- panels: IID, alpha=1.0, 0.5, 0.1

Add a small side annotation with samples/client and classes/client.

**Why:** directly supports the missing-class/coverage argument.

---

## Figure 2 — Unknown-score separation and operating point

Representative E2 run, preferably aggregated or chosen by a predeclared representative-seed rule.

Panel A: known vs unknown empirical score/rank distributions.
Panel B: PR curve (ROC can move to supplement).

**Why:** more diagnostic than a generic “before/after” bar chart.

---

## Figure 3 — Non-IID closed-set robustness

Macro-F1 vs alpha for FedAvg-Student, FedProx-Student, SCAFFOLD-Student, FedTROS.

- mean across seeds
- 95% CI ribbons/error bars

**Why:** directly visualizes robustness to heterogeneity.

---

## Figure 4 — Known/unknown operating trade-off under non-IID

Use K-FUR vs Unknown Recall/U-F1 by alpha and method, or two compact panels:

- AUROC vs alpha
- Unknown Recall vs K-FUR

**Why:** shows the rejection trade-off instead of reporting only one scalar.

---

## Figure 5 — Leave-one-attack-out heatmap

Rows: BP, DoS, MitM, FoT.
Columns: AUROC, AUPRC, FPR@95TPR, U-Recall, U-F1, K-FUR.

**Why:** visually demonstrates attack-dependent OSR difficulty.

---

## Figure 6 — Component contribution / paired-delta plot

Forest/dot plot of each ablation’s change from full FedTROS-MC across seeds.

Prefer deltas with CI over a crowded bar chart.

**Why:** answers “which component actually matters?”

---

## Figure 7 — Accuracy/Macro-F1 vs cumulative communicated MB

Architecture-matched baselines only.

**Why:** more informative than total communication alone.

---

## Figure 8 — Fixed-data scalability

Two panels:

1. open-set Macro-F1 / AUROC vs number of clients;
2. median round time vs number of clients.

Annotate median samples/client and classes/client.

**Why:** connects statistical degradation to fragmentation and system cost.

---

# 14. Figures Better Moved to Supplementary Material

1. Full ROC **and** PR curves for every condition.
2. Full before/after confusion matrices for every unknown class.
3. All per-round convergence curves.
4. PCA/UMAP latent visualizations.
5. Per-class F1 plots.
6. Client-level distribution boxplots for every alpha.
7. Full runtime decomposition by every low-level component.
8. Hyperparameter sensitivity curves.

### Specific comments on current figures

- The current ROC/PR figure is useful, but one curve can move to supplement if main-space is limited.
- The current 2-D PCA prototype plot is a **diagnostic**, not evidence that the high-dimensional detector works. Keep only if the source feature path is verified; preferably supplementary.
- The current “before vs after rejection” comparison is redundant with the open-set table and confusion matrix.
- The current alpha=1.0/0.1 convergence plots should be regenerated across seeds with CI; remove stray annotations such as `0.48`.
- The current communication plot must be regenerated using architecture-matched baselines and actual transmitted tensors.
- The current scalability moving-average curves should not be the primary evidence; report seed-level endpoints and full-horizon statistics.

---

# 15. Final Run Naming and Output Contract

## 15.1 Run ID template

```text
{experiment}_{method}_{dataset}_{partition}_{unknown}_{clients}c_seed{seed}
```

Examples:

```text
e3_fedtros_bnat_a01_closed_10c_seed42
e4_fedtros_pr_bnat_a05_fotunk_10c_seed42
e8_fedtros_pr_bnat_a05_mitmunk_10c_seed42
```

## 15.2 Required output directory

Each run must contain:

```text
resolved_config.yaml
data_manifest.json
partition_manifest.json
model_manifest.json
git_commit.txt
environment_lock.txt
seed_manifest.json
metrics_round.csv
metrics_final.json
client_metrics_final.csv
confusion_closed.csv                 # when applicable
confusion_open.csv                   # when applicable
osr_scores.parquet                   # sample id, true label, pred, score/rank, accepted/rejected
prototype_bank.npz                   # open-set runs
prototype_metadata.json
rank_calibration.json
communication_round.csv
timing_round.csv
result_manifest.json
run.log
```

## 15.3 `result_manifest.json` minimum fields

```text
run_id
experiment_id
method
method_version
commit
seed
dataset
dataset_hash
known_labels
unknown_labels
num_clients
num_rounds
local_training_budget
alpha_or_iid
prototype_feature_source
prototype_fit_sample_ids_hash
calibration_sample_ids_hash
threshold_mode
threshold_value
calibration_kfur
final_test_kfur
student_parameter_count
teacher_parameter_count
transmitted_bytes_total
final_metrics
status
```

---

# 16. Post-Refactor Target CLI Contract

The exact final Hydra key names must match the refactored repository. The commands below define the **desired semantic contract**, not permission to keep obsolete DQN-era keys.

## 16.1 Canonical method config

Target:

```text
+method=fedtros_pr
teacher.type=variational_classifier
open_set.detector=prototype_rank
open_set.feature_source=student_embedding       # preferred if no OSR branch exists
open_set.prototype.fit_ratio=0.70
open_set.calibration.ratio=0.30
open_set.calibration.target_known_fpr=0.05
dataset.preprocessing.schema_scope=known_train
```

### Keys that should disappear if the branch/RL code is removed

```text
+method=dkd_fedos
open_set.evt.backend=fed_digos
open_set.fed_digos.*
training.dkd_student_osr_enabled
training.dkd_student_open_set_enabled
training.generator.enabled
training.local_episodes_per_round
RL/DQN/replay/reward/gamma/epsilon/target-network keys
```

---

# 17. Canonical Running Plan

## Phase R0 — Validation only

Run unit/integration tests and a 2-client tiny smoke experiment.

```bash
poetry run pytest -q tests/
poetry run python run.py experiment=smoke +method=fedtros_pr seed=42 federated.num_clients=2 federated.num_rounds=2
```

Acceptance: E0 checklist passes and all mandatory manifests are generated.

---

## Phase R1 — One-seed pilot after refactor

Use seed 42 only to verify the full workflow, **not to produce final paper statistics**.

Run in this order:

1. E1 BNaT IID closed.
2. E2 BNaT IID open FoT.
3. E3 alpha=0.5 closed.
4. E4 alpha=0.5 open.
5. A1 teacher variants at alpha=0.5.
6. A4 PR variants on one frozen checkpoint.

Do not launch the complete grid until these results are numerically sane and all artifacts are traceable.

---

## Phase R2 — Architecture decision gate

### A1 first

Run no-teacher, deterministic-teacher, VCT variants.

If VCT does not justify itself, simplify **before** the expensive final grid.

### OSR branch gate

If a dedicated branch exists, run A5. If it does not win reproducibly, remove it and freeze `student_embedding` as the PR feature source.

### A2 anchor gate

Verify adaptive anchor is better than fixed/no anchor under alpha=0.1.

Only after these gates freeze the final method identity.

---

## Phase R3 — Core five-seed headline experiments

### E1

```text
2 datasets × 3 main methods × 5 seeds
```

(plus optional reference-aligned BNaT 3-client run)

### E3

```text
3 alpha values × 3–4 methods × 5 seeds
```

### E4

At minimum:

```text
3 alpha × 3 training methods with PR × 5 seeds
```

Then detector variants can reuse frozen FedTROS checkpoints, avoiding retraining.

### E8

```text
4 unknown attacks × 5 seeds
```

---

## Phase R4 — Dataset-wise validation

Rebuild E5 scripts from scratch. Do not use the current mislabeled “otherdataset” command file.

For each dataset:

1. generate data manifest;
2. freeze known/unknown protocol;
3. generate alpha=0.5 partition manifests for all seeds;
4. train;
5. fit/calibrate PR on known-only data;
6. evaluate once on final unknown test;
7. aggregate results.

---

## Phase R5 — Scalability and systems

Run E6 and E7 after the final method is frozen because they are expensive.

For E6, pre-generate 10/50/100-client partitions and save local sample/class coverage summaries before training.

For E7, compute communication from actual payloads during the same final runs; do not reconstruct byte counts from checkpoint files afterward unless used only as a validated cross-check.

---

## Phase R6 — Sensitivity and secondary ablations

Run S1 and lower-priority A3/A4 expansions after the headline grid, using 3 seeds first. Expand a sensitivity point to 5 seeds only when it affects a paper claim.

---

# 18. Bash Orchestration Templates

The following are target templates after config refactoring.

## 18.1 Common environment

```bash
set -euo pipefail
mkdir -p logs outputs manifests
SEEDS=(17 42 73 101 137)
ALPHAS=(1.0 0.5 0.1)
```

## 18.2 E3 non-IID closed-set grid

```bash
for seed in "${SEEDS[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    for method in fedavg_student fedprox_student fedtros_pr; do
      run_id="e3_${method}_bnat_a${alpha//./}_closed_10c_seed${seed}"
      poetry run python run.py experiment=e3 +method=${method} \
        seed=${seed} \
        federated.num_clients=10 \
        federated.num_rounds=100 \
        dataset.preprocessing.alpha=${alpha} \
        dataset.preprocessing.schema_scope=known_train \
        open_set.enabled=false \
        tracking.run_id=${run_id} \
        2>&1 | tee "logs/${run_id}.log"
    done
  done
done
```

Add SCAFFOLD only after its implementation passes matched-architecture tests.

## 18.3 E4 non-IID open-set grid

```bash
for seed in "${SEEDS[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    for method in fedavg_student fedprox_student fedtros_pr; do
      run_id="e4_${method}_pr_bnat_a${alpha//./}_fotunk_10c_seed${seed}"
      poetry run python run.py experiment=e4 +method=${method} \
        seed=${seed} \
        federated.num_clients=10 \
        federated.num_rounds=100 \
        dataset.preprocessing.alpha=${alpha} \
        dataset.preprocessing.schema_scope=known_train \
        'dataset.known_labels=[Normal,BP,DoS,MitM]' \
        open_set.enabled=true \
        open_set.detector=prototype_rank \
        open_set.feature_source=student_embedding \
        open_set.prototype.fit_ratio=0.70 \
        open_set.calibration.ratio=0.30 \
        open_set.calibration.target_known_fpr=0.05 \
        tracking.run_id=${run_id} \
        2>&1 | tee "logs/${run_id}.log"
    done
  done
done
```

If a real OSR branch survives A5, change `open_set.feature_source` only after the ablation justifies it.

## 18.4 E8 leave-one-attack-out

```bash
declare -A KNOWN
KNOWN[bp]='[Normal,DoS,MitM,FoT]'
KNOWN[dos]='[Normal,BP,MitM,FoT]'
KNOWN[mitm]='[Normal,BP,DoS,FoT]'
KNOWN[fot]='[Normal,BP,DoS,MitM]'

for seed in "${SEEDS[@]}"; do
  for unk in bp dos mitm fot; do
    run_id="e8_fedtros_pr_bnat_a05_${unk}unk_10c_seed${seed}"
    poetry run python run.py experiment=e8 +method=fedtros_pr \
      seed=${seed} \
      federated.num_clients=10 \
      federated.num_rounds=100 \
      dataset.preprocessing.alpha=0.5 \
      dataset.preprocessing.schema_scope=known_train \
      "dataset.known_labels=${KNOWN[$unk]}" \
      open_set.enabled=true \
      open_set.detector=prototype_rank \
      open_set.feature_source=student_embedding \
      open_set.prototype.fit_ratio=0.70 \
      open_set.calibration.ratio=0.30 \
      tracking.run_id=${run_id} \
      2>&1 | tee "logs/${run_id}.log"
  done
done
```

## 18.5 E6 fixed-data scalability

```bash
CLIENTS=(10 50 100)

for seed in "${SEEDS[@]}"; do
  for n in "${CLIENTS[@]}"; do
    run_id="e6_fedtros_pr_bnat_a05_fotunk_${n}c_seed${seed}"
    poetry run python run.py experiment=e6 +method=fedtros_pr \
      seed=${seed} \
      federated.num_clients=${n} \
      federated.num_rounds=100 \
      dataset.preprocessing.alpha=0.5 \
      dataset.preprocessing.schema_scope=known_train \
      'dataset.known_labels=[Normal,BP,DoS,MitM]' \
      open_set.enabled=true \
      open_set.detector=prototype_rank \
      open_set.evaluate_every_n_rounds=5 \
      tracking.run_id=${run_id} \
      2>&1 | tee "logs/${run_id}.log"
  done
done
```

If per-round PR recalibration is not necessary, prefer saving checkpoints every 5 rounds and performing offline PR evaluation to keep timing interpretation clean.

---

# 19. Resume and Fault-Tolerance Policy After VCT Refactor

## 19.1 Do not resume old DQN-era runs

Old `dkd_fedos_student_latest.pt` checkpoints belong to a different algorithm. They must not be continued and then reported as VCT results.

## 19.2 Student-only resume is not enough if the VCT is persistent

The old resume file assumes a student-only checkpoint is sufficient. After introducing a private VCT teacher, exact continuation may also require private per-client teacher state.

A reproducible resumable checkpoint must capture, as applicable:

```text
current global student
current round
all persistent private client VCT states
teacher optimizer/scheduler states if persistent
server state
RNG states (Python/NumPy/PyTorch/CUDA)
client partition manifest hash
selected-client schedule or RNG state
resolved config hash
```

If private teachers are intentionally reinitialized each round, document and test that behavior explicitly; otherwise save them.

## 19.3 Resume integrity test

Run a controlled experiment:

1. train rounds 1–10 uninterrupted;
2. separately train rounds 1–5, save, resume 6–10;
3. compare final parameters/metrics within deterministic tolerance.

Do not trust resume until this test passes.

---

# 20. Automated Result Aggregation

Do not manually type final table values into the manuscript.

Create one aggregator:

```text
scripts/analysis/build_q1_results.py
```

It should:

1. discover only `status=COMPLETE` runs;
2. reject runs with mismatched commit/config/data hashes;
3. group by experiment/method/condition;
4. calculate mean, SD, CI, paired deltas;
5. produce machine-readable CSV/Parquet tables;
6. generate final figures from the same aggregated data;
7. write a provenance JSON listing every run ID used in every table/figure.

Recommended outputs:

```text
paper_artifacts/table_e1_iid.csv
paper_artifacts/table_e3_noniid_closed.csv
paper_artifacts/table_e4_noniid_osr.csv
paper_artifacts/table_e5_datasets.csv
paper_artifacts/table_e8_loo.csv
paper_artifacts/table_ablation.csv
paper_artifacts/table_efficiency.csv
paper_artifacts/fig_heterogeneity.pdf
paper_artifacts/fig_osr_tradeoff.pdf
paper_artifacts/fig_loo_heatmap.pdf
paper_artifacts/fig_ablation_deltas.pdf
paper_artifacts/fig_comm_pareto.pdf
paper_artifacts/fig_scalability.pdf
paper_artifacts/provenance.json
```

---

# 21. Current Command Files: What Must Be Retired or Fixed

## 21.1 `01_dkd_fedos_run_commands(1).txt`

Retire as an executable final-paper source after refactor because it contains:

- old `dkd_fedos` identity;
- old `fed_digos` backend;
- DQN-era `local_episodes_per_round` terminology;
- obsolete OSR-branch flags;
- single-seed runs;
- generator-era flags;
- historical method outputs that are not VCT results.

Keep only as an archived historical record.

## 21.2 `02_fedavg_run_commands(1).txt` and `03_fedprox_run_commands(1).txt`

Do not reuse unchanged. Rebuild as **FedAvg-Student** and **FedProx-Student** configurations using the exact same student architecture and local optimization budget as FedTROS.

## 21.3 `04_scale_run_commands(1).txt`

Problems to fix:

- one seed only;
- old method/config names;
- PR evaluation every round may dominate timing and conflict with post-federation detector interpretation;
- final PDF uses a 56-round common horizon although command target is 100;
- no explicit fragmentation manifest.

Rebuild as E6 described above.

## 21.4 `05_otherdataset_run_commands - Copy(1).txt`

This file is not a real external-dataset command suite; it largely duplicates the BNaT-era experiment commands. Replace it entirely with one script/config per external dataset after the final dataset protocols are frozen.

## 21.5 `dkd_fedos_resume_commands.txt`

Do not use to resume VCT-era experiments. Student-only checkpoint continuation can be invalid if the private teacher state persists across rounds. Replace with full experiment-state checkpointing and a tested resume-integrity workflow.

## 21.6 `new_1(1).txt`

Archive. It contains obsolete `pnpff` run IDs even though the configured detector is `prototype_rank`, which can create provenance/naming confusion.

---

# 22. What to Remove from the Current Experimental Narrative

After the new runs, remove or rewrite the following patterns from Section 4:

1. **Single-seed superiority claims** such as “maintains approximately 95% across all settings” without repeated-run uncertainty.
2. **Last-ten-round SD as statistical uncertainty.** Keep only as temporal stability evidence.
3. **21.65× communication claim** unless regenerated from actual matched transmitted payloads.
4. **“Cross-dataset generalization”** wording for within-dataset training/testing.
5. **“Fairness”** wording for mean/worst-client F1 unless fairness is formally defined.
6. **PCA geometry as proof** of open-set separability. Use only as diagnostic visualization.
7. **Claims that alpha=0.5 is optimal.** It is an observed partition condition, not a tuned model setting.
8. **Claims that more clients inherently hurt performance.** In E6 the global data budget is fixed, so fragmentation is a confound.
9. **Any claim of an OSR branch** unless the final source audit proves the branch exists and A5 justifies retaining it.
10. **Any DQN/RL-era result as VCT evidence.** All headline results must be rerun.

---

# 23. Paper Section Mapping After Experiments

## Experimental Setup

Include:

- datasets and exact class mapping from machine-generated manifests;
- IID and Dirichlet partitioning;
- client counts;
- exact local training budget;
- seed policy;
- known-only preprocessing;
- prototype-fit/calibration/test separation;
- hardware/software environment;
- primary/secondary metrics;
- statistical protocol.

## IID Utility

Use E1 briefly as sanity validation. Do not let external contextual benchmark comparison dominate the paper.

## Non-IID Closed-Set Robustness

E3 is the primary evidence for the coverage-adaptive federated training design.

## Open-Set Recognition

E2 isolates detector behavior; E4 is the central complete scenario.

## Unknown-Class Robustness

E8 should be a dedicated subsection, not a minor appendix, because it prevents overgeneralizing from FoT.

## Dataset-Wise Robustness

E5 demonstrates that the behavior is not unique to BNaT while honestly showing difficult datasets.

## Ablation and Sensitivity

A1–A4 are mandatory for reviewer defense; A5 only if a branch exists; S1 establishes hyperparameter stability.

## Efficiency and Scalability

Use E6/E7 with corrected terminology and architecture-matched communication accounting.

---

# 24. Final Go/No-Go Criteria Before Writing New Results Text

Do **not** regenerate the paper’s Results/Discussion until all of the following are true:

- [ ] DQN/RL removed from canonical method and config.
- [ ] VCT implementation passes unit/integration tests.
- [ ] OSR branch existence resolved from source.
- [ ] If no branch exists, PR consumes verified student embedding.
- [ ] preprocessing fitted on known train only.
- [ ] prototype-fit/calibration data disjoint.
- [ ] no unknown data used during fit/tuning/calibration.
- [ ] final method/config frozen before final unknown test.
- [ ] architecture-matched FedAvg-Student and FedProx-Student available.
- [ ] at least five seeds for headline tables.
- [ ] paired partitions reused across methods.
- [ ] A1 teacher gate completed.
- [ ] A2 adaptive-anchor ablation completed.
- [ ] A4 detector ablation completed.
- [ ] E8 includes BP, DoS, MitM, FoT holdouts.
- [ ] E5 external-dataset scripts rebuilt rather than copied from BNaT.
- [ ] E6 all client counts complete the same predeclared horizon.
- [ ] communication measured from actual transmitted payloads.
- [ ] resume integrity tested.
- [ ] all tables/figures generated automatically from run manifests.
- [ ] every manuscript number points to a specific run group/provenance record.

---

# 25. Recommended Final Experiment Priority if Compute Is Limited

If resources are constrained, run in this order:

### Tier 1 — Mandatory acceptance evidence

1. E0 protocol verification
2. A1 teacher gate
3. A2 anchor ablation
4. A4 PR ablation
5. E3 five-seed non-IID closed-set
6. E4 five-seed non-IID open-set
7. E8 four unknown classes × five seeds

### Tier 2 — Strong Q1 strengthening

8. E1 IID utility on BNaT/BTAT
9. E5 BTAT/CIC-IDS2017/ToN-IoT
10. E7 communication/compute

### Tier 3 — Expensive but valuable

11. E6 10/50/100-client scalability
12. S1 extended sensitivity
13. SCAFFOLD-Student baseline
14. E6B fixed-local-data scalability

Do not sacrifice Tier 1 multi-seed/ablation evidence merely to produce many decorative datasets or plots.

---

# 26. Final Recommended Experimental Story for the Q1 Paper

The final paper should tell the experimental story in this order:

1. **Sanity:** FedTROS retains competitive known-class utility under IID data.
2. **Heterogeneity:** architecture-matched baselines degrade under client label skew; FedTROS is more stable.
3. **Mechanism:** the adaptive anchor, VCT transfer, and PR detector each have measurable roles; unnecessary components are removed.
4. **Open set:** closed-set predictions absorb unseen attacks, while PR provides an explicit known/unknown trade-off.
5. **Joint challenge:** FedTROS-MC remains effective when non-IID heterogeneity and unknown attacks occur together.
6. **Unknown identity:** performance varies substantially by held-out attack, preventing overclaiming from one FoT scenario.
7. **Dataset robustness:** behavior is tested independently on several security datasets, including difficult failure cases.
8. **Systems:** only the student is communicated, but private-teacher compute is reported honestly; communication comparisons use matched student baselines.
9. **Scaling:** fixed-data fragmentation primarily harms open-set behavior and client-tail performance as federation size increases.
10. **Limitations:** difficult unknown classes/datasets, high FPR@95TPR where score overlap remains, and scaling/runtime trade-offs are reported rather than hidden.

This is a substantially stronger and more defensible Q1 experiment design than rerunning the old DQN-era command set unchanged.

---

# 27. Immediate Next Action After This Plan

Before launching any final run, perform the post-refactor source audit and produce a short `FINAL_METHOD_MANIFEST.md` containing:

```text
student architecture and feature dimension
VCT architecture and loss
exact student local objective
exact anchor formula and active coefficients
exact KD/alignment path
exact server aggregation rule
whether a dedicated OSR branch exists: YES/NO
exact PR feature source
prototype/boundary construction
rank calibration implementation
preprocessing scope
checkpoint/resume state contents
all canonical config keys
```

Only then instantiate the run scripts above from the **actual final config schema**. This prevents another paper–code divergence.
