# Q1 Experiment And Results Demo

This document is a manuscript-style draft of the experiment and results section
for the Q1 paper, written to match the current runbook and local repository
artifacts. It is a demo draft, not final evidence.

Current local reference run:

- Run directory: `outputs/validation_minimal`
- Dataset: B-NAT
- Method: FMRL_LA
- Seed: 42
- Clients: 3
- Logical federated rounds: 1
- Flower rounds: 2
- Open-set method: EVT-based rejection

## 1. Experimental Setup

We evaluate a federated intrusion-detection framework on B-NAT under a
controlled known/unknown protocol. The known classes are `Normal`, `BP`,
`DoS`, and `MitM`. The held-out unknown attack class is `FoT`.

The implementation uses:

- `model.state_dim = 31`
- `model.num_actions = 4`
- `training.batch_size = 512` in the full protocol
- `training.local_episodes_per_round = 15`
- `training.steps_per_episode = 100`
- `federated.num_rounds = 100` for final Q1 runs
- `open_set.evt.enabled = true`

The plotting and report pipeline reads the saved artifacts from each run
directory rather than template arrays.

## 2. Dataset And Partitioning

### 2.1 B-NAT label protocol

The raw B-NAT CSV contains five labels:

| Raw label | Role in this paper |
|---|---|
| Normal | Known benign traffic |
| BP | Known attack class |
| DoS | Known attack class |
| MitM | Known attack class |
| FoT | Unknown attack / open-set holdout |

### 2.2 Preprocessing contract

The preprocessing pipeline:

- removes leakage-prone metadata and identifiers,
- fits scaling and encoding only on training data,
- reuses the same transforms for validation and test,
- writes partition manifests and tensor datasets,
- keeps unknown samples out of closed-set training.

### 2.3 Current run evidence

The reference run produced:

- `data/processed/known_train.pt`
- `data/processed/validation.pt`
- `data/processed/closed_set_test.pt`
- `data/processed/shared_closed_set_test.pt`
- `data/processed/open_set_test.pt`
- `data/processed/shared_open_set_test.pt`
- `data/processed/client_1_train.pt`
- `data/processed/client_2_train.pt`
- `data/processed/client_3_train.pt`

## 3. Methods

We compare:

- FMRL_LA
- FedAvg
- FedProx
- Centralized baseline without OSR
- Centralized baseline with OSR

The final manuscript should report paired results by seed and partition. The
main claims are:

1. Known-class classification improves or remains competitive under FL.
2. Open-set rejection reduces unsafe unknown misclassification.
3. The method is robust to client heterogeneity and seed variation.
4. The method generalizes across external datasets once the label mapping is
   finalized.

## 4. Metrics

### 4.1 Primary closed-set metrics

- Accuracy
- Balanced accuracy
- Macro precision
- Macro recall
- Macro F1

### 4.2 Primary open-set metrics

- AUROC
- AUPRC
- FPR@95%TPR
- Unknown-detection F1
- Unknown recall

### 4.3 Systems metrics

- Final accuracy vs. client count
- Cumulative transmitted MB
- Accuracy per MB
- Seed-wise variance

## 5. Current Reference Results

The following values come from `outputs/validation_minimal/evaluation_metrics.json`
and `outputs/validation_minimal/test_metrics.json`.

### 5.1 Closed-set results

| Metric | Value |
|---|---:|
| Test accuracy | 0.3901 |
| Balanced accuracy | 0.2810 |
| Macro precision | 0.2350 |
| Macro recall | 0.2810 |
| Macro F1 | 0.1947 |
| Test loss | 1.3693 |

Per-class accuracy:

| Class | Accuracy |
|---|---:|
| Normal | 0.4386 |
| BP | 0.6760 |
| DoS | 0.0000 |
| MitM | 0.0093 |

### 5.2 Open-set results

| Metric | Value |
|---|---:|
| AUROC | 0.7937 |
| AUPRC | 0.4584 |
| FPR@95%TPR | 0.3473 |
| Unknown detection rate | 0.9538 |
| Unknown F1 | 0.6589 |
| Open-set global delta | 0.7178 |
| Known accuracy after OSR | 0.3121 |
| Overall accuracy after OSR | 0.4903 |

Interpretation for the paper:

- The open-set module detects unknown attacks well at the chosen operating
  point.
- Closed-set class imbalance remains visible in the per-class scores, so macro
  metrics are the correct primary reporting choice.
- The current demo run is only a one-logical-round validation, so it should be
  described as a pipeline check rather than final Q1 evidence.

## 6. Figures And Data Sources

The repository renderer produces 14 Q1 plots. The table below maps each plot to
its data file and publication role.

| # | Plot | Data file | Paper role |
|---:|---|---|---|
| 1 | Scalability: Nodes vs Final Accuracy | `scalability.csv` | Client-count sensitivity |
| 2 | Non-IID Data Distribution | `client_class_distribution.csv` | Partition explanation |
| 3 | Convergence and Variance: Mild Non-IID | `comparison_metrics.csv` | Mild heterogeneity comparison |
| 4 | Convergence and Variance: Hard Non-IID | `comparison_metrics.csv` | Hard heterogeneity comparison |
| 5 | Known vs Unknown EVT Score Distribution | `open_set_scores.csv` | Open-set separation |
| 6 | Openness vs AUROC Performance | `openness_metrics.csv` | Open-set sensitivity |
| 7 | ROC Curve for Unknown Zero-Day Attacks | `open_set_roc_curve.csv` | Detection trade-off |
| 8 | Cross-Dataset Generalization | `cross_dataset_metrics.csv` | External validity |
| 9 | Before OSR Confusion Matrix | `before_osr_confusion_matrix.csv` | Baseline failure mode |
| 10 | After OSR Confusion Matrix | `after_osr_confusion_matrix.csv` | OSR improvement |
| 11 | Seed Robustness Boxplot | `seed_robustness.csv` | Reproducibility |
| 12 | t-SNE/UMAP Latent Space Separation | `latent_embeddings.csv` | Representation quality |
| 13 | Communication Efficiency | `communication_metrics.csv` | Deployability |
| 14 | Architectural Ablation | `ablation_metrics.csv` | Causal contribution |

### 6.1 Current local plot manifest

`outputs/validation_minimal/plots/plot_manifest.json` confirms that the run
produced the 14 required PNG/PDF pairs.

## 7. Manuscript-Ready Claims

The Q1 paper can safely claim the following only after the full suite is
re-run:

- FMRL_LA is evaluated against FedAvg and FedProx under matched partitions.
- Closed-set and open-set metrics are reported separately.
- Unknown attacks are rejected using EVT-calibrated thresholds.
- Client heterogeneity and seed variation are treated as first-class factors.
- The paper uses saved artifacts, not synthetic template arrays, for every
  reported figure and table.

## 8. Draft Results Paragraph

On the current validation run, the proposed system achieved 39.01% test
accuracy, 28.10% balanced accuracy, and 19.47% macro F1 on the known classes.
Open-set evaluation reported 0.7937 AUROC, 0.4584 AUPRC, 0.3473 FPR@95%TPR,
and 0.6589 unknown-detection F1. These numbers show that the open-set
rejection path is active and that the report pipeline is reading real outputs,
but they are not yet the final paper numbers because the run used only one
logical federated round.

## 9. What Still Needs Final Runs

- Full 100-round FMRL_LA / FedAvg / FedProx comparisons.
- The scalability CSV with client counts 3, 10, 20, 50, and 100.
- Seed robustness CSV with at least five seeds, ideally 10-15.
- External dataset configs and aggregated metrics for B-TAT, ToN-IoT, and
  CIC-IDS2017.
- Communication-efficiency CSV derived from actual bytes transmitted.
- Latent-embedding export from real model states if Plot 12 is to be final.

## 10. Artifact Index

- `outputs/validation_minimal/run.log`
- `outputs/validation_minimal/debug.log`
- `outputs/validation_minimal/metrics.csv`
- `outputs/validation_minimal/metrics.jsonl`
- `outputs/validation_minimal/evaluation_metrics.json`
- `outputs/validation_minimal/test_metrics.json`
- `outputs/validation_minimal/open_set_metrics.json`
- `outputs/validation_minimal/open_set_scores.csv`
- `outputs/validation_minimal/open_set_roc_curve.csv`
- `outputs/validation_minimal/open_set_pr_curve.csv`
- `outputs/validation_minimal/before_osr_confusion_matrix.csv`
- `outputs/validation_minimal/after_osr_confusion_matrix.csv`
- `outputs/validation_minimal/plots/plot_manifest.json`
