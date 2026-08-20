# Plot Project Input Data Contract (Workstream C4)
============================================================

**Document Version:** 1.0 (Read-Only Specification)
**Target Repository:** `plots` (`d:/Research/Code/plots`)
**Authority:** The `plots` repository is the single source of truth for rendering **all 29 publication figures** (600 DPI, B&W + color print dual-encoding) and publication LaTeX tables.

---

## 1. Directory & File Structure Expected by `plots/`

```
plots/
├── data/
│   ├── metadata/
│   │   └── result.json                           # Central structured results & figure registry
│   └── processed/
│       ├── client_distribution_alpha01.csv       # Figure 02
│       ├── client_distribution_iid_alpha_sweep.csv # Figure 02b
│       ├── convergence_alpha1.csv                # Figure 03, 25
│       ├── convergence_alpha01.csv               # Figure 04, 25
│       ├── exp2_scores.csv                       # Figures 05, 07, 08, 23, 24
│       ├── exp2_roc_curve.csv                    # Figure 07 (precalculated or derived)
│       ├── exp2_pr_curve.csv                     # Figure 08 (precalculated or derived)
│       ├── exp2_confusion_before.csv             # Figures 09, 09c, 09p, 24, 26
│       ├── exp2_confusion_after.csv              # Figures 10, 09c, 09p, 24, 26
│       ├── exp2_latent_projection.csv            # Figure 12
│       ├── communication_alpha1.csv              # Figure 13
│       ├── communication_alpha01_fedavg.csv      # Figure 13
│       ├── scalability_10_clients.csv            # Figures 01, 16–22
│       ├── scalability_50_clients.csv            # Figures 01, 16–22
│       ├── scalability_100_clients.csv           # Figures 01, 16–22
│       └── provenance_manifest.json              # Provenance tracing
```

---

## 2. Complete Figure-by-Figure Input Mapping (29 Figures)

### Base Figures (01–15)

| Figure Stem | Output Filename | Primary Source | Required Columns / Fields | Units / Value Range |
|---|---|---|---|---|
| `01_scalability_clients` | `01_scalability_clients.png` | `result.json:plot_01_scalability` or `scalability_*.csv` | `clients`, `overall_accuracy`, `known_accuracy`, `unknown_f1`, `auroc` | Percent [0, 100] |
| `02_internal_non_iid_distribution` | `02_internal_non_iid_distribution.png` | `client_distribution_alpha01.csv` | `client_id`, `Normal`, `BP`, `DoS`, `MitM`, `FoT` | Sample counts (int) |
| `02_internal_non_iid_distribution_2x2` | `02_internal_non_iid_distribution_2x2.png` | `client_distribution_iid_alpha_sweep.csv` | `regime`, `client_id`, `Normal`, `BP`, `DoS`, `MitM`, `FoT` | Sample counts (int) |
| `03_convergence_mild_non_iid` | `03_convergence_mild_non_iid.png` | `convergence_alpha1.csv` | `round`, `method`, `macro_f1_percent`, `band` | `macro_f1_percent` (%) |
| `04_convergence_hard_non_iid` | `04_convergence_hard_non_iid.png` | `convergence_alpha01.csv` | `round`, `method`, `macro_f1_percent`, `band` | `macro_f1_percent` (%) |
| `05_prototype_rank_score_distribution` | `05_prototype_rank_score_distribution.png` | `exp2_scores.csv` | `prototype_rank_score`, `known_or_unknown` | Score [0.0, 1.0], label: "known"/"unknown" |
| `06_openness_vs_auroc` | `06_openness_vs_auroc.png` | `result.json:plot_06_openness` | `openness`, `auroc`, `method` | Openness [0.0, 1.0], AUROC (%) |
| `07_roc_unknown_detection` | `07_roc_unknown_detection.png` | `exp2_roc_curve.csv` or `exp2_scores.csv` | `fpr`, `tpr`, `method` | Coordinates [0.0, 1.0] |
| `08_pr_unknown_detection` | `08_pr_unknown_detection.png` | `exp2_pr_curve.csv` or `exp2_scores.csv` | `recall`, `precision`, `method` | Coordinates [0.0, 1.0] |
| `09_confusion_before_open_set` | `09_confusion_before_open_set.png` | `exp2_confusion_before.csv` | Index & columns: `Normal`, `BP`, `DoS`, `MitM`, `Unknown` | Sample counts (int) |
| `10_confusion_after_open_set` | `10_confusion_after_open_set.png` | `exp2_confusion_after.csv` | Index & columns: `Normal`, `BP`, `DoS`, `MitM`, `Unknown` | Sample counts (int) |
| `09_10_confusion_comparison` | `09_10_confusion_comparison.png` | `exp2_confusion_before.csv`, `exp2_confusion_after.csv` | Same as Fig 09 & 10 | 2-panel comparison |
| `09_10_confusion_comparison_percentage` | `09_10_confusion_comparison_percentage.png` | `exp2_confusion_before.csv`, `exp2_confusion_after.csv` | Same as Fig 09 & 10 | Row-normalized percentages [0, 100] |
| `11_non_iid_robustness_boxplot` | `11_non_iid_robustness_boxplot.png` | `result.json:plot_11_robustness` | `heterogeneity` ($\alpha$), `accuracy`, `method` | Percent [0, 100] |
| `12_latent_prototype_geometry` | `12_latent_prototype_geometry.png` | `exp2_latent_projection.csv` | `x`, `y`, `label`, `is_prototype` | PCA 2D coordinates (float) |
| `13_communication_efficiency` | `13_communication_efficiency.png` | `communication_alpha1.csv` | `round`, `method`, `cumulative_mb`, `val_acc` | Megabytes, Accuracy (%) |
| `14_ablation_study` | `14_ablation_study.png` | `result.json:plot_14_ablation` | `module`, `macro_f1`, `delta` | Macro-F1 (%), Delta (%) |
| `15_method_comparison` | `15_method_comparison.png` | `result.json:plot_15_method_comparison` | `method`, `overall_accuracy`, `macro_f1`, `auroc`, `unknown_f1` | Percent [0, 100] |

---

### Scalability Figures (16–20)

Source: `scalability_10_clients.csv`, `scalability_50_clients.csv`, `scalability_100_clients.csv`.

| Figure Stem | Output Filename | Key Fields Required | Notes |
|---|---|---|---|
| `16_scalability_convergence_common_rounds` | `16_scalability_convergence_common_rounds.png` | `round`, `round_openset_f1_macro` | Truncated to minimum common round (round 56) |
| `17_scalability_quality_vs_wallclock` | `17_scalability_quality_vs_wallclock.png` | `cumulative_hours`, `round_openset_f1_macro` | `cumulative_hours = cumsum(round_time_sec) / 3600` |
| `18_scalability_runtime_breakdown` | `18_scalability_runtime_breakdown.png` | `client_fit_wall_time_sec`, `open_set_round_eval_time_sec`, `server_aggregation_time_sec`, `round_time_sec` | Tail 10 rounds median decomposition |
| `19_scalability_client_fairness` | `19_scalability_client_fairness.png` | `mean_client_macro_f1`, `worst_client_macro_f1`, `std_client_macro_f1` | Error bar plot across 10, 50, 100 clients |
| `20_scalability_open_set_detection` | `20_scalability_open_set_detection.png` | `round_openset_auroc`, `round_openset_unknown_recall`, `round_openset_fpr95` | Multi-metric error bar across client scales |

---

### Trade-off & Analysis Figures (21–26)

| Figure Stem | Output Filename | Source | Key Fields Required |
|---|---|---|---|
| `21_scalability_close_open_tradeoff` | `21_scalability_close_open_tradeoff.png` | Summary of scalability logs | `Final-10 close-set acc`, `close-set macro-F1`, `open-set overall acc`, `open-set macro-F1`, `known acc`, `unknown recall` |
| `22_scalability_efficiency_robustness_pareto` | `22_scalability_efficiency_robustness_pareto.png` | Scalability logs | `Median round time (min)`, `Final-10 open-set macro-F1 (%)`, `Final-10 unknown recall (%)` |
| `23_open_set_threshold_operating_curve` | `23_open_set_threshold_operating_curve.png` | `exp2_scores.csv` | `prototype_rank_score`, `known_or_unknown`, `selected_threshold_used` |
| `24_classwise_open_set_tradeoff` | `24_classwise_open_set_tradeoff.png` | `exp2_confusion_before.csv`, `exp2_confusion_after.csv` | Per-class TP, FP, FN before vs after rejection |
| `25_non_iid_accuracy_retention` | `25_non_iid_accuracy_retention.png` | `convergence_alpha1.csv`, `convergence_alpha01.csv` | Accuracy retention delta between $\alpha=1.0$ and $\alpha=0.1$ |
| `26_confusion_delta_after_rejection` | `26_confusion_delta_after_rejection.png` | `exp2_confusion_before.csv`, `exp2_confusion_after.csv` | Diverging matrix: `(After - Before) / Row_Sum` |

---

## 3. Strict Schema for `scalability_{N}_clients.csv`

Every scalability log file must contain the following columns:
```csv
round,num_clients,mean_client_macro_f1,std_client_macro_f1,worst_client_macro_f1,client_fit_wall_time_sec,round_time_sec,server_aggregation_time_sec,open_set_round_eval_time_sec,round_openset_f1_macro,round_openset_overall_acc,round_openset_known_acc,round_openset_auroc,round_openset_fpr95,round_openset_unknown_recall
```

## 4. Strict Schema for `exp2_scores.csv` (Sample-Level Open-Set)

```csv
sample_id,true_label,closed_pred,open_pred,known_or_unknown,raw_score,prototype_rank_score,selected_threshold_used,final_reject
```
- `known_or_unknown`: string `"known"` or `"unknown"`
- `prototype_rank_score`: float in $[0.0, 1.0]$
- `final_reject`: int `0` (accepted as known) or `1` (rejected as unknown)
