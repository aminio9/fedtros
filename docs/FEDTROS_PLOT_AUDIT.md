# FedTROS Internal Plotting Audit (Workstream C1)
================================================================

**Document Version:** 1.0 (Post-Refactor Q1 Canonical)
**Scope:** `fedtros` Research Repository (Internal Plotting Inventory & Retirement Classification)
**Strict Rule:** Training produces pure data (CSV, JSON, NumPy tensors), not paper figures. All publication figure rendering is delegated to the dedicated, read-only `plots` project.

---

## 1. Executive Summary

This audit catalogs all plotting code, figure generation, visualization functions, and graphical artifacts within the FedTROS training repository. Every plotting component has been analyzed against the core principle:
- **Numerical and scientific metric calculations are preserved and isolated** into data-producing modules.
- **Publication figure rendering is demoted/retired** from the training pipeline.
- Visualizations are categorized into five lifecycle states: `DELETE`, `SUPPLEMENTARY`, `DIAGNOSTIC`, `LEGACY`, or `KEEP (DATA-ONLY)`.

---

## 2. Detailed Plotting Inventory

| # | File Path | Function / Block | Called By | Training Dep? | Result Dep? | Duplicates Plot Project? | Keep Data Calc? | Visualization Action | Classification |
|---|-----------|-------------------|-----------|---------------|-------------|--------------------------|-----------------|----------------------|----------------|
| 1 | `src/plotting/theme.py` | `apply_theme()`, `CUSTOM_COLORS`, `CMAP_SUNSET` | `src/plotting/plots.py` | No | No | Yes (`plots/src/style.py`) | No (Styles only) | Demote to diagnostic theme | `LEGACY` |
| 2 | `src/plotting/registry.py` | `REQUIRED_PLOTS`, `PlotSpec`, `plot_specs_by_id()` | `src/plotting/plots.py` | No | No | Yes (`plots/src/config.py`) | Yes (IDs & metadata) | Keep as internal registry | `LEGACY` |
| 3 | `src/plotting/plots.py` | `_plot_scalability()` | `render_required_plots()` | No | No | Yes (Fig 01, 16–20) | Yes (`scalability_round_metrics.csv`) | Remove from training output | `DIAGNOSTIC` |
| 4 | `src/plotting/plots.py` | `_plot_non_iid_distribution()` | `render_required_plots()` | No | No | Yes (Fig 02) | Yes (`client_class_distribution.csv`) | Remove from training output | `DIAGNOSTIC` |
| 5 | `src/plotting/plots.py` | `_plot_convergence()` | `render_required_plots()` | No | No | Yes (Fig 03, 04) | Yes (`federated_history.csv`) | Remove from training output | `DIAGNOSTIC` |
| 6 | `src/plotting/plots.py` | `_plot_score_distribution()` | `render_required_plots()` | No | No | Yes (Fig 05) | Yes (`fedtros_osr_scores.csv`) | Remove from training output | `DIAGNOSTIC` |
| 7 | `src/plotting/plots.py` | `_plot_openness_vs_auroc()` | `render_required_plots()` | No | No | Yes (Fig 06) | Yes (`openness_metrics.csv`) | Remove from training output | `DIAGNOSTIC` |
| 8 | `src/plotting/plots.py` | `_plot_roc_curve()` | `render_required_plots()` | No | No | Yes (Fig 07) | Yes (`open_set_roc_curve.csv`) | Remove from training output | `DIAGNOSTIC` |
| 9 | `src/plotting/plots.py` | `_plot_cross_dataset()` | `render_required_plots()` | No | No | Yes (Fig 08) | Yes (`cross_dataset_metrics.csv`) | Remove from training output | `DIAGNOSTIC` |
| 10 | `src/plotting/plots.py` | `_plot_matrix()` (Confusion) | `render_required_plots()` | No | No | Yes (Fig 09, 10) | Yes (`before/after_osr_confusion_matrix.csv`) | Remove from training output | `DIAGNOSTIC` |
| 11 | `src/plotting/plots.py` | `_plot_box()` (Seed Variance) | `render_required_plots()` | No | No | Yes (Fig 11) | Yes (`seed_robustness.csv`) | Remove from training output | `DIAGNOSTIC` |
| 12 | `src/plotting/plots.py` | `_plot_latent()` (PCA Scatter) | `render_required_plots()` | No | No | Yes (Fig 12) | Yes (`latent_embeddings.csv`) | Remove from training output | `DIAGNOSTIC` |
| 13 | `src/plotting/plots.py` | `_plot_communication_efficiency()` | `render_required_plots()` | No | No | Yes (Fig 13) | Yes (`communication_metrics.csv`) | Remove from training output | `DIAGNOSTIC` |
| 14 | `src/plotting/plots.py` | `_plot_ablation()` | `render_required_plots()` | No | No | Yes (Fig 14) | Yes (`ablation_metrics.csv`) | Remove from training output | `DIAGNOSTIC` |
| 15 | `src/plotting/plots.py` | `render_training_plots()` | `src/plotting/report.py` | No | No | Yes (Internal loss/acc) | Yes (`metrics.csv`) | Demote to optional debug preview | `DIAGNOSTIC` |
| 16 | `src/plotting/report.py` | `generate_plots()` | `run.py:plot`, `scripts/plot.py` | No | No | Yes (`plots/scripts/generate_all.py`)| Yes (Manifest generation) | Keep as diagnostic entrypoint | `LEGACY` |
| 17 | `src/plotting/io.py` | `load_csv_if_exists()`, `first_existing()` | `src/plotting/plots.py` | No | No | No | Yes (Move to `src/analysis/loaders.py`)| Utility functions | `KEEP (DATA-ONLY)` |
| 18 | `scripts/plot.py` | Standalone CLI runner | Manual CLI | No | No | Yes (`plots/scripts/*.py`) | Yes (For backward CLI compatibility)| Keep as diagnostic tool | `LEGACY` |
| 19 | `scripts/scalability_report.py` | `_save_line_plot()`, `_save_bar_plot()` | Manual CLI | No | No | Yes (`plots/src/figures/scalability_plots.py`)| Yes (`compute_scalability_table_summary`)| Remove matplotlib calls, export CSVs | `DIAGNOSTIC` |
| 20 | `src/artifacts/embeddings.py` | `export_latent_embeddings()` | `src/evaluation/run.py` | No | Yes | No | Yes (PCA 2D projection) | None (Generates CSV/JSON only) | `KEEP (DATA-ONLY)` |
| 21 | `src/artifacts/communication.py`| `build_communication_metrics()` | `src/federated/run.py` | No | Yes | No | Yes (Payload byte tracking) | None (Generates CSV/JSON only) | `KEEP (DATA-ONLY)` |
| 22 | `src/evaluation/openset_eval.py` | `evaluate_open_set()` | `src/evaluation/run.py` | Yes | Yes | No | Yes (ROC, PR, CM, AUROC, AUPRC, FPR95)| None (Generates CSV/JSON only) | `KEEP (DATA-ONLY)` |
| 23 | Legacy FedTROS OSR/EVT evaluation path | Archived | Removed from active source | Yes | No | No | Historical only | None | `ARCHIVED` |

---

## 3. Classification Key

- **`DELETE`**: Redundant or unverified decorative scripts that produce outdated or incorrect figures.
- **`LEGACY`**: Preserved for backward CLI compatibility (`scripts/plot.py`, `src/plotting/report.py`), but decoupled from core training pipelines and superseded by `src/analysis/`.
- **`DIAGNOSTIC`**: Fast, lightweight preview renderers used only for developer debugging during training; strictly excluded from the paper publication figure set.
- **`SUPPLEMENTARY`**: Data generators producing extended tables or appendix material.
- **`KEEP (DATA-ONLY)`**: Scientific numeric calculations (empirical-rank calibration, score rankings, ROC/PR coordinate generation, confusion matrices, communication byte counts) which must never be deleted.

---

## 4. Separation of Responsibilities

```
+-------------------------------------------------------------+
|                      FedTROS Repo                           |
|                                                             |
|  [ Training / FL Simulation ]                               |
|        │                                                    |
|        ▼                                                    |
|  [ Numerical Evaluation & Instrumentation ]                 |
|        │ (Pure Data: CSV, JSON, Parquet, Npz)               |
|        ▼                                                    |
|  [ src/analysis Package ]                                   |
|        ├── query.py          (Declarative run discovery)    |
|        ├── loaders.py        (Structured data ingestion)    |
|        ├── validation.py     (Compatibility & integrity)    |
|        ├── aggregation.py    (Multi-seed mean, CI, delta)   |
|        ├── statistics.py     (p-values, effect sizes)       |
|        ├── tables.py         (Machine-readable LaTeX & CSV) |
|        └── export.py         (Standard data contracts)      |
|                                                             |
|  [ scripts/export_plot_data.py Adapter ]                    |
|        │                                                    |
+--------┼────────────────────────────────────────────────----+
         │ (Canonical Plot Data Contract)
         ▼
+-------------------------------------------------------------+
|               plots (Good Plotting Project)                 |
|                                                             |
|  [ 29 Publication Figures (600 DPI, B&W + Color, LaTeX) ]   |
|  [ Publication LaTeX Tables (E1–E8, Scalability, Ablation) ]|
+-------------------------------------------------------------+
```
