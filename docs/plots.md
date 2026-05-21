# Plots

The source of truth for the visual style is `Experimentplan/testplot.py`, but the repo now renders one high-quality image per experiment instead of a single dashboard canvas. The palette and 14 required Q1 figures are mirrored in `src/plotting/theme.py` and `src/plotting/registry.py`.

Required figures:

1. Scalability: nodes vs final accuracy.
2. Non-IID client data distribution.
3. Mild non-IID convergence and variance.
4. Hard non-IID convergence and variance.
5. Known vs unknown EVT score distributions.
6. Openness vs AUROC.
7. Unknown-detection ROC.
8. Cross-dataset generalization.
9. Before-OSR confusion matrix.
10. After-OSR confusion matrix.
11. Seed robustness boxplot.
12. t-SNE/UMAP latent separation.
13. Communication efficiency.
14. Architectural ablation.

Output files are rendered individually as `plots/01_<plot_id>.png` and `plots/01_<plot_id>.pdf` style artifacts, plus `plots/plot_manifest.json` for traceability.

Evaluation can also write `latent_embeddings.csv` automatically when
`evaluation.export_latent_embeddings=true`.

Open-set plots 9 and 10 now read from the dedicated files written by `src/evaluation/openset_eval.py`:

- `before_osr_confusion_matrix.csv`
- `after_osr_confusion_matrix.csv`

Plot 5 uses `open_set_scores.csv` together with `open_set_metrics.json` when the calibrated EVT threshold is available.

Multi-run convergence plots read `comparison_metrics.csv` when available. Generate it with:

```bash
poetry run python scripts/compare_runs.py runs='[outputs/run1,outputs/run2]'
```

Suite-level CSVs should be staged with:

```bash
poetry run python scripts/build_suite_artifacts.py runs='[outputs/run1,outputs/run2,outputs/run3]'
```

That command writes `scalability.csv`, `openness_metrics.csv`,
`cross_dataset_metrics.csv`, `seed_robustness.csv`, `latent_embeddings.csv`,
`communication_metrics.csv`, and `ablation_metrics.csv` into the suite run
directory together with `suite_artifacts_manifest.json`.

Run:

```bash
poetry run python scripts/plot.py run_dir=outputs/run_id
```

Missing required data produces warnings and a labeled missing-data figure instead of synthetic evidence.
