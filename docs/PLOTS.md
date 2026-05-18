# Plots

The source of truth is `Experimentplan/testplot.py`. Its palette and 14 required Q1 dashboard panels are mirrored in `src/plotting/theme.py` and `src/plotting/registry.py`.

Required dashboard plots:

1. Scalability: nodes vs final accuracy.
2. Non-IID client data distribution.
3. Mild non-IID convergence and variance.
4. Hard non-IID convergence and variance.
5. Known vs unknown score distributions.
6. Openness vs AUROC.
7. Unknown-detection ROC.
8. Cross-dataset generalization.
9. Before-OSR confusion matrix.
10. After-OSR confusion matrix.
11. Seed robustness boxplot.
12. t-SNE/UMAP latent separation.
13. Communication efficiency.
14. Architectural ablation.

Run:

```bash
poetry run python scripts/plot.py run_dir=outputs/run_id
```

Missing required data produces warnings and a labeled missing-data panel instead of synthetic evidence.
