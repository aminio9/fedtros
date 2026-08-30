# FedTROS-MC execution schedule

Status date: 2026-08-30

This schedule is the operational checklist for generating publication evidence after the implementation remediation. A run is publication evidence only when it is completed, canonical, linked to a manifest, and passes `scripts/validate_publication_evidence.py`.

| Phase | Studies | Required settings | Status | Exit criterion |
|---|---|---|---|---|
| 0. Verification | E0-VERIFY | smoke/config contract checks | completed | tests and config validation pass |
| 1. IID baselines | E1-IID-CS, E2-IID-OSR | 10 clients, 100 rounds, seeds 17/42/73/101/137; FedAvg/FedProx/SCAFFOLD/local-only/centralized matched baselines | pending | all manifests completed and canonical |
| 2. Non-IID/generalization | E3-NIID-CS, E4-NIID-FOSR | 10 clients, 100 rounds, five seeds | pending | closed-set and unknown-attack metrics available |
| 3. Dataset transfer | E5-DATASET | every declared dataset, 100 rounds, five seeds | pending | per-dataset aggregate and confidence intervals |
| 4. Efficiency/scale | E6-SCALE, E7-EFFICIENCY | declared client/round grid, five seeds | pending | accuracy--rejection--cost trade-off reported |
| 5. Robustness | E8-LOAO | leave-one-attack-out, five seeds | pending | unknown recall/FPR reported per held-out attack |
| 6. Component evidence | A1-TEACHER, A2-ANCHOR, A3-TRANSFER, A4-PR, A5-FEATURE, S1-SENSITIVITY | fixed protocol, no unknown leakage; five seeds for final tables; A4 common frozen representation | pending | ablation deltas and sensitivity intervals |
| 7. Publication freeze | validator, plots, bundle export | validator must exit 0 | blocked by phases 1--6 | frozen manifest, tables, figures, and archive hash |

## Run order

1. Run E0 and the full test suite.
2. Run E1/E2 to establish matched closed/open-set baselines.
3. Run E3/E4, then E5; do not interpret a single seed as a final result.
4. Run E6/E7/E8 and collect runtime, communication, and robustness metrics.
5. Run A1--A5 and S1 using the same split manifests and seeds as the headline studies.
6. Run the evidence validator; fix every reported error before exporting.
7. Rebuild the manuscript tables/figures from the frozen export and rerun the LaTeX build.

## Gate command

```text
python scripts/validate_publication_evidence.py --runs-dir outputs/runs --report outputs/publication_evidence_status.json
```

The current repository intentionally fails this gate because the available runs are smoke/development artifacts (two clients, two rounds, seed 42) and do not constitute paper evidence. The code-first completion has prepared and validated the full matrix, but the 100-round, five-seed campaign remains to be executed in the GPU/Colab environment.
