# FedTROS-PR Results and Separate Plot-Repository Integration Report

**Date:** 2026-08-19
**Status:** Implemented and synthetic end-to-end contract validation passed.

## Final responsibility boundary

### FedTROS repository

Owns:

- training/evaluation;
- canonical local run results;
- W&B monitoring;
- provenance;
- multi-seed aggregation;
- confidence intervals and paired deltas;
- paper tables;
- publication-bundle export.

FedTROS does **not** render publication figures.

### Separate `plots/` repository

Owns:

- publication-bundle schema/checksum validation;
- visual transforms/layout;
- PNG/PDF/SVG rendering;
- output verification.

The plots repository does **not** train, read FedTROS Python modules, or recompute headline statistics.

## Internal FedTROS plotting removal

The following were removed from the active pipeline and archived:

- `src/plotting/`;
- `scripts/plot.py`;
- `scripts/scalability_report.py`;
- old fixed plot-data exporter;
- plotting Hydra config;
- internal plotting test;
- old generated plot-data artifacts.

Matplotlib imports were removed from scientific/federated source. A source test guards against reintroducing Matplotlib/Seaborn/Plotly into canonical FedTROS code.

## Canonical scientific run data

FedTROS saves numeric artifacts required for later rendering, including:

- closed/open confusion arrays;
- raw OSR sample scores;
- exported ROC/PR coordinate data;
- per-round histories;
- client-level performance;
- class-distribution manifests;
- actual communication bytes;
- runtime decomposition;
- Prototype-Rank calibration/prototype artifacts.

No figure image is required to reconstruct a scientific result.

## Versioned integration contract

FedTROS exports:

```text
publication_exports/<freeze_id>/
    manifest.json
    runs.csv
    aggregates.csv
    paired_deltas.csv                 # when applicable
    E1-IID-CS/
    E2-IID-OSR/
    E3-NIID-CS/
    E4-NIID-FOSR/
    E5-DATASET/
    E6-SCALE/
    E7-EFFICIENCY/
    E8-LOAO/
    A1-TEACHER/
    A2-ANCHOR/
    A3-TRANSFER/
    A4-PR/
    A5-FEATURE/
    S1-SENSITIVITY/
    provenance/artifact_sources.json
```

`manifest.json` includes schema name/version, canonical method/teacher/detector, freeze ID, code commit, source run IDs, config/split hashes, checksums, and studies present.

The plots repository refuses unsupported or corrupted bundles.

## Active Q1 figure registry

The separate plot repository now renders the current eight main-paper figures:

1. client heterogeneity / class-support heatmap;
2. unknown-score separation + PR operating diagnostic;
3. non-IID closed-set Macro-F1 vs alpha with CI;
4. known/unknown operating trade-off under non-IID;
5. leave-one-attack-out heatmap;
6. paired component-ablation deltas;
7. performance vs cumulative communication;
8. fixed-data scalability performance/runtime.

The legacy 29-figure implementation is archived and is not the active publication contract.

## Statistical ownership

Means, across-seed SD, 95% CIs, paired deltas, and compatibility checks are computed in FedTROS. The plot repository renders those values and may perform only visual transforms such as ordering, display normalization, or unit conversion.

## Integration validation

A synthetic collection of completed canonical runs was processed through:

```text
FedTROS analysis
 -> build_q1_results.py
 -> export_publication_bundle.py
 -> plots bundle loader/checksum validator
 -> generate_all.py --strict
 -> verify_outputs.py
```

All eight main figures were generated and the plot output verifier returned PASS.
