# FedTROS-PR -> plots publication bundle

The research and plotting repositories remain independent Python projects. They communicate only through an immutable, checksum-verified file contract.

## Producer

```bash
python scripts/export_publication_bundle.py \
  --outputs-dir outputs \
  --target-root publication_exports \
  --freeze-id <freeze-id>
```

Schema:

- `schema_name = fedtros_pr_publication_bundle`
- `schema_version = 1`
- `method = FedTROS-PR`
- canonical tabular format: CSV

The bundle `manifest.json` records the source run IDs, config/split hashes, code commit, studies present, and SHA-256 checksums of all exported files.

## Consumer

The separate plots repository loads the bundle through:

```python
from src.data.fedtros_bundle import load_publication_bundle
bundle = load_publication_bundle(PATH, verify_checksums=True)
```

The plot renderer refuses an unsupported schema/method or corrupted checksum.

## Current study exports

Each present study can include:

- `summary_runs.csv` — scalar run-level results.
- `summary.csv` — FedTROS-computed multi-seed mean/SD/95% CI.
- `round_curves.csv` — per-round data.
- `scores.csv` — sample-level open-set scores.
- `roc.csv`, `pr.csv` — scientific curve coordinates computed by FedTROS.
- `client_metrics.csv`.
- `client_distribution.csv`.
- `communication.csv`.
- `runtime.csv`.
- `paired_deltas.csv` — paired ablation deltas computed in FedTROS.
- `raw_artifacts/` — selected numeric confusion matrices.

The plots repository may perform visual transformations (ordering, units, heatmap normalization) but must not create a second statistical source of truth.

## Main figure registry

1. Client class-support heatmap: IID + alpha 1.0/0.5/0.1 when available.
2. Unknown-score separation + PR diagnostic.
3. Non-IID closed-set Macro-F1 vs alpha with CI.
4. Known/unknown operating trade-off under non-IID.
5. Leave-one-attack-out heatmap.
6. Paired component-ablation deltas.
7. Performance vs cumulative model-parameter communication.
8. Fixed-data scalability performance/runtime.
