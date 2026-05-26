# Evaluation

## Closed-Set

Closed-set evaluation loads `checkpoint.path`, `evaluation.closed_set_data`, and `evaluation.class_names`. It writes:

- `test_metrics.json`
- `test_classification_report.json`
- `test_confusion_matrix.csv` (labeled rows/columns)
- `test_predictions.jsonl`

Metrics include `test/loss`, `test/accuracy`, balanced accuracy, macro precision/recall/F1, and per-class accuracy.

## Open-Set / EVT

Open-set evaluation fits EVT models on known calibration samples from `validation.pt`, calibrates a threshold on that validation split, and evaluates samples with unknown labels encoded as `-1`.

Outputs:

- `open_set_metrics.json`
- `open_set_scores.csv` with `y_true`, `raw_pred`, `y_pred`, `unknown_score`, and `is_unknown`
- `open_set_roc_curve.csv`
- `open_set_pr_curve.csv`
- `before_osr_confusion_matrix.csv`
- `after_osr_confusion_matrix.csv`
- `evt/evt_models.pkl`
- `evt/evt_meta.json`

Primary metric names:

- `open_set/auroc`
- `open_set/auprc`
- `open_set/fpr95`
- `open_set/unknown_detection_rate`
- `open_set/unknown_f1`

## Latent Embeddings

When `evaluation.export_latent_embeddings=true`, evaluation also writes
`latent_embeddings.csv` to `evaluation.output_dir` with `x`, `y`, and `label`
columns. The default projection is a deterministic 2D PCA over the prior-network
latent vectors from the closed-set and open-set evaluation samples.
