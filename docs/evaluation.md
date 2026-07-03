# Evaluation

<<<<<<< HEAD
## Closed-Set

Closed-set evaluation loads `checkpoint.path`, `evaluation.closed_set_data`, and `evaluation.class_names`. It assumes the full source label set is present in the closed-set tensors and no unknown labels are encoded there. It writes:

- `test_metrics.json`
- `test_classification_report.json`
- `test_confusion_matrix.csv` (labeled rows/columns)
- `test_predictions.jsonl`

Metrics include `test/loss`, `test/accuracy`, balanced accuracy, macro precision/recall/F1, and per-class accuracy.

## Open-Set / EVT

Open-set evaluation fits EVT models on known calibration samples from `validation.pt`, calibrates a threshold on that validation split, and evaluates samples with unknown labels encoded as `-1`.
=======
## Closed Set

Closed-set evaluation lives in `src/evaluation/closed_set.py`. Reusable metric computation lives in `src/evaluation/metrics.py`.

Inputs:

- checkpoint loaded into the CVAE-DQN agent
- `evaluation.closed_set_data`
- `evaluation.class_names`

Outputs:

- `<prefix>_metrics.json`
- `<prefix>_classification_report.json`
- `<prefix>_confusion_matrix.csv`
- `<prefix>_predictions.jsonl` when enabled

Metrics include loss, accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, and per-class recall with a single label ordering.

## Open Set

Open-set evaluation lives in `src/evaluation/open_set.py` and currently uses EVT reconstruction scoring.
>>>>>>> ea28efe (Initial commit with updated source code)

Outputs:

- `open_set_metrics.json`
<<<<<<< HEAD
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
`latent_embeddings.csv` to `evaluation.output_dir` with `x`, `y`, `label`, and
`source` columns. The default projection is a deterministic 2D PCA over the
prior-network latent vectors from the active evaluation tensor: `closed_set`
uses the closed-set test data, while `open_set` uses the open-set test data
without duplicating the closed-set rows.
=======
- `open_set_scores.csv`
- `open_set_roc_curve.csv`
- `open_set_pr_curve.csv`
- `open_set_oscr_curve.csv`
- `open_set_threshold_sensitivity.csv`
- `before_osr_confusion_matrix.csv`
- `after_osr_confusion_matrix.csv`

Metric keys include:

- `open_set/auroc`
- `open_set/aupr_out`
- `open_set/aupr_in`
- `open_set/auoscr`
- `open_set/fpr95`
- `open_set/unknown_detection_rate`
- `open_set/unknown_f1`
- `open_set/unknown_precision`
- `open_set/known_accuracy_after_rejection`
- `open_set/known_rejection_rate`

## Latent Embeddings

When `evaluation.export_latent_embeddings=true`, `src/artifacts/embeddings.py` exports latent vectors from the active evaluation tensor. Closed-set mode exports closed-set rows; open-set mode exports the open-set tensor without duplicating closed-set rows.
>>>>>>> ea28efe (Initial commit with updated source code)
