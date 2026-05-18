# Evaluation

## Closed-Set

Closed-set evaluation loads `checkpoint.path`, `evaluation.closed_set_data`, and `evaluation.class_names`. It writes:

- `test_metrics.json`
- `test_classification_report.json`
- `test_confusion_matrix.csv`
- `test_predictions.jsonl`

Metrics include `test/loss`, `test/accuracy`, balanced accuracy, macro precision/recall/F1, and per-class accuracy.

## Open-Set / EVT

Open-set evaluation fits EVT models on known calibration samples, calibrates a threshold using validation/known samples, and evaluates samples with unknown labels encoded as `-1`.

Outputs:

- `open_set_metrics.json`
- `open_set_scores.csv`
- `open_set_roc_curve.csv`
- `open_set_pr_curve.csv`
- `evt/evt_models.pkl`
- `evt/evt_meta.json`

Primary metric names:

- `open_set/auroc`
- `open_set/auprc`
- `open_set/fpr95`
- `open_set/unknown_detection_rate`
- `open_set/unknown_f1`
