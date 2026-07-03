# Evaluation

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

Outputs:

- `open_set_metrics.json`
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
