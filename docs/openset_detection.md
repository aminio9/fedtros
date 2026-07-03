# Open-Set Detection

## Module Boundaries

- `src/evaluation/open_set.py`: high-level evaluation, reports, curves, confusion matrices.
- `src/openset/evt.py`: EVT/GPD model fitting, probability scoring, serialization.
- `src/openset/scorers.py`: MSP, energy, prototype-distance, Mahalanobis, and no-rejection scorer utilities.
- `src/openset/thresholding.py`: validation-only threshold selection and score-direction handling.

`src/evaluation/openset_eval.py` is a compatibility shim only.

## First-Class Evaluator

The active `run.py` open-set path evaluates EVT over reconstruction error:

```text
pred = argmax Q(mu_p(s), s)
mu_q, _ = Recognition(s, pred)
s_hat = Generator(mu_q, pred)
score = mean((s_hat - s)^2) * error_scale_factor
unknown_probability = EVT_class(pred).predict_probability_unknown(score)
```

Higher scores mean more unknown-like.

## Calibration Protocol

Default config:

```yaml
open_set:
  evt:
    threshold_mode: validation_known_fpr
    calibration_protocol: validation_only
    target_known_fpr: 0.05
```

EVT fitting and threshold calibration use known validation samples only. Unknown test samples are not used for training, fitting, calibration, early stopping, or checkpoint selection.

## EVT Tail Semantics

Preferred keys:

- `tail_fraction`: fraction in `(0, 1]`
- `tail_percent`: percent in `(0, 100]`

Legacy `tail_size_percent` is accepted only with explicit `tail_semantics` because values such as `1.0` are ambiguous.

## Metrics And Outputs

Open-set evaluation writes:

- `open_set_metrics.json`
- `open_set_scores.csv`
- `open_set_roc_curve.csv`
- `open_set_pr_curve.csv`
- `open_set_oscr_curve.csv`
- `open_set_threshold_sensitivity.csv` when both known and unknown samples exist
- `before_osr_confusion_matrix.csv`
- `after_osr_confusion_matrix.csv`
- `evt/evt_models.pkl`
- `evt/evt_meta.json`

Logged metrics include AUROC, AUPR-In, AUPR-Out, FPR@95TPR, AUOSCR, unknown precision/recall/F1, known rejection rate, known accuracy after rejection, and global threshold.

## Scorer Baselines

The scorer utilities and configs exist for MSP, energy, prototype distance, Mahalanobis distance, and no rejection. They are covered by unit tests and cheap validation. They still need a first-class evaluation runner before they can be reported as matched experiment baselines.
Use `build_open_set_scorer_from_config` to instantiate these utilities in tests or future runners. `run.py` now rejects these scorer configs for full experiment pipelines so they cannot be mistaken for completed baselines.

`open_set=openmax_evt` is a scaffold alias only. It is rejected until a real OpenMax evaluator is implemented.
