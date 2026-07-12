# PNPFF Open-Set Detection

The `prototype_rank` component now implements the base Positive-Negative
Prototypes Fusion Framework (PNPFF) from Zhong and Cui (2025). It no longer
means a KMeans prototype-distance empirical rank.

## Execution location

Flower clients continue to train the federated student and its existing OSR
branch. They do not fit PNPFF prototypes and do not make open-set rejection
decisions. For DKD-FedOS, client `evaluate()` continues to skip redundant EVT
rejection. After server aggregation, the global evaluation path loads the
aggregated student, splits the shared known-only validation set, fits PNPFF,
and evaluates the mixed known/unknown open-set test tensor.

This preserves the strict federated privacy boundary: no local client samples,
labels, histograms, or prototypes are uploaded for PNPFF fitting.

## Detector

PNPFF uses the frozen 128-dimensional global student backbone features. A
trainable identity-initialized 128-to-128 projection is optimized together with
seven positive prototypes and one negative prototype for every known class.
Training uses the positive classification, negative classification,
moving-radius, and prototype-diversity objectives from Equations 3-12 of the
paper. Positive and negative objectives are applied in alternating optimizer
steps.

At inference, the class score is the weighted fusion from Equations 27-28. The
accepted class is the maximum fused PNPFF class score. The repository-facing
unknown score is `1 - max(fused_class_score)`, so larger values always mean
more unknown. The default fixed rejection threshold is `0.5`; a known-FPR
validation quantile is available as a sensitivity option.

The shared validation set is deterministically stratified using the run seed:
70% fits the PNPFF projection/prototypes and 30% selects the best epoch and,
when configured, calibrates the known-FPR threshold. Unknown labels are rejected
from this entire process.

## Artifacts

Final or round evaluation writes `pnpff_state.pt`, `pnpff_metadata.json`, the
compatibility artifact `fed_digos_prototypes.json`, and PNPFF distances,
probabilities, fused scores, confidence, prediction, threshold, and unknown
score columns in `open_set_scores.csv`.

Only base PNPFF is implemented. APNPFF and APNPFF++ adversarial generation are
outside this implementation.
