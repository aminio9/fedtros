# PNPFF Open-Set Detection

The `prototype_rank` component implements a robust PNPFF-Fed adaptation of the
Positive-Negative Prototypes Fusion Framework from Zhong and Cui (2025). It no
longer means a KMeans rank or the uncalibrated paper confidence alone.

## Execution location

Flower clients continue to train the federated student and its existing OSR
branch. They do not fit PNPFF prototypes and do not make open-set rejection
decisions. For DKD-FedOS, client `evaluate()` continues to skip redundant EVT
rejection. After server aggregation, the global evaluation path loads the
aggregated student, splits the shared known-only validation set, fits PNPFF once,
and evaluates the mixed known/unknown open-set test tensor.

This preserves the strict federated privacy boundary: no local client samples,
labels, histograms, or prototypes are uploaded for PNPFF fitting.

## Detector

PNPFF owns and adapts a private copy of the aggregated student backbone. A
trainable identity-initialized 128-dimensional projection is optimized together
with seven positive prototypes and one negative prototype for every known class.
Embeddings and prototypes are unit-normalized to prevent distance/logit
explosion; this normalization is part of the learned embedding function.
Training uses the positive classification, negative classification,
moving-radius, and prototype-diversity objectives from Equations 3-12 of the
paper. Positive and negative objectives use separate alternating optimizer
steps and class-balanced batches. The assignment-diversity softmax uses
`-T * distance`: Equation 8 prints a positive sign, but the paper's distance
definition and accompanying similarity prose require the negative sign.

Known-derived cross-class manifold mixup supplies privacy-safe pseudo-unknowns.
They regularize both prototype probability distributions toward uniform and
support score calibration. This is an APNPFF++-inspired robustness extension;
no real unknown or FoT sample is used during fitting or calibration.

At inference, the class score is the weighted fusion from Equations 27-28. The
accepted class is the maximum fused PNPFF class score. The repository-facing
raw unknown score is `1 - max(fused_class_score)`. This raw paper score is
exported separately. The operational score is a monotonic, unknown-high
calibration fitted on held-out known and independently generated pseudo-unknown
samples. Its default threshold maximizes pseudo-unknown F1 subject to a 5%
known false-unknown-rate constraint. Fixed `tau=0.5` remains available only for
paper-reproduction experiments.

The shared validation set is deterministically stratified using the run seed:
70% fits the adapted embedding/prototypes and 30% selects the best epoch and
calibrates rejection. Selection uses balanced known accuracy and pseudo-unknown
AUROC, not imbalanced ordinary accuracy. Unknown labels are rejected from this
entire process. A failed direction, finite-value, AUROC, or known-FPR health
check aborts `prototype_rank`; combined fusion modes exclude an unhealthy
prototype component.

## Artifacts

Final evaluation writes `pnpff_state.pt`, `pnpff_metadata.json`, the
compatibility artifact `fed_digos_prototypes.json`, and PNPFF distances,
probabilities, fused scores, confidence, prediction, threshold, and unknown
raw/calibrated score and health columns in `open_set_scores.csv`. Per-round
PNPFF fitting is disabled by default because it previously dominated runtime;
set `pnpff.fit_each_round=true` only for an explicit ablation.

The prototype losses and fusion equations are base PNPFF. GAN-based APNPFF is
not implemented; only the known-derived mixup robustness idea is reused.
