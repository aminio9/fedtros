# DKD-FedOS Phase 2: Student-Decoder EVT Open-Set Recognition

This phase replaces the fragile local-generator open-set detector with a global student-decoder EVT detector.

## Scope

Phase 2 uses only the unknown-attack recognition idea from Yang et al. (IEEE TIFS 2025):

1. train a known-class model;
2. reconstruct known samples with a trained decoder;
3. fit EVT/GPD on the upper tail of known reconstruction errors;
4. reject a test sample as unknown when its reconstruction error exceeds the class-specific EVT threshold.

The dynamic update stage from Yang et al. is **not** implemented in this project. Detected unknown samples are not merged into training and no incremental class-expansion update is performed.

## Why the detector moved from local generator to global student decoder

The old open-set evaluator used the local teacher generator:

```text
prior_net + recognition_net + generation_net -> reconstruction error -> EVT
```

That is weak under non-IID FL because a client may not observe every known class. A local generator can reconstruct its local classes well but assign high reconstruction error to a missing known class, falsely rejecting it as unknown.

Phase 2 instead uses:

```text
global student encoder + global student decoder -> reconstruction error -> class-wise EVT
```

The student decoder is aggregated with the global student, so the reconstruction model receives global known-class knowledge across clients.

## Active experiments

The student decoder is enabled only in open-set experiments:

```text
E2: open-set IID
E4: open-set non-IID
```

Closed-set experiments keep the previous behavior:

```text
E1: closed-set IID, no student decoder EVT
E3: closed-set non-IID, no student decoder EVT
```

## Open-set protocol

For B-NAT / blockchain traffic:

```text
Known classes:  Normal, BP, DoS, MitM
Unknown class:  FoT
```

The open-set run must satisfy:

```text
model.num_actions = 4
FoT absent from training/validation
FoT present only in shared_open_set_test.pt with label -1
```

The logs should include:

```text
OPEN-SET PROTOCOL ACTIVE
backend=student_decoder
calibration_unknown_samples=0
open_test_unknown_samples > 0
```

## EVT implementation

`src/openset/evt.py` implements a class-wise `EVTModel`:

```text
R_k = reconstruction errors of correctly classified validation samples from known class k
u_k = high threshold selected by Mean Excess Function (MEF) linearity
excess = R_k - u_k, where R_k > u_k
GPD(excess; xi_k, omega_k) fitted with MLE
T_k = class-specific rejection threshold at target_known_fpr
```

The fallback is quantile-based if the MEF segment is unstable or too small.

The final decision is intentionally simple:

```text
c = argmax(student_classifier(x))
x_hat = student_decoder(student_feature(x), c)
e = MSE(x, x_hat)

if e > T_c:
    Unknown
else:
    c
```

No uncertainty score, no teacher/student disagreement score, and no local/global reconstruction score fusion are used.

## Outputs

Open-set evaluation writes:

```text
open_set_metrics.json
open_set_scores.csv
openset_report.txt
before_osr_confusion_matrix.csv
after_osr_confusion_matrix.csv
open_set_roc_curve.csv
open_set_pr_curve.csv
student_reconstruction_errors_known.csv
student_reconstruction_errors_unknown.csv
evt_thresholds.json
evt/evt_models.pkl
evt/evt_meta.json
```

## Important config values

```yaml
open_set:
  evt:
    backend: student_decoder
    classwise: true
    threshold_method: mef
    tail_size_percent: 0.10
    min_errors_per_class: 50
    min_tail_size: 20
    target_known_fpr: 0.05
    fit_correct_only: true
```

For ablations only, the legacy teacher/generator backend can be selected:

```bash
open_set.evt.backend=teacher_generator
```

The paper method should use `student_decoder`.
