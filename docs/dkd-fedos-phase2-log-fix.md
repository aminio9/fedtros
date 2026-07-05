# DKD-FedOS Phase 2 Log and Evaluation Fix

This patch fixes the Phase 2 open-set run failure observed when launching DKD-FedOS with an open-set experiment overlay and a method overlay such as:

```bash
poetry run python run.py experiment=exp2 +method=dkd_fedos seed=42
```

## Problem

The run log showed that the open-set protocol was active (`num_actions=4`, FoT held out), but the student reconstruction decoder was disabled:

```text
student_reconstruction_enabled=False
student_reconstruction_weight=0.000
```

Then every client attempted EVT fitting during Flower `evaluate()` and failed with:

```text
RuntimeError: EVT backend=student_decoder requires training.dkd_student_reconstruction_enabled=true.
```

## Root Cause

The `method=dkd_fedos` Hydra overlay defined `training.dkd_student_reconstruction_enabled=false` and `training.dkd_student_reconstruction_weight=0.0`. When the method overlay was applied after the experiment overlay, it silently overrode the E2/E4 open-set settings.

## Fixes

1. Removed reconstruction enable/weight defaults from `src/configs/method/dkd_fedos.yaml`.
   - Closed-set runs still get `false/0.0` from `training/default.yaml`.
   - Open-set E2/E4 enable the decoder from their experiment overlays.

2. Added runtime config normalization in `src/utils/config.py`.
   - If DKD-FedOS uses `open_set.evt.backend=student_decoder`, the decoder is enabled and a positive reconstruction weight is enforced.
   - E1/E3 stay unchanged because EVT is disabled there.

3. Added a fail-fast Phase 2 contract check.
   - Invalid open-set student-decoder runs now fail before training instead of producing repeated per-client EVT tracebacks.

4. Disabled per-client EVT during DKD-FedOS Flower `evaluate()` by default.
   - Phase 2 now performs one final global-student open-set evaluation after federated training.
   - Client evaluation logs no longer spam EVT fitting errors.

5. Improved evaluation logs.
   - Open-set runs now label the per-round known-only classifier report as `Known-Class Report (pre-OSR)` instead of `Closed-Set Report`.
   - Final open-set evaluation logs clearly show the protocol, backend, EVT fitting method, thresholds, and decision rule.

## Expected Open-Set Logs

A correct E2/E4 DKD-FedOS run should show:

```text
DKD-FedOS PHASE-2 STUDENT-DECODER EVT ACTIVE
student_reconstruction_enabled=True
student_reconstruction_weight=0.100
Student reconstruction head | enabled=True
avg_dkd_student_reconstruction_loss=...
Client X: skipping per-client EVT during DKD-FedOS evaluate(); final global student-decoder EVT runs after FL.
OPEN-SET PROTOCOL ACTIVE ... backend=student_decoder
Fitting class-wise Yang-style EVT on global-student reconstruction errors
Class-wise EVT fitted
Open-set metrics | backend=student_decoder | ...
```

## Closed-Set Safety

E1 and E3 remain unchanged:

```text
open_set.evt.enabled=false
training.dkd_student_reconstruction_enabled=false
training.dkd_student_reconstruction_weight=0.0
```
