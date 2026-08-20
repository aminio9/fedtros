# FedTROS-PR Execution, Logging, Tracking, and Checkpoint Migration Report

**Date:** 2026-08-19
**Status:** Implemented in the supplied source bundle; full Flower/W&B GPU execution remains to be validated on the target Python 3.11/3.12 server.

## Final architecture

The active execution path is now:

```text
declarative study YAML
 -> scripts/run_study.py
 -> scripts/run.py
 -> validated resolved config + automatic run identity
 -> Python operational logging
 -> scientific training/evaluation
 -> local ResultStore
 -> W&B (online/offline) or NullTracker (disabled)
 -> checkpoint/manifests/finalization
```

There is no active MLflow/CompositeTracker/LocalTracker experiment-tracking stack. Local scientific persistence is handled by `ResultStore`; W&B is the single interactive tracker.

## Canonical execution tools

- `scripts/doctor.py` — pre-flight environment and integration checks.
- `scripts/studies.py` — list/show E0–E8, A1–A5, S1 contracts.
- `scripts/run.py` — one exact study cell only.
- `scripts/run_study.py` — matrix expansion, filtering, dry-run, only-missing, resume, force-new, multi-GPU independent scheduling.
- `scripts/runs.py` — run-state inspection.
- `scripts/resume.py` — strict compatible checkpoint continuation.
- `scripts/evaluate.py` — evaluation-only workflow.
- `scripts/build_q1_results.py` — non-visual statistics/tables.
- `scripts/export_publication_bundle.py` — separate plot-repository integration.

Legacy shell launchers and overlapping runner scripts are archived under `archive/migration_2026/old_execution/` and are not imported by the active system.

## W&B tracker

Active package:

```text
src/infrastructure/tracking/
    base.py
    wandb_tracker.py
    null_tracker.py
    factory.py
```

Supported modes:

```text
online
offline
disabled
```

Scientific modules do not import W&B. `RunServices` receives metrics from scientific code, saves them locally through `ResultStore`, and forwards small structured metrics to the tracker.

## Local ResultStore

`src/experiment/result_store.py` owns durable result persistence independently of W&B. Canonical run directories contain resolved configuration, manifests, metrics, predictions, artifacts, checkpoints, and logs. This is the publication/reproducibility source of truth.

## Logging/status cleanup

Removed from active operation:

- manual `tee` requirement;
- manual `mkdir -p logs`;
- `run_status.txt`;
- DONE/FAILED shell appends;
- Host1/Host2 provenance comments.

Run state is stored as one of:

```text
CREATED
RUNNING
COMPLETED
FAILED
INTERRUPTED
RESUMED
CANCELLED
```

in the run manifest and mirrored to W&B summary metadata.

## Automatic run identity

Users no longer supply `tracking.run_id`. Run identity is generated from the study plus scientifically meaningful selectors and a configuration-hash suffix. Collisions are validated before execution.

## Checkpoint/resume

Checkpoint schema v2 carries canonical method/teacher/config/git information. Old DQN checkpoints are rejected as incompatible. Resume code verifies configuration compatibility and restores private VCT/RNG state where persistent continuation requires it.

## E7 instrumentation

Communication is measured from the actual NumPy arrays transported by the Flower simulation path. Communication keys are summed across participating clients rather than sample-weighted.

Runtime instrumentation records synchronous client critical-path time, teacher/student local time, aggregation, open-set evaluation, server round wall time, residual/orchestration time, and cumulative time. Whole-pipeline phase timing is stored separately in `metrics/pipeline_timing.json` so it does not overwrite per-round timing.

## Parallel-safe preprocessing and paired partitions

Generated preprocessing tensors/manifests are now isolated under `outputs/runs/<run_id>/data/` rather than a shared `data/processed` directory. This makes independent multi-GPU study execution safe from cross-run file clobbering. Scientific pairing across FedAvg-Student, FedProx-Student, FedTROS-PR, and ablations is enforced separately by an immutable study-level paired-partition file keyed by dataset, protocol, client count, alpha/IID state, and seed. Loading that file validates the known/unknown label protocol and training-label hash before any client assignment is reused.

## Dependency migration

- W&B is an active dependency.
- Matplotlib/Seaborn and MLflow were removed from the FedTROS direct dependency set.
- The stale pre-refactor Poetry lock was archived because it described the old dependency graph. Regenerate it on Python 3.11/3.12 with `poetry lock`.
