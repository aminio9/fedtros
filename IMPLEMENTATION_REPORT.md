# FedTROS-PR VCT Implementation Report

**Implementation target:** FedTROS-PR VCT Q1 experiment infrastructure and separate publication-renderer integration
**Canonical method:** FedTROS-PR — Federated Teacher-Regularized Open-Set Recognition with Prototype-Rank Rejection
**Teacher:** Variational Classifier Teacher (VCT)
**Open-set detector:** Prototype-Rank Rejection (`prototype_rank`)

## 1. Implemented architecture

The source bundle now uses the following separation of responsibilities:

```text
FedTROS repository
  study config -> run/study runner -> training/evaluation
  -> canonical local ResultStore
  -> W&B (online/offline/disabled) for observability
  -> multi-seed statistics/tables
  -> versioned publication bundle

Separate plots repository
  publication bundle -> checksum/schema validation
  -> Q1 figure rendering only
```

There are no Python imports between the two repositories. The versioned publication bundle is the integration API.

## 2. Scientific-core migration

Implemented/verified:

- canonical `FedTROS-PR` / `fedtros_pr` naming;
- supervised Variational Classifier Teacher in `src/models/variational_teacher.py`;
- deterministic VCT posterior mean for transfer;
- one-way teacher-to-student KD and feature alignment;
- coverage-adaptive global-student anchor;
- student-only federated payload;
- support-weighted canonical aggregation;
- Prototype-Rank rejection in `src/openset/prototype_rank_pipeline.py`;
- MSP, Energy, positive-only, positive+boundary raw, and full-rank detector variants for A4;
- known-only disjoint prototype-fit / threshold-calibration protocol;
- known-only learned preprocessing;
- canonical BNaT taxonomy handling;
- A5 feature-source gate: a real trainable student OSR branch exists in source, but it is disabled by default until the ablation justifies it;
- old RL/DQN active implementation paths removed from the canonical source tree.

## 3. Tracking and local result persistence

The old LocalTracker/CompositeTracker/MLflow split was replaced by two independent concerns:

### Interactive tracker

`src/infrastructure/tracking/`

- `ExperimentTracker`
- `WandBTracker`
- `NullTracker`
- tracker factory

W&B supports `online`, `offline`, and `disabled` run modes through project configuration.

Scientific/model code does not import W&B directly.

### Durable scientific results

`src/experiment/result_store.py`

Local canonical run files remain the publication/reproducibility source of truth. W&B is not required for scientific correctness.

## 4. FedTROS internal plotting removal

Removed from the active FedTROS workflow:

- `src/plotting/`;
- old plotting launch scripts;
- scalability plotting script;
- plotting Hydra group/config;
- Matplotlib/Seaborn direct project dependencies;
- Matplotlib imports from federated-client scientific code;
- old fixed plot-data adapter.

Numeric calculations were preserved as structured results. A source guard now tests that plotting libraries do not re-enter canonical FedTROS source.

## 5. Separate plot-repository integration

The plots repository is now the only publication renderer.

FedTROS exports an immutable bundle using:

```bash
python scripts/export_publication_bundle.py \
  --outputs-dir outputs \
  --target-root publication_exports \
  --freeze-id <freeze-id>
```

The bundle contains a schema/version manifest, checksums, source run IDs, config/split hashes, aggregate statistics, and study-specific scientific data.

The plots repository validates the bundle before rendering:

```bash
python scripts/generate_all.py --bundle <bundle> --output-dir outputs/figures --strict
python scripts/verify_outputs.py --bundle <bundle> --figures-dir outputs/figures
```

The active figure registry is aligned to the Q1 experiment master plan and contains eight main-paper figures rather than the legacy 29-figure contract.

## 6. Experiment execution interface

Canonical study configs exist for:

- E0, E1, E2, E3, E4, E5, E6, E7, E8;
- A1, A2, A3, A4, A5;
- S1.

Headline seeds are 17, 42, 73, 101, 137.

Implemented CLI helpers:

- `scripts/doctor.py` — environment/pre-flight checks;
- `scripts/studies.py` — list/show declarative studies;
- `scripts/run.py` — exactly one resolved experiment cell;
- `scripts/run_study.py` — study matrix execution;
- `scripts/runs.py` — run-state inspection;
- `scripts/resume.py` — exact compatible resume;
- `scripts/evaluate.py` — evaluation entry point;
- `scripts/build_q1_results.py` — non-visual statistics/tables/provenance;
- `scripts/export_publication_bundle.py` — FedTROS -> plots contract;
- `scripts/import_legacy_results.py` — historical-result import boundary.

`run_study.py` supports dry-run, only-missing, resume, force-new, stage filters, method/dataset/alpha/client/variant filters, W&B mode, independent multi-GPU scheduling, and continue-on-error.

`run.py study=...` now resolves the supplied study plus selectors to exactly one declared matrix cell rather than merely relabeling a generic experiment.

Preprocessing output is materialized inside each immutable run as `outputs/runs/<run_id>/data/`. This removes the shared `data/processed` race that would otherwise allow parallel study cells to overwrite tensors/manifests. Matched methods remain scientifically paired through the study-level immutable partition file, whose schema validates seed, alpha/IID state, client count, known/unknown protocol, train size, and a SHA-256 hash of the deterministic training-label vector.

## 7. Communication and runtime instrumentation

E7 communication is measured from the actual NumPy model arrays transmitted by the Flower simulation path. The metric explicitly excludes protocol/TLS/framework headers.

Canonical per-round communication output:

`metrics/communication_round.csv`

contains downlink, uplink, round total, and cumulative bytes.

Per-client actual payload values are aggregated as communication totals, not sample-weighted learning metrics.

Runtime instrumentation now records:

- client fit critical-path time;
- private teacher time;
- student time;
- server aggregation time;
- open-set evaluation time;
- actual server round wall time;
- residual/orchestration time;
- cumulative round time.

Canonical per-round file:

`metrics/timing_round.csv`

Top-level whole-run phase timing is stored separately in `metrics/pipeline_timing.json`, so it cannot overwrite E6/E7 per-round timing.

## 8. Checkpoint and resume integrity

The new checkpoint contract uses schema version 2 with canonical method/teacher/config/git metadata.

Historical DQN teacher checkpoints fail as incompatible rather than being silently partially loaded.

Persistent private VCT state and RNG/config evidence are included where required for exact continuation.

## 9. Analysis and publication statistics

FedTROS contains a non-visual analysis layer for:

- run querying;
- compatibility checks;
- multi-seed mean/SD/95% CI;
- paired deltas;
- temporal-vs-seed variance separation;
- table generation;
- provenance.

The plots repository receives already-aggregated scientific values and does not become a second statistical source of truth.

## 10. Validation performed in this implementation environment

### Successful static/unit validation

- source compilation with `compileall`;
- no-internal-plotting tests;
- tracking-backend tests;
- study-contract tests;
- checkpoint-contract tests;
- publication-bundle-contract tests;
- analysis/aggregation/export tests;
- disjoint Prototype-Rank calibration test;
- coverage-adaptive anchor tests;
- VCT tests;
- teacher-student integration tests;
- local-training smoke test;
- scientific-core tests;
- preprocessing and external-dataset tests;
- RL-removal guard tests.

### Successful two-repository integration validation

A synthetic completed-run collection was passed through:

```text
FedTROS run loader
 -> build_q1_results
 -> publication bundle exporter
 -> separate plots bundle loader
 -> all eight Q1 figure generators
 -> output verifier
```

The plot verifier returned PASS for the eight-figure registry.

### Environment limitation

The current execution sandbox uses Python 3.13 and does not contain the target Hydra/Flower/W&B GPU environment. Network installation is unavailable here. Therefore a full Flower GPU training run and a live W&B online run were not executed in this sandbox.

The project itself targets Python >=3.11,<3.13. The stale pre-refactor Poetry lock was archived; regenerate the lock on the target Python 3.11/3.12 server before installation.

## 11. Required first commands on the target Linux GPU server

```bash
poetry lock
poetry install
poetry run python scripts/doctor.py --plots-repo ../plots --wandb-mode online
poetry run python scripts/run_study.py E0-VERIFY --stage smoke --dry-run
poetry run python scripts/run_study.py E0-VERIFY --stage smoke
```

Do not start the final five-seed grid until E0 and the A1/A2/A4/A5 architecture gates are complete.

## 12. Final Prototype-Rank cleanup before packaging

The remaining active EVT/Weibull fallback was removed from the canonical Prototype-Rank path before final packaging.

Changes:

- removed active `src/openset/evt.py`;
- removed EVT model fitting from `prototype_rank_pipeline.py`;
- removed EVT fallback decisions from final open-set evaluation;
- moved target-FPR/minimum-calibration settings under `calibration.*`;
- added a known-only empirical raw-prototype threshold for the A4 raw-score ablation;
- made open-set evaluation fail explicitly when a valid empirical-rank calibrator is unavailable;
- preserved MSP, Energy, raw-prototype, and Prototype-Rank scores as matched post-hoc ablation paths without EVT;
- retained the disjoint known-only prototype-fit/calibration contract.

Final dependency-independent test result in this packaging environment:

```text
96 passed
```

Two additional tests require target-environment dependencies unavailable in this sandbox (`hydra-core` and `flwr`) and therefore could not be collected here. They remain in the suite and should run after `poetry install` on Python 3.11/3.12.

The separate publication repository was also revalidated with a synthetic schema-correct publication bundle. All eight active Q1 main figures were generated and `verify_outputs.py` returned `PASS` with no missing outputs.
