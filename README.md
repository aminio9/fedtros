# FedTROS-PR

**FedTROS-PR — Federated Teacher-Regularized Open-Set Recognition with Prototype-Rank Rejection**

This repository is the scientific training/evaluation side of a two-repository research workflow. The private teacher is a supervised **Variational Classifier Teacher (VCT)** and the final known/unknown decision is **Prototype-Rank Rejection (PR)**. Only student parameters are federated. The separate `plots` repository renders publication figures from a versioned publication bundle; it never imports FedTROS Python modules.

## Architecture

```text
study YAML -> run_study.py / run.py -> FedTROS-PR train/evaluate
                                     |-> W&B (monitoring only)
                                     `-> outputs/runs/<run_id> (scientific source of truth)
                                                |
                                   build_q1_results.py
                                                |
                              export_publication_bundle.py
                                                |
                              separate plots repository
```

FedTROS deliberately contains **no publication plotting subsystem**. W&B provides live run monitoring; local structured results and manifests are the reproducibility/publication source of truth.

## Server setup

Use Python **3.11 or 3.12**. The migration changed direct dependencies (W&B added; internal Matplotlib/Seaborn removed), so the historical `poetry.lock` was archived and **must be regenerated on the target environment**:

```bash
cd fedtros
poetry lock
poetry install
```

For online W&B runs, authenticate outside source control:

```bash
wandb login
```

For an offline server, no login is required for the scientific pipeline; select `--wandb-mode offline` or `tracking.mode=offline` and sync later if desired.

## First command on a new server

```bash
poetry run python scripts/doctor.py --plots-repo ../plots --wandb-mode online
```

Use `--wandb-mode offline` or `disabled` when appropriate.

## Discover experiments

```bash
poetry run python scripts/studies.py list
poetry run python scripts/studies.py show E4-NIID-FOSR --stage paper_final
```

Canonical studies are E0–E8 plus A1–A5 and S1. Headline seeds are `17 42 73 101 137`.

## Safe dry run

```bash
poetry run python scripts/run_study.py E4-NIID-FOSR \
  --stage paper_final \
  --dry-run
```

A dry run does not start training.

## Run one exact study cell

`run.py` refuses ambiguous study requests. Supply enough dimensions to select one cell:

```bash
poetry run python scripts/run.py \
  study=E4-NIID-FOSR \
  stage=paper_final \
  method=fedtros_pr \
  alpha=0.5 \
  seed=42 \
  wandb_mode=online
```

For E8, also specify the held-out attack when needed:

```bash
poetry run python scripts/run.py \
  study=E8-LOAO stage=paper_final seed=42 unknown=MitM
```

## Run a complete five-seed paper study

```bash
poetry run python scripts/run_study.py E4-NIID-FOSR \
  --stage paper_final \
  --seeds 17 42 73 101 137 \
  --only-missing \
  --wandb-mode online
```

For two independent GPUs:

```bash
poetry run python scripts/run_study.py E3-NIID-CS \
  --stage paper_final \
  --gpus 0 1 \
  --max-parallel 2 \
  --only-missing
```

This schedules **independent runs** across devices; it does not change the scientific model into distributed multi-GPU training.

## E0 gate before expensive runs

```bash
poetry run pytest -q
poetry run python scripts/run_study.py E0-VERIFY \
  --stage smoke \
  --wandb-mode disabled
```

Do not launch publication runs until E0 and the required teacher/anchor/PR gates are satisfied.

## Inspect run state

```bash
poetry run python scripts/runs.py summary
poetry run python scripts/runs.py failed
poetry run python scripts/runs.py interrupted
poetry run python scripts/runs.py resumable
poetry run python scripts/runs.py missing E4-NIID-FOSR --stage paper_final
poetry run python scripts/runs.py show RUN_ID
```

## Resume an interrupted VCT run

```bash
poetry run python scripts/resume.py RUN_ID
```

Exact resume validates the schema/config hash and requires persistent private client VCT state when the teacher persists across rounds. Old DQN-era checkpoints are incompatible by design.

## Build paper statistics/tables

```bash
poetry run python scripts/build_q1_results.py \
  --outputs-dir outputs \
  --target paper_results \
  --stage paper_final ablation reproduction
```

This computes non-visual statistics and tables. It does **not** render figures.

## Export the plots-repository bundle

After the final freeze:

```bash
poetry run python scripts/export_publication_bundle.py \
  --outputs-dir outputs \
  --target-root publication_exports \
  --freeze-id fedtros-pr-vct-paper-final-01 \
  --include-stages paper_final ablation reproduction
```

Then, in the separate plots repository:

```bash
cd ../plots
python scripts/generate_all.py \
  --bundle ../fedtros/publication_exports/fedtros-pr-vct-paper-final-01 \
  --output-dir outputs/figures \
  --strict

python scripts/verify_outputs.py \
  --bundle ../fedtros/publication_exports/fedtros-pr-vct-paper-final-01 \
  --figures-dir outputs/figures
```

The plots repository validates bundle schema/checksums before rendering.

## W&B modes

- `online`: live dashboard and system/run monitoring.
- `offline`: local W&B run state plus complete canonical FedTROS outputs.
- `disabled`: no W&B dependency at runtime; canonical local results still complete.

Examples:

```bash
# Whole study offline
poetry run python scripts/run_study.py E3-NIID-CS --wandb-mode offline

# One exact run without W&B
poetry run python scripts/run.py study=E2-IID-OSR seed=42 wandb_mode=disabled
```

## Canonical output contract

Each run is rooted at:

```text
outputs/runs/<run_id>/
```

with resolved configuration, run/data/partition/feature/seed/model manifests, run-local `data/`, `metrics/`, `predictions/`, `artifacts/`, `checkpoints/`, and centralized logs. The `data/` directory is intentionally isolated per run so parallel study cells cannot overwrite one another's preprocessed tensors. Matched methods still reuse the same immutable paired-partition file for a given dataset/protocol/seed. Canonical tabular files use **CSV** so the core experiment pipeline does not depend on a Parquet engine.

See `docs/RUNNING_GUIDE.md`, `docs/WANDB_TRACKING.md`, and `docs/PUBLICATION_BUNDLE_CONTRACT.md` for details.
