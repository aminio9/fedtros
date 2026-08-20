# FedTROS-PR experiment running guide

## 1. Pre-flight

```bash
poetry run python scripts/doctor.py --plots-repo ../plots --wandb-mode online
poetry run python scripts/studies.py list
```

The doctor checks Python, core dependencies, CUDA, W&B mode/auth, output writability, the canonical study set, absence of internal plotting imports, and the separate plots-repository integration files.

## 2. Experiment lifecycle

The canonical lifecycle is:

```text
E0 verification
 -> one-seed engineering pilot (seed 42)
 -> A1/A2/A4 (+A5 only if relevant) architecture gates
 -> configuration/code freeze
 -> five-seed headline studies
 -> E5/E6/E7/S1
 -> build statistics
 -> export immutable publication bundle
 -> render in plots repository
```

### Required architecture gates

- **A1:** no teacher / deterministic teacher / VCT (plus beta=0 control in the study definition).
- **A2:** no anchor / fixed anchor / coverage-adaptive anchor.
- **A4:** MSP / Energy / positive-only prototype / boundary raw / full Prototype-Rank.
- **A5:** only when testing whether the real dedicated OSR branch earns its complexity versus the ordinary student embedding.

## 3. Dry-run a matrix

```bash
poetry run python scripts/run_study.py E3-NIID-CS --stage paper_final --dry-run
```

Filter a matrix:

```bash
poetry run python scripts/run_study.py E3-NIID-CS \
  --stage paper_final \
  --method fedtros_pr \
  --alpha 0.1 \
  --seeds 17 42 \
  --dry-run
```

## 4. Launch only unfinished cells

```bash
poetry run python scripts/run_study.py E4-NIID-FOSR \
  --stage paper_final \
  --only-missing \
  --continue-on-error
```

Use `--force-new` only when a genuinely new run identity is required. Do not use it to hide configuration mistakes.

## 5. GPU scheduling

One GPU:

```bash
CUDA_VISIBLE_DEVICES=0 poetry run python scripts/run_study.py E4-NIID-FOSR --only-missing
```

Several independent GPUs:

```bash
poetry run python scripts/run_study.py E4-NIID-FOSR \
  --gpus 0 1 \
  --max-parallel 2 \
  --only-missing
```

## 6. W&B offline execution

```bash
poetry run python scripts/run_study.py E6-SCALE --wandb-mode offline --only-missing
```

The local scientific run directory remains complete whether W&B is online, offline, or disabled.

Each run also receives its own `outputs/runs/<run_id>/data/` preprocessing directory. This avoids shared-`data/processed` races when study cells are launched concurrently on multiple GPUs. Reproducible cross-method pairing is provided by the immutable partition manifest selected by the study runner, not by sharing generated tensor files between runs.

## 7. Run inspection

```bash
poetry run python scripts/runs.py summary
poetry run python scripts/runs.py failed --study E4-NIID-FOSR
poetry run python scripts/runs.py resumable
poetry run python scripts/runs.py missing E8-LOAO --stage paper_final
```

## 8. Resume

```bash
poetry run python scripts/resume.py RUN_ID
```

Resume is rejected when checkpoint schema, VCT teacher type, config hash, or required private client state does not match.

## 9. Final paper result pipeline

```bash
poetry run python scripts/build_q1_results.py --outputs-dir outputs --target paper_results --stage paper_final ablation reproduction
poetry run python scripts/export_publication_bundle.py \
  --outputs-dir outputs \
  --target-root publication_exports \
  --freeze-id fedtros-pr-vct-paper-final-01 \
  --include-stages paper_final ablation reproduction
```

The export is immutable: reusing an existing freeze ID fails rather than overwriting it.

## 10. First setup on a Linux GPU server

Use Python 3.11 or 3.12 for the final environment.

```bash
cd fedtros
poetry env use python3.12
poetry lock
poetry install
```

For online W&B tracking:

```bash
poetry run wandb login
```

Do not commit the W&B API key to YAML, shell scripts, or Git.

Then run the pre-flight checker:

```bash
poetry run python scripts/doctor.py --plots-repo ../plots --wandb-mode online
```

## 11. Required E0 gate

Before any expensive publication run:

```bash
poetry run python scripts/run_study.py E0-VERIFY --stage smoke --dry-run
poetry run python scripts/run_study.py E0-VERIFY --stage smoke --wandb-mode online
```

Do not start the five-seed grid until E0 passes and the A1/A2/A4 architecture gates are resolved.

## 12. Five-seed paper-final examples

E3 non-IID closed-set:

```bash
poetry run python scripts/run_study.py E3-NIID-CS \
  --stage paper_final \
  --seeds 17 42 73 101 137 \
  --only-missing \
  --wandb-mode online
```

E4 non-IID federated open-set:

```bash
poetry run python scripts/run_study.py E4-NIID-FOSR \
  --stage paper_final \
  --seeds 17 42 73 101 137 \
  --only-missing \
  --wandb-mode online
```

E8 leave-one-attack-out:

```bash
poetry run python scripts/run_study.py E8-LOAO \
  --stage paper_final \
  --seeds 17 42 73 101 137 \
  --only-missing \
  --wandb-mode online
```

## 13. Render the publication figures in the separate plots repository

FedTROS first exports the immutable scientific bundle:

```bash
poetry run python scripts/export_publication_bundle.py \
  --outputs-dir outputs \
  --target-root publication_exports \
  --freeze-id fedtros-pr-vct-paper-final-01 \
  --include-stages paper_final ablation reproduction
```

Then, in the separate repository:

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

The plotting repository never trains models and never recomputes the canonical multi-seed statistics.
