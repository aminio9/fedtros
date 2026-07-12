# E8 Label-Wise Open-Set Runbook

## Objective

Run a label-wise open-set stress test where one non-Normal traffic label is held out as unknown. Normal is always kept as known.

## Correct DKD-FedOS/Fed-DiGOS Contract

Exp8 is an open-set experiment, not a closed-set Exp3-style report. The run must:

- train only on known labels,
- put the held-out label only in `shared_open_set_test.pt`,
- enable Fed-DiGOS open-set evaluation,
- evaluate the aggregated global student after federated training,
- save `open_set_metrics.json` and `open_set_scores.csv`.

The Exp8 config now explicitly enables:

```yaml
training.dkd_student_osr_enabled: true
training.dkd_student_open_set_enabled: true
training.generator.enabled: false
open_set.evt.enabled: true
open_set.evt.backend: fed_digos
open_set.fed_digos.enabled: true
open_set.fed_digos.score_fusion.method: prototype_rank
open_set.fed_digos.proser.enabled: false

`prototype_rank` is retained as the configuration name for compatibility. It
now selects the paper-style PNPFF score fitted after global student aggregation;
it is not the historical post-hoc KMeans empirical-rank detector. The evaluator
also writes `pnpff_state.pt` and `pnpff_metadata.json`.
evaluation.mode: open_set
```

## Execution Commands

```bash
poetry run python run.py experiment=exp8 +method=dkd_fedos seed=42 \
  dataset.known_labels=[Normal,DoS,MitM,FoT] \
  tracking.run_id=e8_bp_dkd_fedos_seed42
```

```bash
poetry run python run.py experiment=exp8 +method=dkd_fedos seed=42 \
  dataset.known_labels=[Normal,BP,MitM,FoT] \
  tracking.run_id=e8_dos_dkd_fedos_seed42
```

```bash
poetry run python run.py experiment=exp8 +method=dkd_fedos seed=42 \
  dataset.known_labels=[Normal,BP,DoS,FoT] \
  tracking.run_id=e8_mitm_dkd_fedos_seed42
```

```bash
poetry run python run.py experiment=exp8 +method=dkd_fedos seed=42 \
  dataset.known_labels=[Normal,BP,DoS,MitM] \
  tracking.run_id=e8_fot_dkd_fedos_seed42
```

Or run all:

```bash
bash scripts/experiments/e8_labelwise_open_set.sh
```

## Expected Log Lines

You should see:

```text
DKD-FedOS open-set final evaluation requested
FED-DIGOS OPEN-SET ACTIVE
calibration_unknown=0
open_test_unknown>0
Fed-DiGOS evaluation | selected=prototype_rank
```

## Expected Outputs

- `open_set_metrics.json`
- `open_set_scores.csv`
- `fed_digos_component_aurocs.json`
- `fed_digos_prototypes.json`
- `fed_digos_rank_calibration.json`
- `known_unknown_score_quantiles.json`
- `score_overlap_report.json`
- `latent_embeddings.csv`

## Common Failure

If Exp8 only prints closed-set shared-test reports and never prints `FED-DIGOS OPEN-SET ACTIVE`, then the run did not execute final Fed-DiGOS evaluation. Check that `evaluation.mode=open_set`, `open_set.evt.enabled=true`, `open_set.evt.backend=fed_digos`, and `open_set.fed_digos.enabled=true` are active in `resolved_config.yaml`.
