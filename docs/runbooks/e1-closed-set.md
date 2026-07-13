# E1 Closed-Set Runbook

## Objective

Run IID closed-set DKD-FedOS on every class of a selected dataset. E1 supports
`bnat`, `btat`, `toniot`, and `cicids2017`; external datasets resolve to 7, 10,
and 15 actions automatically.

## Commands

```bash
poetry run python run.py experiment=exp1 dataset=btat +method=dkd_fedos seed=42
poetry run python run.py experiment=exp1 dataset=toniot +method=dkd_fedos seed=42
poetry run python run.py experiment=exp1 dataset=cicids2017 +method=dkd_fedos seed=42
```

Copy-ready commands with logging and status tracking are in
`docs/runbooks/e1_e5_multidataset_commands.txt`.

## Contract

- all source labels are known;
- stratified 70/10/20 train/validation/test data;
- IID client partitioning;
- Fed-DiGOS, student OSR, student open-set head, and private generator disabled;
- state and action dimensions derived from preprocessing metadata.

## Validation

Confirm `resolved_config.yaml`, `preprocess_metadata.json`, `class_support.csv`,
`split_manifest.csv`, client tensors, checkpoints, and closed-set metrics exist.
