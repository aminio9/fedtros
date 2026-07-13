# E5 Multi-Dataset Open-Set Non-IID Validation

## Objective

Run updated DKD-FedOS/Fed-DiGOS independently on BTAT, ToN-IoT, and
CIC-IDS2017 using frozen open-set mappings and mild Dirichlet non-IID data.

## Preparation and Execution

```bash
poetry run python scripts/prepare_external_datasets.py --dataset all
poetry run python scripts/run_exp5.py --datasets all --seed 42 --smoke
poetry run python scripts/run_exp5.py --datasets all --seed 42
```

Select a subset with `--datasets btat`, `--datasets toniot`, or
`--datasets cicids2017`. Add `--profile full` to disable the deterministic
20,000-row-per-class experiment cap.

Copy-ready per-dataset commands with logging and status tracking are in
`docs/runbooks/e1_e5_multidataset_commands.txt`.

The preparer accepts the verified ToN-IoT file as either
`Train_Test_Network.csv` or `train_test_network.csv`. It uses multiclass `type`
as the target and removes binary `label`. For CIC-IDS2017, either place the
official ZIP at `data/raw/source/cicids2017/MachineLearningCSV.zip` or extract
its eight CSV files anywhere below `data/raw/source/cicids2017/`. Existing
canonical CSVs are reused only after their manifests and SHA-256 values pass.

## Frozen Contract

- seed 42, 10 clients, 100 rounds, 10 local episodes;
- known data split 70/10/20;
- Dirichlet alpha 0.5 and at least 32 training rows per client;
- unknown classes occur only in final open-set evaluation;
- Fed-DiGOS `prototype_rank`, PROSER disabled, student OSR enabled;
- private generator and student-to-teacher updates disabled.

## Validation

Confirm all runs have resolved configs, preprocessing manifests, client
histograms, checkpoints, open-set metrics, and logs. Literature values in
`docs/exp5_literature_context.md` are context only, not same-protocol rows.
