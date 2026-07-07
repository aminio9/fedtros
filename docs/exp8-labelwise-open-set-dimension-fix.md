# Exp8 label-wise open-set dimension contract fix

Exp8 performs leave-one-label-out open-set experiments. In this protocol the preprocessing pipeline is fitted on the selected known labels only. Because categorical one-hot columns are learned from that known-only split, the final tensor feature dimension can change between held-out labels.

Example failure before the fix:

```text
Preprocessing complete | state_dim=30
Initialized BlockchainIntrusionEnv ... dim=30
ValueError: State Dim mismatch on client 1: Config(31) != Env(30)
```

This is not an open-set placeholder feature problem. It is a config/data contract problem: the processed tensor dataset is correct, while the default model config was still hard-coded to 31.

## Correct contract

The code must not force Exp8 back to 31. The processed dataset metadata is the source of truth. After preprocessing, the runtime config synchronizes:

```text
model.state_dim
model.transformer.input_dim
env_metadata.state_dim
```

from `processed/preprocess_metadata.json` before any model, student, server reference, client, or environment is initialized.

If the processed data says `state_dim=30`, the model and environment must both run with 30. If another label-wise split keeps all one-hot categories and says `state_dim=31`, both must run with 31.

## Correct Exp8 command

```bash
poetry run python run.py experiment=exp8 +method=dkd_fedos seed=42 \
  open_set.evt.backend=fed_digos \
  open_set.fed_digos.enabled=true \
  open_set.fed_digos.score_fusion.method=prototype_rank \
  training.dkd_student_osr_enabled=true \
  training.dkd_student_open_set_enabled=true \
  training.generator.enabled=false \
  dataset.known_labels=[Normal,BP,DoS,FoT] \
  tracking.run_id=e8_mitm_dkd_fedos_seed42
```

Expected fixed behavior:

```text
Preprocessing complete | state_dim=30 | num_actions=4
Model state_dim synchronized from preprocessing metadata | config=31 processed=30
Initialized BlockchainIntrusionEnv ... dim=30 | actions=4
Fed-DiGOS student initialized ... input_dim=30 | num_classes=4
```

## Open-set leakage rule

The held-out unknown label must be absent from all training and calibration files. It may appear only in `open_set_test.pt` with unknown label id.
