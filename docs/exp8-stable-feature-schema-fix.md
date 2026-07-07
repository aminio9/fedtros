# Exp8 stable feature-schema fix

## Problem

In label-wise open-set Exp8, the held-out unknown label can remove one categorical value from the known training split. If the one-hot encoder is fitted only on `known_train`, the encoded feature matrix can shrink from the closed-set BNaT schema dimension `31` to `30`.

The previous metadata-sync fix made the runtime follow the processed tensors, so the crash disappeared. That is safe, but it is not the best protocol when the dataset has a fixed source feature schema and all experiments should use the same tabular feature layout.

## Correct solution

The numeric scaler still fits only on known training data.

The categorical one-hot vocabulary is now fitted using a stable, label-free source-schema scope:

```yaml
dataset:
  preprocessing:
    categorical_schema_scope: source
    expected_state_dim: 31
```

This keeps the BNaT feature layout stable across:

- closed-set Exp1 / Exp3,
- standard open-set Exp2 / Exp4,
- label-wise open-set Exp8.

The code does not append a fake open-set placeholder column. It preserves the real one-hot schema. If the raw CSV schema changes and the resulting dimension is not 31, preprocessing stops with a clear error instead of silently training an inconsistent model.

## Privacy and leakage note

Using `categorical_schema_scope=source` only defines the one-hot vocabulary of categorical feature values from the source CSV. It does not use unknown labels as supervision, does not fit numeric scaling statistics on unknown samples, and does not allow unknown samples into training, prototype fitting, rank calibration, EVT fitting, or threshold selection.

For the strictest inductive open-set protocol, set:

```yaml
dataset.preprocessing.categorical_schema_scope=known_train
dataset.preprocessing.expected_state_dim=null
```

That strict setting may legitimately produce `state_dim=30` for some label-wise runs. For your BNaT paper runs, use the fixed source schema so every label-wise Exp8 run keeps the expected `state_dim=31`.

## Expected Exp8 behavior

```text
Categorical schema scope | scope=source
Preprocessing complete | state_dim=31 | num_actions=4
Initialized BlockchainIntrusionEnv ... dim=31
Student model initialized ... input_dim=31 | num_classes=4
```
