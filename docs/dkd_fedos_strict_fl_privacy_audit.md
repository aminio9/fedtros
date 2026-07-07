# DKD-FedOS Strict-FL Privacy Audit

## Result

The previous implementation did not send raw samples or per-sample labels, but it did send aggregated label-distribution metadata in Flower metrics. This fix removes those server-bound metrics and changes aggregation so the server no longer depends on label coverage or class entropy.

## Removed from client-to-server metrics

```text
label_histogram
true_label_histogram
action_histogram
per_class_policy_accuracy
class_entropy
label_coverage
digos_class_entropy
digos_label_coverage
missing_classes
present_classes
imbalance_ratio
min_class_count
max_class_count
coverage_quality
```

## Still allowed locally on clients

Clients still compute local class information internally for training stability:

```text
class-balanced CE,
class-balanced reward,
absent-class/global-anchor protection,
private local debugging.
```

These values are sanitized before Flower returns metrics to the server.

## Server aggregation after fix

Old label-aware aggregation:

```text
sqrt(n_i / n_max) * label_coverage_i * class_entropy_i
```

New strict-FL aggregation:

```text
sqrt(n_i / n_max)
```

The OSR branch also stops filtering by label coverage/class entropy and uses sample count plus label-free OSR quality metrics.

## Expected tradeoff

Privacy story becomes stronger. Non-IID performance may be slightly lower than the label-aware version because the server can no longer explicitly downweight low-class-coverage clients. The local protections remain active: class-balanced CE, teacher-to-student KD, feature alignment, global anchor, and prototype-rank OSR.

## Validation

Static validation run:

```bash
python -m compileall -q src tests
```

passed in the sandbox.
