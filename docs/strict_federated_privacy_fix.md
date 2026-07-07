# Strict FL Privacy Fix for DKD-FedOS / Fed-DiGOS

## Problem

The previous DKD-FedOS implementation did not send raw traffic samples or per-sample labels to the server, but it did upload aggregated label-distribution metadata in the Flower metrics dictionary, including:

```text
label_histogram
label_coverage
class_entropy
missing_classes
present_classes
imbalance_ratio
```

Those values are useful for debugging and label-aware reliability weighting, but they weaken the privacy claim. A stricter federated-learning statement should avoid sending any class-distribution summaries to the server.

## Fix

The client now sanitizes server-bound metrics before returning a Flower fit result. The client may still compute local label statistics internally for:

```text
class-balanced local loss,
global-anchor absent-class protection,
local debug logic,
OSR branch local training.
```

However, these class-distribution values are removed before upload. The server receives only:

```text
student model parameters / updates,
num_examples,
label-free training metrics and quality metrics.
```

The private local teacher is still never uploaded.

## Aggregation change

The old reliability score was label-aware:

```text
reliability_i = sqrt(n_i / n_max) * label_coverage_i * class_entropy_i
```

This has been replaced with strict-FL sample-support weighting:

```text
reliability_i = sqrt(n_i / n_max)
```

The OSR branch aggregation also no longer filters by label coverage or class entropy. It now filters only by:

```text
minimum sample count,
finite OSR update norm,
known-only OSR score gap.
```

## What changes in the method?

The method becomes more privacy-preserving but slightly less label-aware on the server side.

Still preserved:

```text
local class-balanced CE,
private local RL teacher,
teacher-to-student KD,
feature alignment,
global-anchor regularization,
known-only OSR training,
prototype-rank open-set detection.
```

Removed from server-side decision making:

```text
client label histogram,
client label coverage,
client class entropy,
client present/missing class lists,
client imbalance ratio.
```

## Paper-safe wording

Use this wording:

```text
Raw blockchain traffic samples and per-sample labels remain local to each client. The server receives only student model updates, the number of local examples required for standard federated weighting, and label-free training/quality metrics. Class histograms, label coverage, class entropy, and present/missing-class lists are not uploaded to the server.
```

Do not write:

```text
The server uses client class entropy and label coverage for aggregation.
```

Do not write:

```text
The server receives class-balanced labels.
```

## Expected effect

This fix strengthens the privacy story. Under severe non-IID, performance may drop slightly compared with label-aware reliability weighting because the server can no longer identify clients with low class coverage. The remaining non-IID protections are local: class-balanced CE, private teacher KD, feature alignment, and global-anchor loss.
