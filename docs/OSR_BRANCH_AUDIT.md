# Student OSR Branch Audit

**Audit date:** 2026-08-19
**Verdict:** `DEDICATED_OSR_BRANCH = YES`, but it is **not canonical by default** until A5 earns its complexity.

## Acceptance criteria

The source contains a genuine optional student OSR branch because it has: (1) separate trainable student parameters, (2) a distinct OSR representation, (3) an explicit local OSR training objective, (4) checkpoint/federated student state when enabled, and (5) a Multicenter Conformal feature-source path that can consume the branch representation.

## Canonical pre-A5 behavior

The default FedTROS-MC configuration now sets:

```yaml
training:
  student_osr_enabled: false

open_set:
  prototype_rank:
    prototype:
      feature_source: student_embedding
```

Thus normal E1-E8 runs use the deterministic penultimate federated-student embedding for Multicenter Conformal. The optional branch is enabled only by the predeclared `A5-FEATURE` variant:

- `student_embedding`: branch disabled, PR uses the normalized student embedding.
- `osr_branch_embedding`: branch enabled/trained, PR uses `osr_mu`.

If A5 does not show a reproducible benefit in AUROC/AUPRC/Unknown-F1 without unacceptable known-utility or efficiency cost, the branch should be removed from the final canonical method.

## Naming rule

The branch is called the **optional student OSR branch**. Known-derived boundary samples must never be described as real or pseudo unknown attacks.
