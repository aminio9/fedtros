# FedTROS remediation ledger

This ledger tracks the method-consistency and experimental-reproducibility remediation requested for the FedTROS-MC paper.

| Section | Status | Commit | Verification |
|---|---|---|---|
| Baseline inventory | completed | c3f6e8e | Recorded current dirty tree, run inventory, and failing tests |
| Canonical method/configuration | completed | b97986f | Canonical training flag propagated; focused tests passed |
| Multicenter conformal/provenance/metrics | completed | aa7f281 | Disjoint K selection, pooled covariance fallback, metrics, and split provenance |
| Study/export contracts | completed | 126524c | FedTROS-MC identities, A4/A5 routing, stage validation, and schema fix |
| Plot-data publication gate | completed | 06df643 | Added fail-fast evidence validator and gated publication export |
| Final verification | in progress | — | Full test suite, clean status, and commit review |

## Operating rules

- Existing user changes are preserved unless a change is necessary for a listed defect.
- Each completed section receives its own focused commit.
- No result is treated as publication evidence until it is linked to a completed, canonical, multi-seed run.
- Smoke and development runs are engineering checks only.

## Baseline (2026-08-30)

- Current implementation tree contains pre-existing uncommitted refactor work.
- Completed current runs: four smoke cells (two clients, two rounds, seed 42) plus one unmanifested directory.
- Automated tests before remediation: 150 passed, 1 failed (publication-bundle schema mismatch).
