# FedTROS remediation ledger

This ledger tracks the method-consistency and experimental-reproducibility remediation requested for the FedTROS-MC paper.

| Section | Status | Commit | Verification |
|---|---|---|---|
| Baseline inventory | in progress | — | Record current dirty tree, run inventory, and failing tests |
| Canonical method/configuration | pending | — | Unit tests for canonical coverage and matched method resolution |
| Multicenter conformal/provenance/metrics | pending | — | Conformal protocol, artifact, leakage, and metric tests |
| Study/export contracts | pending | — | Study validation, ablation routing, and publication bundle tests |
| Plot-data publication gate | pending | — | Reject unsupported historical/incomplete data; validate one frozen bundle |
| Final verification | pending | — | Full test suite, clean status, and commit review |

## Operating rules

- Existing user changes are preserved unless a change is necessary for a listed defect.
- Each completed section receives its own focused commit.
- No result is treated as publication evidence until it is linked to a completed, canonical, multi-seed run.
- Smoke and development runs are engineering checks only.

## Baseline (2026-08-30)

- Current implementation tree contains pre-existing uncommitted refactor work.
- Completed current runs: four smoke cells (two clients, two rounds, seed 42) plus one unmanifested directory.
- Automated tests before remediation: 150 passed, 1 failed (publication-bundle schema mismatch).
