# FedTROS remediation ledger

This ledger tracks the method-consistency and experimental-reproducibility remediation requested for the FedTROS-MC paper.

| Section | Status | Commit | Verification |
|---|---|---|---|
| Baseline inventory | completed | c3f6e8e | Recorded current dirty tree, run inventory, and failing tests |
| Canonical method/configuration | completed | b97986f | Canonical training flag propagated; focused tests passed |
| Multicenter conformal/provenance/metrics | completed | aa7f281 | Disjoint K selection, pooled covariance fallback, metrics, and split provenance |
| Study/export contracts | completed | 126524c | FedTROS-MC identities, A4/A5 routing, stage validation, and schema fix |
| Plot-data publication gate | completed | 06df643 | Added fail-fast evidence validator and gated publication export |
| Canonical metric/reporting contract | completed | fa758f8 | KFR alias, FedTROS-MC reporting labels, and canonical loader normalization |
| Baseline integration | completed | 2a11705 | SCAFFOLD/local-only server-client dispatch; centralized import fix; 29 focused tests passed |
| Final verification | completed | 8230e68 | Full suite: 151 passed; evidence gate remains intentionally red until paper runs exist |
| Final-test provenance | completed | 35713d2 | Added deterministic final-test identifier artifact with manifest path/hash; provenance test passed |
| Paper-stage protocol gate | completed | 1789abb | Publication stages now enforce 100 rounds/10 clients for all non-scalability studies; ablations use five seeds |
| A4/A5 split traceability | completed | 1327f75 | Added row-level prototype-fit/prototype-validation artifact and checksum; conformal contract tests passed |
| Matched baseline matrix | completed | 2d399e9 | Added centralized method config and exposed FedAvg/FedProx/SCAFFOLD/local-only/centralized in core studies |

## Operating rules

- Existing user changes are preserved unless a change is necessary for a listed defect.
- Each completed section receives its own focused commit.
- No result is treated as publication evidence until it is linked to a completed, canonical, multi-seed run.
- Smoke and development runs are engineering checks only.

## Baseline (2026-08-30)

- Current implementation tree contains pre-existing uncommitted refactor work.
- Completed current runs: four smoke cells (two clients, two rounds, seed 42) plus one unmanifested directory.
- Automated tests before remediation: 150 passed, 1 failed (publication-bundle schema mismatch).
