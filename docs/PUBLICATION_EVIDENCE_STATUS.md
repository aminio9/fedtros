# Publication evidence status

The current plot bundle is not publication evidence. It contains historical or
provisional FedTROS-PR artifacts, mostly one-seed/two-client/two-round outputs,
while the paper defines FedTROS-MC and requires the canonical multi-seed matrix.

Run the gate before exporting figures:

```text
python scripts/validate_publication_evidence.py --runs-dir outputs/runs --report outputs/publication_evidence_status.json
```

The gate must pass before `scripts/export_publication_bundle.py` is used for a
paper freeze. Smoke/development runs and legacy PR results must remain diagnostic.
