# Environment setup

FedTROS targets Python `>=3.11,<3.13` and a CUDA-capable PyTorch environment for final GPU runs.

The pre-refactor lock file was intentionally archived because it represented obsolete direct dependencies (including internal plotting) and did not contain the new W&B dependency. On the supported server environment regenerate it:

```bash
poetry lock
poetry install
```

Then verify:

```bash
poetry run python scripts/doctor.py --plots-repo ../plots
```

Do not use the archived lock file for new VCT publication runs.
