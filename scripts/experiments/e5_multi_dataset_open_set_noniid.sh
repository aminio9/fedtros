#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper. Dataset selection and the frozen DKD-FedOS contract
# now live in the cross-platform Python launcher.
poetry run python scripts/run_exp5.py --datasets all --seed 42 "$@"
