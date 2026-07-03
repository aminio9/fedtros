#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "WARNING: run_full_suite.sh launches E1-E8 sequentially and is expensive."
echo "Run scripts/experiments/run_validation_tiny.sh and python scripts/cheap_validation.py first."

bash "$script_dir/e1_closed_set.sh"
bash "$script_dir/e2_open_set.sh"
bash "$script_dir/e3_federated_noniid.sh"
bash "$script_dir/e4_combined_open_set_noniid.sh"
bash "$script_dir/e5_multi_dataset_open_set_noniid.sh"
bash "$script_dir/e6_ablation.sh"
bash "$script_dir/e7_efficiency_scalability.sh"
bash "$script_dir/e8_labelwise_open_set.sh"
