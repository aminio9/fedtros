#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "$script_dir/e1_closed_set.sh"
bash "$script_dir/e2_open_set.sh"
bash "$script_dir/e3_federated_noniid.sh"
bash "$script_dir/e4_combined_open_set_noniid.sh"
bash "$script_dir/e5_ablation.sh"
bash "$script_dir/e6_efficiency_scalability.sh"
