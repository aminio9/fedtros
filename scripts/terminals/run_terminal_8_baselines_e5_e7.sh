#!/usr/bin/env bash
# ==============================================================================
# Terminal 8: Baselines for Multi-Dataset & Efficiency (E5, E7)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../.." && pwd))"
cd "${ROOT_DIR}"

mkdir -p logs
LOG_FILE="${ROOT_DIR}/logs/terminal_8_baselines_e5_e7.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================================================="
echo "🚀 [Terminal 8/8] Starting E5 & E7 Baselines (Multi-Dataset & Efficiency)"
echo "📅 Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📁 Root Directory: ${ROOT_DIR}"
echo "📝 Log File: ${LOG_FILE}"
echo "=============================================================================="

export PATH="$HOME/.local/bin:$PATH"

echo "📌 [1/2] Running E5-DATASET Baselines across 4 datasets (B-NAT, B-TAT, ToN-IoT, CIC-IDS2017) at alpha 0.5..."
poetry run python scripts/run_study.py E5-DATASET \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedavg fedprox scaffold local_only centralized \
    --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [2/2] Running E7-EFFICIENCY Baselines (FL overhead comparison at alpha 0.5)..."
poetry run python scripts/run_study.py E7-EFFICIENCY \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedavg fedprox scaffold local_only centralized \
    --only-missing --output-dir outputs runtime=gpu_fast

echo ""
echo "=============================================================================="
echo "🎉 [Terminal 8/8] E5, E7 Baselines COMPLETED SUCCESSFULLY!"
echo "📅 Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
