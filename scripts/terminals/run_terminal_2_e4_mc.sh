#!/usr/bin/env bash
# ==============================================================================
# Terminal 2: Central Open-Set Benchmark FedTROS-MC (E4 across 3 alphas)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../.." && pwd))"
cd "${ROOT_DIR}"

mkdir -p logs
LOG_FILE="${ROOT_DIR}/logs/terminal_2_e4_mc.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================================================="
echo "🚀 [Terminal 2/8] Starting E4-NIID-FOSR FedTROS-MC Benchmark"
echo "📅 Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📁 Root Directory: ${ROOT_DIR}"
echo "📝 Log File: ${LOG_FILE}"
echo "=============================================================================="

export PATH="$HOME/.local/bin:$PATH"

echo "📌 Running E4-NIID-FOSR across all 3 Dirichlet alphas: [1.0 (mild), 0.5 (moderate), 0.1 (extreme)]..."
echo "   (ConFID Dual-Path Ensemble: active by default | feature=student_hidden_l1)"
poetry run python scripts/run_study.py E4-NIID-FOSR \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo ""
echo "=============================================================================="
echo "🎉 [Terminal 2/8] E4 FedTROS-MC COMPLETED SUCCESSFULLY!"
echo "📅 Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
