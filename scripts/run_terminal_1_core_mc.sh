#!/usr/bin/env bash
# ==============================================================================
# Terminal 1: Core FedTROS-MC Canonical Runs (E1, E2, E3)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs
LOG_FILE="${ROOT_DIR}/logs/terminal_1_core_mc.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================================================="
echo "🚀 [Terminal 1/8] Starting Core FedTROS-MC Studies (E1, E2, E3)"
echo "📅 Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📁 Root Directory: ${ROOT_DIR}"
echo "📝 Log File: ${LOG_FILE}"
echo "=============================================================================="

export PATH="$HOME/.local/bin:$PATH"

echo "📌 [1/3] Running E1-IID-CS (FedTROS-MC)..."
poetry run python scripts/run_study.py E1-IID-CS \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [2/3] Running E2-IID-OSR (FedTROS-MC ConFID Ensemble)..."
poetry run python scripts/run_study.py E2-IID-OSR \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [3/3] Running E3-NIID-CS across all 3 Dirichlet alphas: [1.0, 0.5, 0.1] (FedTROS-MC)..."
poetry run python scripts/run_study.py E3-NIID-CS \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo ""
echo "=============================================================================="
echo "🎉 [Terminal 1/8] E1, E2, E3 FedTROS-MC COMPLETED SUCCESSFULLY!"
echo "📅 Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
