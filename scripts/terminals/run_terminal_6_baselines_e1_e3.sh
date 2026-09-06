#!/usr/bin/env bash
# ==============================================================================
# Terminal 6: Baselines for Core Closed & Open-Set (E1, E2, E3)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../.." && pwd))"
cd "${ROOT_DIR}"

mkdir -p logs
LOG_FILE="${ROOT_DIR}/logs/terminal_6_baselines_e1_e3.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================================================="
echo "🚀 [Terminal 6/8] Starting Core Baselines (E1, E2, E3)"
echo "📅 Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📁 Root Directory: ${ROOT_DIR}"
echo "📝 Log File: ${LOG_FILE}"
echo "=============================================================================="

export PATH="$HOME/.local/bin:$PATH"

echo "📌 [1/3] Running E1-IID-CS Baselines on B-NaT and B-TAT (fedavg, fedprox, scaffold, local_only, centralized)..."
poetry run python scripts/run_study.py E1-IID-CS \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedavg fedprox scaffold local_only centralized \
    --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [2/3] Running E2-IID-OSR Baselines on B-NaT..."
poetry run python scripts/run_study.py E2-IID-OSR \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedavg fedprox scaffold local_only centralized \
    --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [3/3] Running E3-NIID-CS Baselines across all 3 Dirichlet alphas: [1.0, 0.5, 0.1]..."
poetry run python scripts/run_study.py E3-NIID-CS \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedavg fedprox scaffold local_only centralized \
    --only-missing --output-dir outputs runtime=gpu_fast

echo ""
echo "=============================================================================="
echo "🎉 [Terminal 6/8] E1, E2, E3 Baselines COMPLETED SUCCESSFULLY!"
echo "📅 Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
