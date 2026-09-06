#!/usr/bin/env bash
# ==============================================================================
# Terminal 3: Core Architectural Ablations (A1-TEACHER, A2-ANCHOR, A3-TRANSFER)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../.." && pwd))"
cd "${ROOT_DIR}"

mkdir -p logs
LOG_FILE="${ROOT_DIR}/logs/terminal_3_ablations_a1_a3.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================================================="
echo "🚀 [Terminal 3/8] Starting Core Ablations (A1, A2, A3)"
echo "📅 Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📁 Root Directory: ${ROOT_DIR}"
echo "📝 Log File: ${LOG_FILE}"
echo "=============================================================================="

export PATH="$HOME/.local/bin:$PATH"

echo "📌 [1/3] Running A1-TEACHER (Variational Teacher Ablation across alphas 0.1 and 0.5)..."
poetry run python scripts/run_study.py A1-TEACHER \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [2/3] Running A2-ANCHOR (Retention Anchor & Coverage Ablation across alphas 0.1 and 0.5)..."
poetry run python scripts/run_study.py A2-ANCHOR \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [3/3] Running A3-TRANSFER (Knowledge Transfer & Aligner Ablation at alpha 0.5)..."
poetry run python scripts/run_study.py A3-TRANSFER \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo ""
echo "=============================================================================="
echo "🎉 [Terminal 3/8] A1, A2, A3 Ablations COMPLETED SUCCESSFULLY!"
echo "📅 Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
