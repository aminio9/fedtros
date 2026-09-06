#!/usr/bin/env bash
# ==============================================================================
# Terminal 3: Ablation Studies (A1–A5)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs

LOG_FILE="${ROOT_DIR}/logs/terminal_3_ablations.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================================================="
echo "🚀 [Terminal 3] Starting Ablation Studies (A1–A5)"
echo "📅 Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📁 Root Directory: ${ROOT_DIR}"
echo "📝 Log File: ${LOG_FILE}"
echo "=============================================================================="

# Pre-flight environment validation
echo "🔍 Validating environment..."
if ! command -v poetry &>/dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v poetry &>/dev/null; then
    echo "❌ Error: poetry is not installed or not in PATH."
    exit 1
fi

echo "✅ Python: $(poetry run python --version)"
echo "✅ GPU Check:"
poetry run python -c "import torch; print('  CUDA Available:', torch.cuda.is_available()); print('  Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ------------------------------------------------------------------------------
# Ablation Studies
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "▶️ Executing Ablation Matrix A1 to A5"
echo "=============================================================================="

echo "📌 [1/5] Running A1-TEACHER (Variational Teacher Ablation)..."
poetry run python scripts/run_study.py A1-TEACHER \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [2/5] Running A2-ANCHOR (Retention Anchor & Coverage Ablation)..."
poetry run python scripts/run_study.py A2-ANCHOR \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [3/5] Running A3-TRANSFER (Knowledge Transfer & Aligner Ablation)..."
poetry run python scripts/run_study.py A3-TRANSFER \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [4/5] Running A4-PR (Multi-Center Geometry vs Fixed Prototype-Rank)..."
poetry run python scripts/run_study.py A4-PR \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [5/5] Running A5-FEATURE (Representation Depth & ConFID Dual-Score Ensemble)..."
poetry run python scripts/run_study.py A5-FEATURE \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo ""
echo "=============================================================================="
echo "🎉 [Terminal 3] ALL ABLATION STUDIES COMPLETED SUCCESSFULLY!"
echo "📅 Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
