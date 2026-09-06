#!/usr/bin/env bash
# ==============================================================================
# Terminal 2: Dataset Generalization & Efficiency (E5, E7)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs

LOG_FILE="${ROOT_DIR}/logs/terminal_2_dataset.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================================================="
echo "🚀 [Terminal 2] Starting Dataset Generalization & Efficiency (E5, E7)"
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
# Phase 1: Canonical Method (FedTROS-MC)
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "▶️ [Phase 1/2] Running FedTROS-MC on E5, E7"
echo "=============================================================================="

echo "📌 [1/2] Running E5-DATASET across all declared datasets (B-NAT, B-TAT, ToN-IoT, CIC-IDS2017)..."
poetry run python scripts/run_study.py E5-DATASET \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [2/2] Running E7-EFFICIENCY (FL communication, runtime, memory profiling)..."
poetry run python scripts/run_study.py E7-EFFICIENCY \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

# ------------------------------------------------------------------------------
# Phase 2: Matched Baselines
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "▶️ [Phase 2/2] Running Matched Baselines for E5, E7"
echo "=============================================================================="

echo "📌 Running E5-DATASET Baselines..."
poetry run python scripts/run_study.py E5-DATASET \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedavg fedprox scaffold local_only centralized \
    --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 Running E7-EFFICIENCY Baselines..."
poetry run python scripts/run_study.py E7-EFFICIENCY \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedavg fedprox scaffold local_only centralized \
    --only-missing --output-dir outputs runtime=gpu_fast

echo ""
echo "=============================================================================="
echo "🎉 [Terminal 2] ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "📅 Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
