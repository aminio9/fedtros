#!/usr/bin/env bash
# ==============================================================================
# Terminal 1: Core Closed & Open-Set Experiments (E1, E2, E3)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs

LOG_FILE="${ROOT_DIR}/logs/terminal_1_core.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================================================="
echo "🚀 [Terminal 1] Starting Core Studies (E1, E2, E3)"
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
echo "▶️ [Phase 1/2] Running FedTROS-MC on E1, E2, E3"
echo "=============================================================================="

echo "📌 [1/3] Running E1-IID-CS (Closed-Set IID Baseline)..."
poetry run python scripts/run_study.py E1-IID-CS \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [2/3] Running E2-IID-OSR (Open-Set IID with ConFID Ensemble)..."
poetry run python scripts/run_study.py E2-IID-OSR \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [3/3] Running E3-NIID-CS across all 3 Dirichlet alphas: [1.0, 0.5, 0.1]..."
poetry run python scripts/run_study.py E3-NIID-CS \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

# ------------------------------------------------------------------------------
# Phase 2: Matched Baselines
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "▶️ [Phase 2/2] Running Matched Baselines for E1, E2, E3"
echo "=============================================================================="

echo "📌 Running E1-IID-CS Baselines (fedavg, fedprox, scaffold, local_only, centralized)..."
poetry run python scripts/run_study.py E1-IID-CS \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedavg fedprox scaffold local_only centralized \
    --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 Running E2-IID-OSR Baselines..."
poetry run python scripts/run_study.py E2-IID-OSR \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedavg fedprox scaffold local_only centralized \
    --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 Running E3-NIID-CS Baselines across all 3 Dirichlet alphas..."
poetry run python scripts/run_study.py E3-NIID-CS \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedavg fedprox scaffold local_only centralized \
    --only-missing --output-dir outputs runtime=gpu_fast

echo ""
echo "=============================================================================="
echo "🎉 [Terminal 1] ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "📅 Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
