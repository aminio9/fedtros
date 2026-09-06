#!/usr/bin/env bash
# ==============================================================================
# Terminal 4: Scalability, LOAO, Sensitivity & E4 Benchmark (E4, E6, E8, S1)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs

LOG_FILE="${ROOT_DIR}/logs/terminal_4_heavy.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================================================="
echo "🚀 [Terminal 4] Starting Heavy Studies (E4, E6, E8, S1)"
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
echo "▶️ [Phase 1/2] Running FedTROS-MC on E4, E6, E8, S1"
echo "=============================================================================="

echo "📌 [1/4] Running E4-NIID-FOSR across all 3 Dirichlet alphas: [1.0, 0.5, 0.1]..."
echo "    (ConFID Dual-Path Ensemble: active by default | feature=student_hidden_l1)"
poetry run python scripts/run_study.py E4-NIID-FOSR \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [2/4] Running E6-SCALE (Scalability across 10, 20, 50, 100 clients)..."
poetry run python scripts/run_study.py E6-SCALE \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [3/4] Running E8-LOAO (Leave-One-Attack-Out Cross-Class OSR)..."
poetry run python scripts/run_study.py E8-LOAO \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [4/4] Running S1-SENSITIVITY (Tempered Aggregation Gamma Sensitivity)..."
poetry run python scripts/run_study.py S1-SENSITIVITY \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

# ------------------------------------------------------------------------------
# Phase 2: E4 Matched Baselines (15 runs = 5 methods x 3 alphas)
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "▶️ [Phase 2/2] Running E4-NIID-FOSR Matched Baselines across all 3 alphas"
echo "=============================================================================="

echo "📌 Running E4 Baselines (fedavg, fedprox, scaffold, local_only, centralized across 1.0, 0.5, 0.1)..."
poetry run python scripts/run_study.py E4-NIID-FOSR \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedavg fedprox scaffold local_only centralized \
    --only-missing --output-dir outputs runtime=gpu_fast

echo ""
echo "=============================================================================="
echo "🎉 [Terminal 4] ALL HEAVY EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "📅 Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
