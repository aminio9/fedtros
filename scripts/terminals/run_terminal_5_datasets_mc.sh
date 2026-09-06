#!/usr/bin/env bash
# ==============================================================================
# Terminal 5: Dataset Generalization, Scale, LOAO, Sensitivity (E5, E6, E7, E8, S1)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../.." && pwd))"
cd "${ROOT_DIR}"

mkdir -p logs
LOG_FILE="${ROOT_DIR}/logs/terminal_5_datasets_mc.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================================================="
echo "🚀 [Terminal 5/8] Starting Dataset, Scale, LOAO, Sensitivity (FedTROS-MC)"
echo "📅 Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📁 Root Directory: ${ROOT_DIR}"
echo "📝 Log File: ${LOG_FILE}"
echo "=============================================================================="

export PATH="$HOME/.local/bin:$PATH"

echo "📌 [1/5] Running E5-DATASET (B-NAT, B-TAT, ToN-IoT, CIC-IDS2017) with FedTROS-MC at alpha 0.5..."
poetry run python scripts/run_study.py E5-DATASET \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [2/5] Running E6-SCALE (Client Scalability: 10, 50, 100 clients at alpha 0.5)..."
poetry run python scripts/run_study.py E6-SCALE \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [3/5] Running E7-EFFICIENCY (FL Profiling: Comm / Memory / Runtime at alpha 0.5)..."
poetry run python scripts/run_study.py E7-EFFICIENCY \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [4/5] Running E8-LOAO (Leave-One-Attack-Out: BP, DoS, MitM, FoT at alpha 0.5)..."
poetry run python scripts/run_study.py E8-LOAO \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo "📌 [5/5] Running S1-SENSITIVITY (Teacher KL Beta Sensitivity at alpha 0.5)..."
poetry run python scripts/run_study.py S1-SENSITIVITY \
    --stage main --wandb-mode disabled --seeds 42 \
    --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

echo ""
echo "=============================================================================="
echo "🎉 [Terminal 5/8] E5, E6, E7, E8, S1 FedTROS-MC COMPLETED SUCCESSFULLY!"
echo "📅 Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
