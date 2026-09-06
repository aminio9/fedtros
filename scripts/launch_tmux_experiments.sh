#!/usr/bin/env bash
# ==============================================================================
# FedTROS-MC 8-Terminal High-Throughput tmux Experiment Manager
# ==============================================================================
# Usage:
#   ./scripts/launch_tmux_experiments.sh start        -> Start all 8 tmux sessions
#   ./scripts/launch_tmux_experiments.sh status       -> Check running status & progress
#   ./scripts/launch_tmux_experiments.sh attach <1..8>-> Attach to session N
#   ./scripts/launch_tmux_experiments.sh stop         -> Kill all 8 tmux sessions
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs

SESSIONS=(
    "t1_core_mc:scripts/run_terminal_1_core_mc.sh:logs/terminal_1_core_mc.log:Core FedTROS-MC (E1, E2, E3)"
    "t2_e4_mc:scripts/run_terminal_2_e4_mc.sh:logs/terminal_2_e4_mc.log:E4 Open-Set FedTROS-MC (3 alphas)"
    "t3_ablations_a1_a3:scripts/run_terminal_3_ablations_a1_a3.sh:logs/terminal_3_ablations_a1_a3.log:Ablations (A1, A2, A3)"
    "t4_ablations_a4_a5:scripts/run_terminal_4_ablations_a4_a5.sh:logs/terminal_4_ablations_a4_a5.log:Ablations (A4, A5)"
    "t5_datasets_mc:scripts/run_terminal_5_datasets_mc.sh:logs/terminal_5_datasets_mc.log:Datasets, Scale, LOAO, Sensitivity (E5-E8, S1)"
    "t6_baselines_e1_e3:scripts/run_terminal_6_baselines_e1_e3.sh:logs/terminal_6_baselines_e1_e3.log:Baselines (E1, E2, E3)"
    "t7_baselines_e4:scripts/run_terminal_7_baselines_e4.sh:logs/terminal_7_baselines_e4.log:Baselines E4 (15 runs = 5 methods x 3 alphas)"
    "t8_baselines_e5_e7:scripts/run_terminal_8_baselines_e5_e7.sh:logs/terminal_8_baselines_e5_e7.log:Baselines E5 & E7 (Datasets & Efficiency)"
)

# Make all terminal scripts executable
chmod +x "${SCRIPT_DIR}"/run_terminal_*.sh || true

action="${1:-status}"

case "${action}" in
    start)
        echo "=============================================================================="
        echo "🚀 Launching all 8 parallel tmux experiment sessions (32 GB GPU Profile)..."
        echo "=============================================================================="
        
        if ! command -v tmux &>/dev/null; then
            echo "❌ tmux is not installed. Install via: sudo apt install -y tmux"
            exit 1
        fi
        
        idx=1
        for entry in "${SESSIONS[@]}"; do
            IFS=":" read -r name script log desc <<< "${entry}"
            if tmux has-session -t "${name}" 2>/dev/null; then
                echo "⚠️ Terminal ${idx} [${name}]: ALREADY RUNNING. Skipping creation."
            else
                echo "▶️ Launching Terminal ${idx} [${name}] -> ${desc}..."
                tmux new-session -d -s "${name}" "bash ${ROOT_DIR}/${script}"
                echo "   ✅ Started '${name}' (logging to ${log})"
            fi
            idx=$((idx + 1))
        done
        echo ""
        echo "🎉 All 8 sessions launched! Monitor them with:"
        echo "   ./scripts/launch_tmux_experiments.sh status"
        ;;

    status)
        echo "=============================================================================="
        echo "📊 FedTROS-MC 8-Terminal Experiment Status"
        echo "=============================================================================="
        idx=1
        for entry in "${SESSIONS[@]}"; do
            IFS=":" read -r name script log desc <<< "${entry}"
            if tmux has-session -t "${name}" 2>/dev/null; then
                status="🟢 RUNNING"
            else
                status="⚪ STOPPED / FINISHED"
            fi
            printf "Terminal %d [%-20s] %s | %s\n" "${idx}" "${name}" "${status}" "${desc}"
            if [[ -f "${log}" ]]; then
                last_line=$(tail -n 1 "${log}" 2>/dev/null || true)
                echo "   📄 Latest log: ${last_line}"
            else
                echo "   📄 Log: (No log file generated yet)"
            fi
            idx=$((idx + 1))
        done
        echo ""
        echo "💡 Quick Attach Commands:"
        echo "   tmux attach -t t1_core_mc           # Terminal 1"
        echo "   tmux attach -t t2_e4_mc             # Terminal 2"
        echo "   tmux attach -t t3_ablations_a1_a3   # Terminal 3"
        echo "   tmux attach -t t4_ablations_a4_a5   # Terminal 4"
        echo "   tmux attach -t t5_datasets_mc       # Terminal 5"
        echo "   tmux attach -t t6_baselines_e1_e3   # Terminal 6"
        echo "   tmux attach -t t7_baselines_e4       # Terminal 7"
        echo "   tmux attach -t t8_baselines_e5_e7   # Terminal 8"
        echo "   Or simply: ./scripts/launch_tmux_experiments.sh attach <1..8>"
        ;;

    attach)
        target="${2:-1}"
        case "${target}" in
            1|t1_core_mc) tmux attach -t t1_core_mc ;;
            2|t2_e4_mc) tmux attach -t t2_e4_mc ;;
            3|t3_ablations_a1_a3) tmux attach -t t3_ablations_a1_a3 ;;
            4|t4_ablations_a4_a5) tmux attach -t t4_ablations_a4_a5 ;;
            5|t5_datasets_mc) tmux attach -t t5_datasets_mc ;;
            6|t6_baselines_e1_e3) tmux attach -t t6_baselines_e1_e3 ;;
            7|t7_baselines_e4) tmux attach -t t7_baselines_e4 ;;
            8|t8_baselines_e5_e7) tmux attach -t t8_baselines_e5_e7 ;;
            *) echo "Unknown session: ${target}. Choose 1 to 8." ;;
        esac
        ;;

    stop)
        echo "⚠️ Stopping all 8 experiment tmux sessions..."
        for entry in "${SESSIONS[@]}"; do
            IFS=":" read -r name script log desc <<< "${entry}"
            if tmux has-session -t "${name}" 2>/dev/null; then
                tmux kill-session -t "${name}"
                echo "🛑 Killed session '${name}'"
            fi
        done
        echo "All sessions stopped."
        ;;

    *)
        echo "Usage: $0 {start|status|attach <1..8>|stop}"
        exit 1
        ;;
esac
