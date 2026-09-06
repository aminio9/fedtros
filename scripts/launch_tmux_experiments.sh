#!/usr/bin/env bash
# ==============================================================================
# FedTROS-MC Master tmux Experiment Launcher & Controller
# ==============================================================================
# Usage:
#   ./scripts/launch_tmux_experiments.sh start    -> Start all 4 tmux sessions
#   ./scripts/launch_tmux_experiments.sh status   -> Check running status & progress
#   ./scripts/launch_tmux_experiments.sh attach N -> Attach to session N (1..4)
#   ./scripts/launch_tmux_experiments.sh stop     -> Kill all 4 tmux sessions
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs

SESSIONS=(
    "exp_e1_e3:scripts/run_terminal_1_core.sh:logs/terminal_1_core.log"
    "exp_e5_e7:scripts/run_terminal_2_dataset.sh:logs/terminal_2_dataset.log"
    "exp_ablations:scripts/run_terminal_3_ablations.sh:logs/terminal_3_ablations.log"
    "exp_heavy:scripts/run_terminal_4_heavy.sh:logs/terminal_4_heavy.log"
)

# Make all terminal scripts executable
chmod +x "${SCRIPT_DIR}"/run_terminal_*.sh || true

action="${1:-status}"

case "${action}" in
    start)
        echo "=============================================================================="
        echo "🚀 Launching all 4 tmux experiment sessions in background..."
        echo "=============================================================================="
        
        # Verify poetry and datasets
        if ! command -v tmux &>/dev/null; then
            echo "❌ tmux is not installed. Install via: sudo apt install -y tmux"
            exit 1
        fi
        
        for entry in "${SESSIONS[@]}"; do
            IFS=":" read -r name script log <<< "${entry}"
            if tmux has-session -t "${name}" 2>/dev/null; then
                echo "⚠️ Session '${name}' is ALREADY RUNNING. Skipping creation."
            else
                echo "▶️ Creating tmux session '${name}' executing '${script}'..."
                tmux new-session -d -s "${name}" "bash ${ROOT_DIR}/${script}"
                echo "   ✅ Started '${name}' (logs to ${log})"
            fi
        done
        echo ""
        echo "🎉 All sessions launched! Check status with:"
        echo "   ./scripts/launch_tmux_experiments.sh status"
        ;;

    status)
        echo "=============================================================================="
        echo "📊 FedTROS-MC tmux Experiment Status"
        echo "=============================================================================="
        idx=1
        for entry in "${SESSIONS[@]}"; do
            IFS=":" read -r name script log <<< "${entry}"
            if tmux has-session -t "${name}" 2>/dev/null; then
                status="🟢 RUNNING"
            else
                status="⚪ STOPPED / FINISHED"
            fi
            echo "Terminal ${idx} [${name}]: ${status}"
            if [[ -f "${log}" ]]; then
                last_line=$(tail -n 1 "${log}" 2>/dev/null || true)
                echo "   📄 Latest log: ${last_line}"
            else
                echo "   📄 Log: (No log file yet)"
            fi
            idx=$((idx + 1))
        done
        echo ""
        echo "💡 Quick Commands:"
        echo "   Attach to Terminal 1: tmux attach -t exp_e1_e3"
        echo "   Attach to Terminal 2: tmux attach -t exp_e5_e7"
        echo "   Attach to Terminal 3: tmux attach -t exp_ablations"
        echo "   Attach to Terminal 4: tmux attach -t exp_heavy"
        echo "   Live log tail:        tail -f logs/terminal_4_heavy.log"
        ;;

    attach)
        target="${2:-1}"
        case "${target}" in
            1|exp_e1_e3) tmux attach -t exp_e1_e3 ;;
            2|exp_e5_e7) tmux attach -t exp_e5_e7 ;;
            3|exp_ablations) tmux attach -t exp_ablations ;;
            4|exp_heavy) tmux attach -t exp_heavy ;;
            *) echo "Unknown session: ${target}. Use 1, 2, 3, or 4." ;;
        esac
        ;;

    stop)
        echo "⚠️ Killing all 4 experiment tmux sessions..."
        for entry in "${SESSIONS[@]}"; do
            IFS=":" read -r name script log <<< "${entry}"
            if tmux has-session -t "${name}" 2>/dev/null; then
                tmux kill-session -t "${name}"
                echo "🛑 Killed '${name}'"
            fi
        done
        echo "All sessions stopped."
        ;;

    *)
        echo "Usage: $0 {start|status|attach <1..4>|stop}"
        exit 1
        ;;
esac
