# Complete `tmux` Guide for Remote Machine Learning & Simulations

`tmux` (Terminal Multiplexer) lets you run long-running processes inside persistent sessions on a remote Linux server. Even if your SSH connection drops, your experiments continue running uninterrupted.

---

## 1. Core Architecture

```
[tmux Server]
    └── Session (e.g., exp_e1_e3)
            ├── Window 0 (e.g., training)
            │       ├── Pane 0 (run_study.py)
            │       └── Pane 1 (watch nvidia-smi)
            └── Window 1 (e.g., logs/system)
                    └── Pane 0 (htop)
```

- **Session**: Persistent workspace that survives SSH disconnects.
- **Window**: Full-screen tab within a session (like browser tabs).
- **Pane**: Sub-terminal split inside a window (split horizontally or vertically).

---

## 2. The Prefix Key: `Ctrl + B`

Almost all tmux internal shortcuts start with the **Prefix Key**:
> **Press `Ctrl` and `B` together, release both, then immediately press the command key.**
> (Notation: `Prefix, <key>`)

---

## 3. Session Commands (From Terminal & Inside tmux)

### From the Bash / SSH Terminal:

| Action | Command |
| :--- | :--- |
| **Start new named session** | `tmux new -s <name>` |
| **List active sessions** | `tmux ls` |
| **Attach to session** | `tmux attach -t <name>` *(or `tmux a -t <name>`)* |
| **Attach to last session** | `tmux a` |
| **Kill a specific session** | `tmux kill-session -t <name>` |
| **Kill ALL tmux sessions** | `tmux kill-server` |

### Inside a tmux Session:

| Action | Shortcut |
| :--- | :--- |
| **Detach (leave running in background)** | `Prefix, d` |
| **Interactive session switcher** | `Prefix, s` *(arrow keys to select, `Enter`)* |
| **Rename current session** | `Prefix, $` |

---

## 4. Window Commands (Tabs inside a Session)

Windows allow you to keep multiple screens inside a single session without opening separate SSH connections.

| Action | Shortcut | Description |
| :--- | :--- | :--- |
| **Create new window** | `Prefix, c` | Creates a new blank terminal tab |
| **Rename current window** | `Prefix, ,` | Name it (e.g. `exp_e1`, `gpu_mon`) |
| **Next window** | `Prefix, n` | Move to next window tab |
| **Previous window** | `Prefix, p` | Move to previous window tab |
| **Switch by number** | `Prefix, 0` to `9` | Jump directly to window index |
| **Interactive window list** | `Prefix, w` | Visual tree of all sessions/windows |
| **Close current window** | `Prefix, &` | Confirms `(y/n)` before closing |

---

## 5. Pane Commands (Splitting Your Screen)

Panes let you split one window into multiple side-by-side or top-and-bottom terminals.

### Splitting:
| Action | Shortcut | Visual |
| :--- | :--- | :--- |
| **Split Vertically** *(Left / Right)* | `Prefix, %` | `[ Left \| Right ]` |
| **Split Horizontally** *(Top / Bottom)* | `Prefix, "` | `[ Top / Bottom ]` |

### Navigating & Managing Panes:
| Action | Shortcut | Description |
| :--- | :--- | :--- |
| **Navigate between panes** | `Prefix, Arrow Key` | Move focus Left, Right, Up, or Down |
| **Toggle Zoom (Fullscreen/Unzoom)** | `Prefix, z` | **Very useful!** Makes active pane full-screen; press again to restore split |
| **Show pane index numbers** | `Prefix, q` | Shows numbers on panes; press index to jump |
| **Cycle through layouts** | `Prefix, Space` | Automatically switches split arrangements |
| **Close current pane** | `Prefix, x` | Confirms `(y/n)` to kill pane (or just type `exit`) |

### Resizing Panes:
- `Prefix, :` to open command prompt, then type:
  - `resize-pane -D 5` *(expand downward 5 lines)*
  - `resize-pane -U 5` *(expand upward 5 lines)*
  - `resize-pane -L 5` *(expand left 5 lines)*
  - `resize-pane -R 5` *(expand right 5 lines)*
- *Or hold `Ctrl` and press Arrow keys immediately after `Prefix`.*

---

## 6. Scrolling & Copy Mode (Inspecting Past Logs)

When your terminal fills with output, you cannot scroll with normal arrow keys without entering **Copy Mode**:

1. **Enter scroll/copy mode**:
   ```
   Prefix, [
   ```
   *(A line indicator like `[0/1250]` appears in the top-right corner).*

2. **Scroll through output**:
   - `Up Arrow` / `Down Arrow`: Scroll line by line.
   - `Page Up` / `Page Down`: Scroll screen by screen.

3. **Search logs**:
   - Press `/`, type your search query (e.g. `Error` or `round=10`), and press `Enter`.
   - Press `n` for next match, `N` for previous match.

4. **Exit scroll mode**:
   - Press `q` or `Esc` to return to regular prompt.

---

## 7. Recommended One-Time Configuration (`~/.tmux.conf`)

To enable mouse support (scroll wheel, click to select panes, resize splits by dragging borders) and increase scrollback buffer size, run this command once on your Ubuntu server:

```bash
cat << 'EOF' > ~/.tmux.conf
# Enable mouse mode (clickable windows, panes, and wheel scroll)
set -g mouse on

# Increase scrollback buffer history from 2,000 to 50,000 lines
set -g history-limit 50000

# Use 256 colors
set -g default-terminal "screen-256color"

# Keep window numbering starting at 1 instead of 0
set -g base-index 1
setw -g pane-base-index 1
EOF

# Reload config inside running tmux (or restart tmux):
tmux source-file ~/.tmux.conf
```

> [!TIP]
> With `set -g mouse on`:
> - You can **click** on any pane to focus it.
> - You can **drag split borders** to resize panes.
> - You can use your **mouse scroll wheel** to immediately scroll through logs without pressing `Prefix, [`.
> - To copy text with mouse: hold `Shift` while selecting with mouse, then right-click or press `Ctrl+Shift+C`.

---

## 8. Ideal FedTROS Machine Learning Workflow

Here is how to set up an ideal monitoring layout for your RTX 3090 Ti:

```bash
# 1. Start a session
tmux new -s training

# 2. Split vertically into Left and Right panes
# Press: Ctrl+B then %

# 3. In the right pane, split horizontally
# Press: Ctrl+B then "

# 4. Set up each pane:
# Left Pane (large):
poetry run python scripts/run_study.py E1-IID-CS --stage main --wandb-mode disabled --seeds 42 --method fedtros_mc --only-missing --output-dir outputs runtime=gpu_fast

# Switch to Right-Top pane (Ctrl+B, Right Arrow):
watch -n 1 nvidia-smi

# Switch to Right-Bottom pane (Ctrl+B, Down Arrow):
htop

# 5. Detach safely and disconnect SSH:
# Press: Ctrl+B then D
```

When you SSH back in later:
```bash
tmux attach -t training
```
Everything will still be running exactly where you left it!
