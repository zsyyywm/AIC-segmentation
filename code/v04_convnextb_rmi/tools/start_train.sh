#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$SCRIPT_DIR")"
PYTHON_BIN="$(command -v python)"

if ! command -v tmux >/dev/null 2>&1; then
    printf 'tmux is required. Install it with: apt-get install -y tmux\n' >&2
    exit 1
fi

SESSION_NAME="aic_$(date +%Y%m%d_%H%M%S)_$$"
printf -v TRAIN_COMMAND '%q ' "$PYTHON_BIN" -u "$SCRIPT_DIR/train.py" "$@"

tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION_NAME" -l "$TRAIN_COMMAND"
tmux send-keys -t "$SESSION_NAME" Enter

printf 'Training session: %s\n' "$SESSION_NAME"
printf 'Detach: Ctrl+B, then D\n'
printf 'Reconnect: tmux attach -t %s\n' "$SESSION_NAME"
printf 'VSCode/SSH disconnects will not stop this session.\n'

if [[ -t 0 && -t 1 ]]; then
    tmux attach-session -t "$SESSION_NAME"
fi
