#!/usr/bin/env bash
# Persistent Claude Code session on the login node, driven from the Claude app via Remote Control.
# Usage: lab-session.sh start|stop|attach|status|ensure   (ensure = start if missing; safe for cron)
set -uo pipefail
[ -f "$HOME/.config/researchctl/env" ] && . "$HOME/.config/researchctl/env"
REPO="${RESEARCHCTL_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
NAME=lab
LOG="${LAB_ROOT:-$HOME}/session.log"
running() { tmux has-session -t "$NAME" 2>/dev/null; }

case "${1:-status}" in
  start|ensure)
    if running; then [ "$1" = start ] && echo "already running"; exit 0; fi
    if ! claude auth status >/dev/null 2>&1; then echo "$(date -Is) not logged in; run 'claude' then /login" >> "$LOG"; exit 1; fi
    tmux new-session -d -s "$NAME" -c "$REPO" \
      "claude remote-control --name plaksha-lab --permission-mode acceptEdits 2>&1 | tee -a '$LOG'"
    echo "$(date -Is) started" >> "$LOG"; echo "started tmux session '$NAME' (open the Claude app -> Code to connect)";;
  stop) tmux kill-session -t "$NAME" && echo stopped;;
  attach) exec tmux attach -t "$NAME";;
  status) running && echo running || echo stopped; tail -n 3 "$LOG" 2>/dev/null;;
  *) echo "usage: $0 start|stop|attach|status|ensure"; exit 2;;
esac
