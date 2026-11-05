#!/usr/bin/env bash
# Inject a compact lab status at session start so the agent does not spend turns rediscovering it.
[ -f "$HOME/.config/researchctl/env" ] && . "$HOME/.config/researchctl/env"
command -v researchctl >/dev/null || { echo "researchctl not installed: run scripts/bootstrap.sh"; exit 0; }
echo "Lab digest (researchctl digest):"
timeout 30 researchctl digest 2>/dev/null || echo "digest unavailable"
