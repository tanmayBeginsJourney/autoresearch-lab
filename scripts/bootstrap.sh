#!/usr/bin/env bash
# One-time (idempotent) setup on the cluster login node. Usage: scripts/bootstrap.sh [step ...]
# Steps: dirs tools claude envs check  (default: all, in order). Prints no secrets.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
: "${SCRATCH:?SCRATCH is not set}"
LAB_ROOT="${LAB_ROOT:-$SCRATCH/lab}"
ENVFILE="$HOME/.config/researchctl/env"
say() { printf '\n== %s\n' "$*"; }

step_dirs() {
  say "dirs + environment file"
  mkdir -p "$LAB_ROOT"/{runs,venvs,models,cache/hf,cache/uv,cache/papers,approvals,tools} "$(dirname "$ENVFILE")"
  cat > "$ENVFILE" <<EOV
# Written by autoresearch-lab/scripts/bootstrap.sh. Heavy caches live on scratch, not \$HOME.
export LAB_ROOT="$LAB_ROOT"
export RESEARCHCTL_REPO="$REPO"
export UV_CACHE_DIR="$LAB_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$LAB_ROOT/tools/python"
export HF_HOME="$LAB_ROOT/cache/hf"
export PATH="\$HOME/.local/bin:$LAB_ROOT/tools/npm/bin:\$PATH"
[ -f "\$HOME/.config/researchctl/secrets.env" ] && . "\$HOME/.config/researchctl/secrets.env"
EOV
  touch "$HOME/.config/researchctl/secrets.env"; chmod 600 "$HOME/.config/researchctl/secrets.env"
  grep -q 'researchctl/env' "$HOME/.bashrc" || echo '[ -f ~/.config/researchctl/env ] && . ~/.config/researchctl/env' >> "$HOME/.bashrc"
  . "$ENVFILE"; echo "LAB_ROOT=$LAB_ROOT"
}

step_tools() {
  say "researchctl"
  . "$ENVFILE"
  uv tool install --force --python 3.12 --editable "$REPO" >/dev/null
  researchctl doctor | head -c 600; echo
}

step_claude() {
  say "Claude Code"
  . "$ENVFILE"
  if command -v claude >/dev/null; then claude --version; return; fi
  if curl -fsSL https://claude.ai/install.sh -o "$LAB_ROOT/tools/install-claude.sh"; then
    bash "$LAB_ROOT/tools/install-claude.sh"
  else
    echo "native installer unreachable; falling back to npm via conda nodejs"
    module load conda/miniforge3
    conda create -y -q -p "$LAB_ROOT/tools/node" nodejs=22 >/dev/null
    PATH="$LAB_ROOT/tools/node/bin:$PATH" npm install -g --prefix "$LAB_ROOT/tools/npm" @anthropic-ai/claude-code >/dev/null
  fi
  hash -r; claude --version
}

step_envs() {
  say "python environments (several GB, on scratch)"
  . "$ENVFILE"
  for e in research interp kernels; do researchctl env sync "$e" | head -c 300; echo; done
}

step_check() {
  say "GPU validation job on one 24GB slice (<=10 min)"
  . "$ENVFILE"
  researchctl exec --profile s24 --minutes 10 -- "$LAB_ROOT/venvs/research/bin/python" "$REPO/scripts/envcheck.py"
}

steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(dirs tools claude envs check)
for s in "${steps[@]}"; do "step_$s"; done
say "next: see docs/operations.md (login, trust, start the persistent session)"
