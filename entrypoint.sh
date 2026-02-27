#!/usr/bin/env bash
set -euo pipefail

FORGE_HOST="${FORGE_HOST:-0.0.0.0}"
FORGE_ARGS="${FORGE_ARGS:-}"

cd /workspace

read -r -a EXTRA_ARGS <<< "${FORGE_ARGS}"

exec python3 launch.py \
  --listen "${FORGE_HOST}" \
  --port 7860 \
  "${EXTRA_ARGS[@]}"
