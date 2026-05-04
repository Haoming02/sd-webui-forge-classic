#!/bin/bash
set -euo pipefail

# TCMalloc reduces memory fragmentation under large model workloads
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# Read COMMANDLINE_ARGS then unset it
RAW_ARGS="${COMMANDLINE_ARGS:-}"
unset COMMANDLINE_ARGS

EXTRA_ARGS=()
if [[ -n "$RAW_ARGS" ]]; then
    read -ra EXTRA_ARGS <<< "$RAW_ARGS"
fi

# Symlink settings files into the config bind-mount so they persist across container recreations
for f in config.json ui-config.json styles.csv user.css; do
    ln -sf /home/forge/sd-webui/config/$f /home/forge/sd-webui/$f
done

exec python /home/forge/sd-webui/launch.py \
    --listen \
    "${EXTRA_ARGS[@]}" \
    "$@"
