#!/bin/bash
set -euo pipefail

# TCMalloc reduces memory fragmentation under large model workloads.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# Detect whether a CUDA-capable GPU is accessible at runtime.
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    HAS_CUDA=true
else
    HAS_CUDA=false
    echo "[entrypoint] No CUDA GPU detected — running in CPU mode."
fi

# Read COMMANDLINE_ARGS then unset it — launch.py also reads this env var,
# so leaving it set causes every flag to appear twice in the final args list.
RAW_ARGS="${COMMANDLINE_ARGS:-}"
unset COMMANDLINE_ARGS

EXTRA_ARGS=()
if [[ -n "$RAW_ARGS" ]]; then
    read -ra EXTRA_ARGS <<< "$RAW_ARGS"
fi

# When running without CUDA, bypass the GPU check and drop GPU-only flags.
if [[ "$HAS_CUDA" == "false" ]]; then
    FILTERED=("--skip-torch-cuda-test")
    for arg in "${EXTRA_ARGS[@]}"; do
        case "$arg" in
            --cuda-malloc|--xformers)
                echo "[entrypoint] Dropping GPU-only flag: $arg" ;;
            *)
                FILTERED+=("$arg") ;;
        esac
    done
    EXTRA_ARGS=("${FILTERED[@]}")
fi

exec python /home/forge/sd-webui/launch.py \
    --listen \
    --port "${FORGE_PORT:-7860}" \
    --skip-python-version-check \
    "${EXTRA_ARGS[@]}" \
    "$@"
