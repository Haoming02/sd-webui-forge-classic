#!/usr/bin/env bash

# export PYTHON=
# export GIT=
# export VENV_DIR=

# AMD GPU (ROCm) defaults.
# These wheels are published by PyTorch at https://download.pytorch.org/whl/rocm6.4/
# torch 2.9.1+rocm6.4 and torchvision 0.24.1+rocm6.4 are the latest stable
# pair for Python 3.13 on linux_x86_64.
export TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm6.4"
export TORCH_COMMAND="pip install torch==2.9.1+rocm6.4 torchvision==0.24.1+rocm6.4 --index-url ${TORCH_INDEX_URL}"

# Tell ROCm/HIP which GPU architecture to target if your card isn't auto-detected.
# Uncomment and set to match your hardware. Common values:
#   RDNA 3  (RX 7000 / W7000):  11.0.0   (gfx1100/1101/1102)
#   RDNA 2  (RX 6000 / W6000):  10.3.0   (gfx1030/1031/1032)
#   CDNA 2  (MI200 series):     leave unset; gfx90a is auto-detected
#   CDNA 3  (MI300 series):     leave unset; gfx942 is auto-detected
# export HSA_OVERRIDE_GFX_VERSION=11.0.0

# If you have multiple GPUs, pick which one PyTorch should use:
# export HIP_VISIBLE_DEVICES=0
# export ROCR_VISIBLE_DEVICES=0

export COMMANDLINE_ARGS="--uv"

# --skip-python-version-check --skip-torch-cuda-test --skip-version-check --skip-prepare-environment --skip-install

exec "$(dirname "$0")/webui.sh"
