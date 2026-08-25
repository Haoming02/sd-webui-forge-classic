#!/usr/bin/env bash

# export PYTHON=
# export GIT=
# export VENV_DIR=

# export TORCH_COMMAND="pip install torch==2.12.0 torchvision==0.27.0"

export COMMANDLINE_ARGS="--uv"

# --skip-python-version-check --skip-torch-cuda-test --skip-version-check --skip-prepare-environment --skip-install

# The compute backend is auto-detected; pin it with
#   --gpu-backend cuda | rocm | cpu
# AMD Radeon on Linux uses the "rocm" backend; see AMD.md.

exec "$(dirname "$0")/webui.sh"
