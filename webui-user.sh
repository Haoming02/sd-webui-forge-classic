#!/usr/bin/env bash

# export PYTHON=
# export GIT=
# export VENV_DIR=

export TORCH_COMMAND="pip install torch==2.10.0 torchvision==0.25.0"

export COMMANDLINE_ARGS="--uv"

# --skip-python-version-check --skip-torch-cuda-test --skip-version-check --skip-prepare-environment --skip-install

exec "$(dirname "$0")/webui.sh"
