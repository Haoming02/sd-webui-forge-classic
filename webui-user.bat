@echo off

:: set PYTHON=
:: set GIT=
:: set VENV_DIR=

set http_proxy=http://127.0.0.1:7897
set https_proxy=http://127.0.0.1:7897
set no_proxy=localhost,127.0.0.1,::1
set NO_PROXY=localhost,127.0.0.1,::1

set COMMANDLINE_ARGS=--uv-symlink --xformers --sage --flash --cuda-malloc --fast-fp8 --cuda-stream --forge-ref-a1111-home "E:\sd-webui-forge-aki-v1.0" --theme dark --pin-shared-memory --api --ad-no-huggingface --nunchaku

:: --xformers --sage --uv
:: --pin-shared-memory --cuda-malloc --cuda-stream
:: --skip-python-version-check --skip-torch-cuda-test --skip-version-check --skip-prepare-environment --skip-install

call webui.bat
