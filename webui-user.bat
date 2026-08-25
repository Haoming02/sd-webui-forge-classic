@echo off

:: set PYTHON=
:: set GIT=
:: set VENV_DIR=

set COMMANDLINE_ARGS=

:: --xformers --sage --uv
:: --pin-shared-memory --cuda-malloc --cuda-stream
:: --skip-python-version-check --skip-torch-cuda-test --skip-version-check --skip-prepare-environment --skip-install

:: The compute backend is auto-detected. To pin it:
:: --gpu-backend cuda | rocm | zluda | directml | cpu
::
:: AMD Radeon users: run webui-user-amd.bat instead, and see AMD.md

call webui.bat
