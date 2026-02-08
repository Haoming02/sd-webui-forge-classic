
:: set COMMANDLINE_ARGS=--gpu-only --skip-install --skip-prepare-environment --no-hashing
::set COMMANDLINE_ARGS=--gpu-device-id 0 --bf16-unet --bf16-vae --reserve-vram 2.0 --disable-ipex-optimize --skip-torch-cuda-test --skip-install --skip-prepare-environment --no-hashing --model-ref "C:\Users\Derek\m
@echo off

:: Use the existing venv Python (avoids pyenv shims)
set "PYTHON=C:\sd-webui-forge-neo\venv\Scripts\python.exe"
set "VENV_DIR=C:\sd-webui-forge-neo\venv"

set COMMANDLINE_ARGS=--gpu-only --skip-install --skip-prepare-environment --no-hashing --use-pytorch-cross-attention --disable-xformers --disable-sage --disable-flash

call webui.bat