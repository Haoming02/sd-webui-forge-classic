
:: set COMMANDLINE_ARGS=--gpu-only --skip-install --skip-prepare-environment --no-hashing
::set COMMANDLINE_ARGS=--gpu-device-id 0 --bf16-unet --bf16-vae --reserve-vram 2.0 --disable-ipex-optimize --skip-torch-cuda-test --skip-install --skip-prepare-environment --no-hashing --model-ref "C:\Users\Derek\m
@echo off

:: Use the existing venv Python (avoids pyenv shims)
set "PYTHON=C:\sd-webui-forge-neo\venv\Scripts\python.exe"
set "VENV_DIR=C:\sd-webui-forge-neo\venv"

:: XPU-safe, fast startup flags + model path
set "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1"
set "ZE_FLAT_DEVICE_HIERARCHY=COMBINED"
set "COMMANDLINE_ARGS=--gpu-only --disable-smart-memory --skip-install --skip-torch-cuda-test --gpu-device-id 0 --bf16-unet --bf16-vae --fp16-text-enc --disable-ipex-optimize --skip-prepare-environment --no-hashing --use-pytorch-cross-attention --disable-xformers --disable-sage --disable-flash --model-ref C:\\Users\\Derek\\models"

call webui.bat