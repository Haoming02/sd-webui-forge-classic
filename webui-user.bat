@echo off

set SD_WEBUI_LOG_LEVEL=ERROR
set "PYTHON=C:\Users\kanek\AppData\Local\Programs\Python\Python313\python.exe"
set GIT=
set VENV_DIR=venv

set TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
set TORCH_ALLOW_TF32_CUDNN_OVERRIDE=1
set CUDA_MODULE_LOADING=LAZY

set TORCHINDUCTOR_CACHE_DIR=E:\sd-webui-forge-neo\torchinductor_cache
set TORCHINDUCTOR_FX_GRAPH_CACHE=1
set TORCHINDUCTOR_AUTOGRAD_CACHE=1
set TRITON_CACHE_DIR=E:\sd-webui-forge-neo\triton_cache

set COMMANDLINE_ARGS=--force-xformers-vae --fast-fp8 --fast-fp16 --onnxruntime-gpu --sage --xformers --force-non-blocking --mmap-torch-files --pin-shared-memory --cuda-stream 2 --share --enable-insecure-extension-access --listen --port 7860 --api

set "CLOUDFLARED=C:\Program Files (x86)\cloudflared\cloudflared.exe"
set "CFLOG=%TEMP%\cftunnel.log"
set "CFURL=%TEMP%\cftunnel_url.txt"

taskkill /f /im cloudflared.exe >nul 2>&1
type nul > "%CFLOG%" 2>nul
del "%CFURL%" >nul 2>&1

call webui.bat