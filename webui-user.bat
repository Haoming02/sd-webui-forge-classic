@echo off

:: set PYTHON=
:: set GIT=
:: set VENV_DIR=
:: --autotune

set PERF= --xformers --sage --flash --nunchaku --bnb --fast-fp8 --fast-fp16 --uv-symlink --cuda-malloc --pin-shared-memory --cuda-stream
set A1111= --model-ref "S:\sd-webui-models"
set COMFY= --forge-ref-comfy-home "S:\comfyui-models"
set STYLE= --styles-file "C:\Users\admin\Documents\GitHub\stable-diffusion-webui-forge-classic\styles.csv"
set SKIP= --skip-python-version-check --skip-torch-cuda-test --skip-version-check --skip-prepare-environment --skip-install

set DIS= --disable-sage --disable-flash --disable-xformers

set COMMANDLINE_ARGS= %PERF% %A1111% %COMFY% %STYLE%
:: %SKIP%

:: %DIS%

:: set SAGE= --sage2-function fp16_cuda --sage-quant-gran per_warp --sage-accum-dtype fp16+fp32
:: %SAGE%

call webui.bat
