@echo off
setlocal

cd /d "%~dp0"
echo [INFO] Stopping existing Forge instances for this workspace...
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "$repo = [regex]::Escape((Resolve-Path '.').Path); Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'launch\.py' -and $_.CommandLine -match $repo } | Select-Object -ExpandProperty ProcessId"`) do (
    echo [INFO] Terminating Forge process PID %%P
    taskkill /PID %%P /T /F >NUL 2>&1
)

for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":7863 .*LISTENING"') do (
    echo [INFO] Releasing port 7863 from PID %%P
    taskkill /PID %%P /T /F >NUL 2>&1
)

set "VENV_DIR=%~dp0venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment not found: %VENV_DIR%
    echo Create it first, then install XPU torch in the venv.
    echo Example:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install torch==2.10.0+xpu torchvision==0.25.0+xpu --index-url https://download.pytorch.org/whl/xpu
    pause
    exit /b 1
)

set "PYTHON=%VENV_PYTHON%"
set "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1"
set "ZE_FLAT_DEVICE_HIERARCHY=COMBINED"
set "SYCL_CACHE_PERSISTENT=1"
set "ONEAPI_DEVICE_SELECTOR=level_zero:gpu"
set "ZE_ENABLE_VALIDATION_LAYER=0"
set "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0"
if not exist "ui-config-clean.json" echo {}> ui-config-clean.json
set "COMMANDLINE_ARGS=--port 7863 --ui-config-file ui-config-clean.json --gpu-only --reserve-vram 1.0 --skip-install --skip-torch-cuda-test --gpu-device-id 0 --bf16-unet --bf16-vae --text-enc-device cpu --bf16-text-enc --skip-prepare-environment --no-hashing --use-pytorch-cross-attention --disable-xformers --disable-sage --disable-flash --model-ref C:\\Users\\Derek\\models"
set "TORCH_COMMAND=pip install torch==2.11.0+xpu torchvision==0.26.0+xpu --index-url https://download.pytorch.org/whl/xpu"

"%VENV_PYTHON%" launch.py
