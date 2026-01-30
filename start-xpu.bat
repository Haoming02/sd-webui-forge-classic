@echo off
setlocal

cd /d "%~dp0"

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
set "COMMANDLINE_ARGS=--gpu-only --skip-install --skip-prepare-environment --no-hashing"
set "TORCH_COMMAND=pip install torch==2.10.0+xpu torchvision==0.25.0+xpu --index-url https://download.pytorch.org/whl/xpu"

call webui.bat

endlocal
