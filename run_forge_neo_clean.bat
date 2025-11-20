@echo off
echo =====================================
echo  Forge Neo - Clean Install Mode
echo =====================================
echo.
echo Starting with clean configuration (no xformers)...
echo - SageAttention enabled (best performance available)
echo - CUDA optimizations for RTX 4060 Ti 16GB
echo - No problematic packages
echo.

REM Navigate to the correct directory
cd /d "C:\AI\Forge_neo\forge-neo"

REM Activate the virtual environment
call venv\Scripts\activate

REM Set UV path for this session
set "PATH=C:\Users\littl\.local\bin;%PATH%"

REM Launch with clean, working configuration
python launch.py --sage --cuda-malloc --cuda-stream --pin-shared-memory   --fast-fp16  

echo.
echo Forge Neo has stopped.
pause