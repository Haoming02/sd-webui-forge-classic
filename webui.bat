@echo off

if exist webui.settings.bat (
    call webui.settings.bat
)

if not defined PYTHON (set PYTHON=python)
if defined GIT (set "GIT_PYTHON_GIT_EXECUTABLE=%GIT%")
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")

set SD_WEBUI_RESTART=tmp/restart
set ERROR_REPORTING=FALSE

mkdir tmp 2>NUL

set UV_CACHE_DIR=
echo %COMMANDLINE_ARGS% | findstr /c:"--uv-local-cache" >nul
if %ERRORLEVEL% == 0 (
    set "UV_CACHE_DIR=%~dp0.uv-cache"
    if not exist "%UV_CACHE_DIR%" mkdir "%UV_CACHE_DIR%"
    echo Using project-local uv cache at %UV_CACHE_DIR%
)

set USE_UV=0
echo %COMMANDLINE_ARGS% | findstr /c:"--uv" >nul
if %ERRORLEVEL% == 0 (
    set USE_UV=1
    goto :find_uv
)
uv help python >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 (
    set USE_UV=1
    goto :check_pip
)
%PYTHON% -c "" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :check_pip
echo Couldn't launch python
goto :show_stdout_stderr

:find_uv
uv help python >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :check_pip
echo Couldn't launch uv
goto :show_stdout_stderr

:check_pip
if not %USE_UV% == 1 goto :check_pip_fallback
uv help pip >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :start_venv
:check_pip_fallback
%PYTHON% -m pip --help >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :start_venv
echo Couldn't launch pip
goto :show_stdout_stderr

:start_venv
if ["%VENV_DIR%"] == ["-"] goto :skip_venv
if ["%SKIP_VENV%"] == ["1"] goto :skip_venv

dir "%VENV_DIR%\Scripts\Python.exe" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :activate_venv

if %USE_UV% == 1 goto :create_venv_uv

for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
echo Creating venv in directory %VENV_DIR% using python %PYTHON_FULLNAME%
%PYTHON_FULLNAME% -m venv "%VENV_DIR%" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :upgrade_pip
echo Unable to create venv in directory "%VENV_DIR%"
goto :show_stdout_stderr

:create_venv_uv
echo Creating venv in directory %VENV_DIR% using uv (3.13 --seed)
uv venv "%VENV_DIR%" --python 3.13 --seed >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :activate_venv
echo Unable to create venv in directory "%VENV_DIR%"
goto :show_stdout_stderr

:upgrade_pip
"%VENV_DIR%\Scripts\Python.exe" -m pip install --upgrade pip
if %ERRORLEVEL% == 0 goto :activate_venv
echo Warning: Failed to upgrade PIP version

:activate_venv
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
call "%VENV_DIR%\Scripts\activate.bat"
echo venv %PYTHON%

:skip_venv
goto :launch

:launch
%PYTHON% launch.py %*
if EXIST tmp/restart goto :skip_venv
pause
exit /b

:show_stdout_stderr

echo.
echo exit code: %errorlevel%

for /f %%i in ("tmp\stdout.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stdout:
type tmp\stdout.txt

:show_stderr
for /f %%i in ("tmp\stderr.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stderr:
type tmp\stderr.txt

:endofscript

echo.
echo Launch Unsuccessful! Exiting...
pause
