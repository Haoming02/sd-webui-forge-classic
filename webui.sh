#!/bin/bash

# Source settings if exists
if [[ -f webui.settings.sh ]]; then
    source webui.settings.sh
fi

# Set defaults
if [[ -z "${PYTHON}" ]]; then
    PYTHON="python3"
fi

if [[ -z "${VENV_DIR}" ]]; then
    VENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/venv"
fi

SD_WEBUI_RESTART="tmp/restart"
ERROR_REPORTING="FALSE"

mkdir -p tmp

# Check if python is available
if command -v uv &> /dev/null; then
    uv help python >tmp/stdout.txt 2>tmp/stderr.txt && UV_AVAILABLE=1
elif command -v "$PYTHON" &> /dev/null; then
    "$PYTHON" -c "" >tmp/stdout.txt 2>tmp/stderr.txt && PYTHON_AVAILABLE=1
else
    echo "Couldn't launch python"
    cat tmp/stdout.txt tmp/stderr.txt
    exit 1
fi

# Check if pip is available
if [[ "$UV_AVAILABLE" == "1" ]]; then
    uv help pip >tmp/stdout.txt 2>tmp/stderr.txt || UV_AVAILABLE=0
fi

if [[ "$UV_AVAILABLE" != "1" ]] && [[ "$PYTHON_AVAILABLE" == "1" ]]; then
    "$PYTHON" -m pip --help >tmp/stdout.txt 2>tmp/stderr.txt || {
        echo "Couldn't launch pip"
        cat tmp/stdout.txt tmp/stderr.txt
        exit 1
    }
fi

# Start venv
if [[ "$VENV_DIR" != "-" ]] && [[ "$SKIP_VENV" != "1" ]]; then
    if [[ -f "$VENV_DIR/bin/python" ]]; then
        # Activate existing venv
        source "$VENV_DIR/bin/activate"
        PYTHON="$VENV_DIR/bin/python"
        echo "venv $PYTHON"
    else
        # Create new venv
        PYTHON_FULLNAME=$("$PYTHON" -c "import sys; print(sys.executable)")
        echo "Creating venv in directory $VENV_DIR using python $PYTHON_FULLNAME"
        "$PYTHON_FULLNAME" -m venv "$VENV_DIR" >tmp/stdout.txt 2>tmp/stderr.txt

        if [[ $? -ne 0 ]]; then
            echo "Unable to create venv in directory $VENV_DIR"
            cat tmp/stdout.txt tmp/stderr.txt
            exit 1
        fi

        # Upgrade pip
        "$VENV_DIR/bin/python" -m pip install --upgrade pip || echo "Warning: Failed to upgrade PIP version"

        # Activate venv
        source "$VENV_DIR/bin/activate"
        PYTHON="$VENV_DIR/bin/python"
        echo "venv $PYTHON"
    fi
fi

# Launch
"$PYTHON" launch.py "$@"

if [[ ! -f tmp/restart ]]; then
    echo ""
    echo "Press any key to continue..."
    read -n 1
fi
