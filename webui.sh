#!/usr/bin/env bash

set -e

# Load optional settings
if [ -f "webui.settings.sh" ]; then
    source webui.settings.sh
fi

# Defaults
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$(cd "$(dirname "$0")" && pwd)/venv}"
SD_WEBUI_RESTART="tmp/restart"
ERROR_REPORTING="FALSE"

mkdir -p tmp

# Check python
if uv help python >tmp/stdout.txt 2>tmp/stderr.txt; then
    :
elif "$PYTHON" -c "" >tmp/stdout.txt 2>tmp/stderr.txt; then
    :
else
    echo "Couldn't launch python"
    goto_show_logs=1
fi

# Check pip
if [ -z "$goto_show_logs" ]; then
    if uv help pip >tmp/stdout.txt 2>tmp/stderr.txt; then
        :
    elif "$PYTHON" -m pip --help >tmp/stdout.txt 2>tmp/stderr.txt; then
        :
    else
        echo "Couldn't launch pip"
        goto_show_logs=1
    fi
fi

# Venv handling
if [ -z "$goto_show_logs" ]; then
    if [ "$VENV_DIR" = "-" ] || [ "$SKIP_VENV" = "1" ]; then
        :
    else
        if [ -x "$VENV_DIR/bin/python" ]; then
            :
        else
            PYTHON_FULLNAME=$("$PYTHON" -c "import sys; print(sys.executable)")
            echo "Creating venv in directory $VENV_DIR using python $PYTHON_FULLNAME"

            "$PYTHON_FULLNAME" -m venv "$VENV_DIR" >tmp/stdout.txt 2>tmp/stderr.txt || {
                echo "Unable to create venv in directory \"$VENV_DIR\""
                goto_show_logs=1
            }
        fi

        if [ -z "$goto_show_logs" ]; then
            "$VENV_DIR/bin/python" -m pip install --upgrade pip || \
                echo "Warning: Failed to upgrade PIP version"

            # Activate venv
            # shellcheck disable=SC1091
            source "$VENV_DIR/bin/activate"
            PYTHON="$VENV_DIR/bin/python"
            echo "venv $PYTHON"
        fi
    fi
fi

# Launch
if [ -z "$goto_show_logs" ]; then
    "$PYTHON" launch.py "$@"

    if [ -f "$SD_WEBUI_RESTART" ]; then
        exec "$0" "$@"
    fi

    exit 0
fi

# Show logs
echo
echo "exit code: $?"

if [ -s tmp/stdout.txt ]; then
    echo
    echo "stdout:"
    cat tmp/stdout.txt
fi

if [ -s tmp/stderr.txt ]; then
    echo
    echo "stderr:"
    cat tmp/stderr.txt
fi

echo
echo "Launch Unsuccessful! Exiting..."
exit 1
