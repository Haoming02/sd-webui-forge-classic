import shlex
import subprocess
from copy import copy
from functools import wraps


def _pre_check():
    try:
        subprocess.run(["uv", "--help"], capture_output=True)
    except FileNotFoundError:
        print("\n[Error] uv is not installed...")
    except Exception:
        print("\n[Error] Failed to access uv...")
    else:
        return

    input("Press Enter to Continue...")
    raise SystemExit


def _set_cache():
    import os

    webui = os.path.dirname(os.path.dirname(__file__))
    cache = os.path.normpath(os.path.join(webui, ".uv-cache"))

    if not os.path.exists(cache):
        print("[uv] Creating .uv-cache folder...")
        os.makedirs(cache)

    os.environ.setdefault("UV_CACHE_DIR", cache)


def patch(symlink: bool, local: bool):
    if hasattr(subprocess, "__original_run"):
        return

    _pre_check()

    if local:
        _set_cache()

    subprocess.__original_run = subprocess.run
    BAD_FLAGS = ("--prefer-binary", "--ignore-installed", "-I")

    @wraps(subprocess.__original_run)
    def patched_run(*args, **kwargs):
        _original_args = copy(args)
        _original_kwargs = copy(kwargs)

        if args:
            command, *_args = args
        else:
            command, _args = kwargs.pop("args", ""), ()

        if isinstance(command, str):
            command = shlex.split(command)
        else:
            command = [arg.strip() for arg in command]

        assert isinstance(command, list)

        if "pip" not in command:
            return subprocess.__original_run(*_original_args, **_original_kwargs)

        cmd = command[command.index("pip") + 1 :]

        cmd = [arg for arg in cmd if arg not in BAD_FLAGS]

        modified_command: list[str] = ["uv", "pip", *cmd]

        if symlink:
            modified_command.extend(["--link-mode", "symlink"])

        command = [*modified_command, *_args]
        if kwargs.get("shell", False):
            command = shlex.join(command).replace("'", '"')

        return subprocess.__original_run(command, **kwargs)

    subprocess.run = patched_run
