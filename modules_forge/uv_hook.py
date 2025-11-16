import shlex
import subprocess
from functools import wraps


def patch(symlink: bool):
    if hasattr(subprocess, "__original_run"):
        return

    subprocess.__original_run = subprocess.run

    @wraps(subprocess.__original_run)
    def patched_run(*args, **kwargs):
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
            return subprocess.__original_run([*command, *_args], **kwargs)

        cmd = command[command.index("pip") + 1 :]

        BAD_FLAGS = ("--prefer-binary", "--ignore-installed", "-I")
        cmd = [arg for arg in cmd if arg not in BAD_FLAGS]

        modified_command: list[str] = ["uv", "pip", *cmd]

        if symlink:
            modified_command.extend(["--link-mode", "symlink"])

        if kwargs.get("shell", False):
            command = shlex.join([*modified_command, *_args])
        else:
            command = [*modified_command, *_args]

        return subprocess.__original_run(command, **kwargs)

    subprocess.run = patched_run
