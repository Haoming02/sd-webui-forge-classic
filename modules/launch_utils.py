"""
This script installs necessary requirements and launches main program in webui.py
"""

import importlib.metadata
import importlib.util
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Final, NamedTuple

from modules import cmd_args, errors
from modules.paths_internal import extensions_builtin_dir, extensions_dir, script_path
from modules.timer import startup_timer
from modules_forge import forge_version
from modules_forge.config import always_disabled_extensions

args, _ = cmd_args.parser.parse_known_args()

python = sys.executable
git = os.environ.get("GIT", "git")
index_url = os.environ.get("INDEX_URL", "")
dir_repos = "repositories"

default_command_live = os.environ.get("WEBUI_LAUNCH_LIVE_OUTPUT") == "1"

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


def check_python_version():
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro

    if not (major == 3 and minor == 13):
        import modules.errors

        modules.errors.print_error_explanation(
            f"""
            This program is tested with 3.13.12 Python, but you have {major}.{minor}.{micro}.
            If you encounter any error regarding unsuccessful package/library installation,
            please downgrade (or upgrade) to the latest version of 3.13 Python,
            and delete the current Python "venv" folder in WebUI's directory.

            Use --skip-python-version-check to suppress this warning
            """
        )


def git_tag():
    return forge_version.version


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": "utf8",
        "errors": "ignore",
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return result.stdout or ""


def _torch_version() -> tuple[str, str]:
    """Given `2.10.0.dev20251111+cu130` ; Return `("2.10.0", "cu130")`"""
    import importlib.metadata

    ver = importlib.metadata.version("torch")
    m = re.search(r"(\d+\.\d+\.\d+)(?:[^+]+)?\+(.+)", ver)

    if m is None:
        print("\n\nFailed to parse PyTorch version...")
        ver = os.environ.get("PYTORCH_VERSION", "2.9.1+cu128")
        print("Assuming: ", ver)
        print('(you can change this with `export PYTORCH_VERSION="..."`)\n\n')
        m = re.search(r"(\d+\.\d+\.\d+)(?:[^+]+)?\+(.+)", ver)

    return m.group(1), m.group(2)


def is_installed(package):
    try:
        dist = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        try:
            spec = importlib.util.find_spec(package)
        except ModuleNotFoundError:
            return False

        return spec is not None

    return dist is not None


def repo_dir(name):
    return os.path.join(script_path, dir_repos, name)


def run_pip(command, desc=None, live=default_command_live):
    if args.skip_install:
        return

    index_url_line = f" --index-url {index_url}" if index_url != "" else ""
    return run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)


def check_run_python(code: str, *, return_error: bool = False) -> bool | tuple[bool, str]:
    result = subprocess.run([python, "-c", code], capture_output=True, shell=False)
    if return_error:
        return result.returncode == 0, result.stderr
    else:
        return result.returncode == 0


def git_fix_workspace(*args, **kwargs):
    raise NotImplementedError()


def run_git(*args, **kwargs):
    raise NotImplementedError()


def git_clone(*args, **kwargs):
    raise NotImplementedError()


def git_pull_recursive(dir):
    for subdir, _, _ in os.walk(dir):
        if os.path.exists(os.path.join(subdir, ".git")):
            try:
                output = subprocess.check_output([git, "-C", subdir, "pull", "--autostash"])
                print(f"Pulled changes for repository in '{subdir}':\n{output.decode('utf-8').strip()}\n")
            except subprocess.CalledProcessError as e:
                print(f"Couldn't perform 'git pull' on repository in '{subdir}':\n{e.output.decode('utf-8').strip()}\n")


def run_extension_installer(extension_dir):
    path_installer = os.path.join(extension_dir, "install.py")
    if not os.path.isfile(path_installer):
        return

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{script_path}{os.pathsep}{env.get('PYTHONPATH', '')}"

        stdout = run(f'"{python}" "{path_installer}"', errdesc=f"Error running install.py for extension {extension_dir}", custom_env=env).strip()
        if stdout:
            print(stdout)
    except Exception as e:
        errors.report(str(e))


def list_extensions(settings_file):
    settings = {}

    try:
        with open(settings_file, "r", encoding="utf8") as file:
            settings = json.load(file)
    except FileNotFoundError:
        pass
    except Exception:
        errors.report(f'\nCould not load settings\nThe config file "{settings_file}" is likely corrupted\nIt has been moved to the "tmp/config.json"\nReverting config to default\n\n' "", exc_info=True)
        os.replace(settings_file, os.path.join(script_path, "tmp", "config.json"))

    disabled_extensions = set(settings.get("disabled_extensions", []) + always_disabled_extensions)
    disable_all_extensions = settings.get("disable_all_extensions", "none")

    if disable_all_extensions != "none" or args.disable_extra_extensions or args.disable_all_extensions or not os.path.isdir(extensions_dir):
        return []

    return [x for x in os.listdir(extensions_dir) if x not in disabled_extensions]


def list_extensions_builtin(settings_file):
    settings = {}

    try:
        with open(settings_file, "r", encoding="utf8") as file:
            settings = json.load(file)
    except FileNotFoundError:
        pass
    except Exception:
        errors.report(f'\nCould not load settings\nThe config file "{settings_file}" is likely corrupted\nIt has been moved to the "tmp/config.json"\nReverting config to default\n\n' "", exc_info=True)
        os.replace(settings_file, os.path.join(script_path, "tmp", "config.json"))

    disabled_extensions = set(settings.get("disabled_extensions", []))
    disable_all_extensions = settings.get("disable_all_extensions", "none")

    if disable_all_extensions != "none" or args.disable_extra_extensions or args.disable_all_extensions or not os.path.isdir(extensions_builtin_dir):
        return []

    return [x for x in os.listdir(extensions_builtin_dir) if x not in disabled_extensions]


def run_extensions_installers(settings_file):
    if not os.path.isdir(extensions_dir):
        return

    with startup_timer.subcategory("run extensions installers"):
        for dirname_extension in list_extensions(settings_file):
            logging.debug(f"Installing {dirname_extension}")

            path = os.path.join(extensions_dir, dirname_extension)

            if os.path.isdir(path):
                run_extension_installer(path)
                startup_timer.record(dirname_extension)

    if not os.path.isdir(extensions_builtin_dir):
        return

    with startup_timer.subcategory("run extensions_builtin installers"):
        for dirname_extension in list_extensions_builtin(settings_file):
            logging.debug(f"Installing {dirname_extension}")

            path = os.path.join(extensions_builtin_dir, dirname_extension)

            if os.path.isdir(path):
                run_extension_installer(path)
                startup_timer.record(dirname_extension)

    return


re_requirement = re.compile(r"\s*(\S+)\s*==\s*([^\s;]+)\s*")


def requirements_met(requirements_file):
    """
    Does a simple parse of a requirements.txt file to determine if all rerqirements in it
    are already installed. Returns True if so, False if not installed or parsing fails.
    """

    import importlib.metadata

    import packaging.version

    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            if line.strip() == "":
                continue

            if (m := re.match(re_requirement, line)) is None:
                continue

            package = m.group(1)
            version_required = m.group(2)

            try:
                version_installed = importlib.metadata.version(package)
            except Exception:
                return False

            if version_installed is None:
                return False

            if packaging.version.parse(version_installed) < packaging.version.parse(version_required):
                return False

    return True


def prepare_environment():
    torch_index_url = os.environ.get("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu130")
    torch_command = os.environ.get("TORCH_COMMAND", f"pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 --extra-index-url {torch_index_url}")
    xformers_package = os.environ.get("XFORMERS_PACKAGE", f"xformers==0.0.34 --extra-index-url {torch_index_url}")
    bnb_package = os.environ.get("BNB_PACKAGE", "bitsandbytes==0.49.1")

    packaging_package = os.environ.get("PACKAGING_PACKAGE", "packaging==26.0")
    gradio_package = os.environ.get("GRADIO_PACKAGE", "gradio==4.40.0 gradio_rangeslider==0.0.8")
    requirements_file = os.environ.get("REQS_FILE", "requirements.txt")

    try:
        # the existence of this file is a signal to webui.sh/bat that webui needs to be restarted when it stops execution
        os.remove(os.path.join(script_path, "tmp", "restart"))
        os.environ.setdefault("SD_WEBUI_RESTARTING", "1")
    except OSError:
        pass

    if not args.skip_python_version_check:
        check_python_version()

    startup_timer.record("checks")

    tag = git_tag()

    print(f"Python {sys.version}")
    print(f"Version: {tag}")

    if args.reinstall_torch or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)
        startup_timer.record("install torch")

    if not args.skip_torch_cuda_test:
        success, err = check_run_python("import torch; assert torch.cuda.is_available()", return_error=True)
        if not success:
            if "older driver" in str(err).lower():
                raise SystemError("Please update your GPU driver to support cu130 ; or manually install older PyTorch")
            raise RuntimeError("PyTorch is not able to access CUDA")
        startup_timer.record("torch GPU test")

    if not is_installed("packaging"):
        run_pip(f"install {packaging_package}", "packaging")

    ver_PY = f"cp{sys.version_info.major}{sys.version_info.minor}"
    ver_SAGE = "2.2.0"
    ver_FLASH = "2.8.3"
    ver_TRITON = "3.6.0"
    ver_NUNCHAKU = "1.2.1"
    ver_TORCH, ver_CUDA = _torch_version()
    v_TORCH = ver_TORCH.rsplit(".", 1)[0]
    v_CUDA = f"{ver_CUDA[0:-1]}.{ver_CUDA[-1]}"

    if os.name == "nt":
        ver_TRITON += ".post25"

        sage_package = os.environ.get("SAGE_PACKAGE", f"https://github.com/woct0rdho/SageAttention/releases/download/v{ver_SAGE}-windows.post4/sageattention-{ver_SAGE}+{ver_CUDA}torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl")
        flash_package = os.environ.get("FLASH_PACKAGE", f"https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.13/flash_attn-{ver_FLASH}+{ver_CUDA}torch{v_TORCH}-{ver_PY}-{ver_PY}-win_amd64.whl")
        triton_package = os.environ.get("TRITION_PACKAGE", f"triton-windows=={ver_TRITON}")
        nunchaku_package = os.environ.get("NUNCHAKU_PACKAGE", f"https://github.com/nunchaku-ai/nunchaku/releases/download/v{ver_NUNCHAKU}/nunchaku-{ver_NUNCHAKU}+{v_CUDA}torch{v_TORCH}-{ver_PY}-{ver_PY}-win_amd64.whl")

    else:
        sage_package = os.environ.get("SAGE_PACKAGE", f"sageattention=={ver_SAGE}")
        flash_package = os.environ.get("FLASH_PACKAGE", f"https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-{ver_FLASH}+{ver_CUDA}torch{v_TORCH}-{ver_PY}-{ver_PY}-linux_x86_64.whl")
        triton_package = os.environ.get("TRITION_PACKAGE", f"triton=={ver_TRITON}")
        nunchaku_package = os.environ.get("NUNCHAKU_PACKAGE", f"https://github.com/nunchaku-ai/nunchaku/releases/download/v{ver_NUNCHAKU}/nunchaku-{ver_NUNCHAKU}+{v_CUDA}torch{v_TORCH}-{ver_PY}-{ver_PY}-linux_x86_64.whl")

    def _verify_nunchaku() -> bool:
        if not is_installed("nunchaku"):
            return False

        import importlib.metadata

        import packaging.version

        ver_installed: str = importlib.metadata.version("nunchaku")
        current: tuple[int] = packaging.version.parse(ver_installed)
        target: tuple[int] = packaging.version.parse(ver_NUNCHAKU)

        return current >= target

    if args.xformers and (not is_installed("xformers") or args.reinstall_xformers):
        run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")
        startup_timer.record("install xformers")

    if args.sage:
        if not is_installed("triton"):
            try:
                run_pip(f"install -U -I --no-deps {triton_package}", "triton")
            except RuntimeError:
                print("Failed to install triton; Please manually install it")
            else:
                startup_timer.record("install triton")
        if not is_installed("sageattention"):
            try:
                run_pip(f"install -U -I --no-deps {sage_package}", "sageattention")
            except RuntimeError:
                print("Failed to install sageattention; Please manually install it")
            else:
                startup_timer.record("install sageattention")

    if args.flash and not is_installed("flash_attn"):
        try:
            run_pip(f"install {flash_package}", "flash_attn")
        except RuntimeError:
            print("Failed to install flash_attn; Please manually install it")
        else:
            startup_timer.record("install flash_attn")

    if args.nunchaku and not _verify_nunchaku():
        try:
            run_pip(f"install {nunchaku_package}", "nunchaku")
        except RuntimeError:
            print("Failed to install nunchaku; Please manually install it")
        else:
            startup_timer.record("install nunchaku")

    if args.bnb and not is_installed("bitsandbytes"):
        try:
            run_pip(f"install {bnb_package}", "bitsandbytes")
        except RuntimeError:
            print("Failed to install bitsandbytes; Please manually install it")
        else:
            startup_timer.record("install bitsandbytes")

    if not is_installed("ngrok") and args.ngrok:
        run_pip("install ngrok", "ngrok")
        startup_timer.record("install ngrok")

    if not is_installed("gradio"):
        run_pip(f"install {gradio_package}", "gradio")

    if not os.path.isfile(requirements_file):
        requirements_file = os.path.join(script_path, requirements_file)

    if not requirements_met(requirements_file):
        run_pip(f'install -r "{requirements_file}"', "requirements")
        startup_timer.record("install requirements")

    if args.onnxruntime_gpu and not is_installed("onnxruntime-gpu"):
        # https://onnxruntime.ai/docs/install/#nightly-for-cuda-13x
        _deps = "flatbuffers numpy packaging protobuf sympy coloredlogs"
        onnxruntime_package = os.environ.get("ONNX_PACKAGE", "onnxruntime-gpu --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/")
        run_pip(f"install {_deps}", "onnxruntime dependencies")
        run_pip(f"install {onnxruntime_package}", "onnxruntime-gpu")
        startup_timer.record("install onnxruntime-gpu")

    if not args.skip_install:
        run_extensions_installers(settings_file=args.ui_settings_file)

    if args.update_all_extensions:
        git_pull_recursive(extensions_dir)
        startup_timer.record("update extensions")

    if not requirements_met(requirements_file):
        run_pip(f'install -r "{requirements_file}"', "requirements")
        startup_timer.record("enforce requirements")

    if "--exit" in sys.argv:
        print("Exiting because of --exit argument")
        exit(0)


class ModelRef(NamedTuple):
    arg_name: str
    relative_path: str


def configure_a1111_reference(a1111_home: Path):
    """Append model paths based on an existing A1111 installation"""

    refs = (
        ModelRef(arg_name="--embeddings-dir", relative_path="embeddings"),
        ModelRef(arg_name="--esrgan-models-path", relative_path="ESRGAN"),
        ModelRef(arg_name="--lora-dirs", relative_path="Lora"),
        ModelRef(arg_name="--ckpt-dirs", relative_path="Stable-diffusion"),
        ModelRef(arg_name="--text-encoder-dirs", relative_path="text_encoder"),
        ModelRef(arg_name="--vae-dirs", relative_path="VAE"),
        ModelRef(arg_name="--controlnet-dir", relative_path="ControlNet"),
        ModelRef(arg_name="--controlnet-preprocessor-models-dir", relative_path="ControlNetPreprocessor"),
    )

    for ref in refs:
        target_path = a1111_home / ref.relative_path
        if not target_path.exists():
            target_path = a1111_home / "models" / ref.relative_path
        if not target_path.exists():
            continue

        sys.argv.extend([ref.arg_name, str(target_path.absolute())])


def configure_comfy_reference(comfy_home: Path):
    """Append model paths based on an existing Comfy installation"""

    refs = (
        ModelRef(arg_name="--ckpt-dirs", relative_path="checkpoints"),
        ModelRef(arg_name="--ckpt-dirs", relative_path="diffusion_models"),
        ModelRef(arg_name="--ckpt-dirs", relative_path="unet"),
        ModelRef(arg_name="--text-encoder-dirs", relative_path="clip"),
        ModelRef(arg_name="--text-encoder-dirs", relative_path="text_encoders"),
        ModelRef(arg_name="--lora-dirs", relative_path="loras"),
        ModelRef(arg_name="--vae-dirs", relative_path="vae"),
    )

    for ref in refs:
        target_path = comfy_home / ref.relative_path
        if not target_path.exists():
            target_path = comfy_home / "models" / ref.relative_path
        if not target_path.exists():
            continue

        sys.argv.extend([ref.arg_name, str(target_path.absolute())])


def _configure_yaml(base: str, config: str | list, arg: str):
    if config is None:
        return
    if isinstance(config, str):
        config = [config]

    assert isinstance(config, list)

    for folder in config:
        path = os.path.abspath(os.path.normpath(os.path.join(base, folder)))
        if os.path.isdir(path):
            sys.argv.extend([arg, str(path)])


def configure_comfy_yaml(comfy_yaml: Path):
    """Append model paths based on an existing Comfy config"""

    import yaml

    with open(comfy_yaml, "r", encoding="utf-8") as file:
        configs: dict[str, dict[str, os.PathLike]] = yaml.safe_load(file)

    for config in configs.values():
        base = config.get("base_path", "")
        _configure_yaml(base, config.get("checkpoints", None), "--ckpt-dirs")
        _configure_yaml(base, config.get("diffusion_models", None), "--ckpt-dirs")
        _configure_yaml(base, config.get("unet", None), "--ckpt-dirs")
        _configure_yaml(base, config.get("clip", None), "--text-encoder-dirs")
        _configure_yaml(base, config.get("text_encoders", None), "--text-encoder-dirs")
        _configure_yaml(base, config.get("loras", None), "--lora-dirs")
        _configure_yaml(base, config.get("vae", None), "--vae-dirs")


def start():
    print(f"Launching {'API server' if '--nowebui' in sys.argv else 'Web UI'} with arguments: {shlex.join(sys.argv[1:])}")

    from modules import logging_config

    logging_config.setup_logging(args.loglevel)

    import webui

    if "--nowebui" in sys.argv:
        webui.api_only()
    else:
        webui.webui()

    from modules_forge import main_thread

    main_thread.loop()


def dump_sysinfo():
    import datetime

    from modules import sysinfo

    text = sysinfo.get()
    filename = f"sysinfo-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d-%H-%M')}.json"

    with open(filename, "w", encoding="utf8") as file:
        file.write(text)

    return filename


VERSION_UID: Final[str] = "PY313"


def verify_version():
    """prompt user to do a clean reinstall"""
    settings_file: os.PathLike = args.ui_settings_file

    if not os.path.isfile(settings_file):
        # config.json does not exist on a fresh git clone
        with open(settings_file, "w", encoding="utf8") as file:
            json.dump({"VERSION_UID": VERSION_UID}, file)
            return

    with open(settings_file, "r", encoding="utf8") as file:
        settings: dict[str, Any] = json.load(file)

    if settings.get("VERSION_UID", None) == VERSION_UID:
        return  # already up-to-date

    os.system("")

    import shutil

    w: int = shutil.get_terminal_size().columns
    R: Final[str] = "\033[0m"
    E: Final[str] = "\033[0;31m"
    Y: Final[str] = "\033[0;33m"
    B: Final[str] = "\033[0;36m"
    G: Final[str] = "\033[0;90m"
    T: Final[str] = " " * 7

    print("\n\n")
    print("=" * w)

    print(f"{Y}ALERT:{R} You are updating from an old version...")
    print(f"{T}The recent WebUI updates include breaking changes!")
    print(f"{T}Please perform a {E}clean reinstall{R}! Remember to {B}back up{R} the models!")
    print(f"{T}{G}(alternatively, simply remove the config.json and ui-config.json files){R}")

    print("=" * w)
    print("\n\n")

    input("Press Enter to Continue...")
