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
from modules_forge import forge_version, gpu_backend
from modules_forge.config import always_disabled_extensions

args, _ = cmd_args.parser.parse_known_args()

python = sys.executable
git = os.environ.get("GIT", "git")
index_url = os.environ.get("INDEX_URL", "")
dir_repos = "repositories"

default_command_live = os.environ.get("WEBUI_LAUNCH_LIVE_OUTPUT") == "1"

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


#: Python versions every supported backend publishes wheels for.
SUPPORTED_PYTHON_MINORS: Final[tuple[int, ...]] = (12, 13)


def check_python_version():
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro

    if not (major == 3 and minor in SUPPORTED_PYTHON_MINORS):
        import modules.errors

        supported = " or ".join(f"3.{m}" for m in SUPPORTED_PYTHON_MINORS)
        modules.errors.print_error_explanation(f"""
            This program is tested with 3.13.12 Python, but you have {major}.{minor}.{micro}.
            If you encounter any error regarding unsuccessful package/library installation,
            please downgrade (or upgrade) to the latest version of {supported} Python,
            and delete the current Python "venv" folder in WebUI's directory.

            AMD (ROCm) wheels are only published for {supported}.

            Use --skip-python-version-check to suppress this warning
            """)


def git_tag():
    return f"{forge_version.version} {forge_version.release}"


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
        # Builds without a local version label (e.g. plain CPU wheels from PyPI)
        m = re.fullmatch(r"(\d+\.\d+\.\d+).*", ver)
        if m is not None:
            return m.group(1), "cpu"

        print("\n\nFailed to parse PyTorch version...")
        ver = os.environ.get("PYTORCH_VERSION", "2.10.0+cu130")
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


re_requirement = re.compile(r"\s*(\S+)\s*==\s*([^\s;]+)\s*")


def requirements_met(requirements_file):
    """
    Does a simple parse of a requirements.txt file to determine
    whether all dependencies are already installed.
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


#: Pinned PyTorch builds per backend.
#:
#: ROCm: AMD publishes Windows wheels through "TheRock" nightly channel. They
#: are date stamped and `torch` pins `rocm[libraries]` to the exact same stamp,
#: so the whole trio has to move together -- hence the pin.
ROCM_LINUX_CHANNEL: Final[str] = os.environ.get("ROCM_CHANNEL", "rocm7.1")
TORCH_ROCM_VERSION: Final[str] = "2.9.1+rocm7.13.0a20260421"
TORCHVISION_ROCM_VERSION: Final[str] = "0.27.0a0+rocm7.13.0a20260421"

#: ZLUDA translates the CUDA driver API, and only implements CUDA 11.x, so the
#: cu118 build of PyTorch is the one to use.
TORCH_ZLUDA_VERSION: Final[str] = "2.7.1+cu118"
TORCHVISION_ZLUDA_VERSION: Final[str] = "0.22.1+cu118"


def resolve_gpu_backend() -> str:
    """Backend to install for; see `modules_forge/gpu_backend.py`."""

    backend = gpu_backend.select_backend(getattr(args, "gpu_backend", None))
    os.environ["SD_GPU_BACKEND"] = backend
    return backend


def torch_install_command(backend: str) -> str:
    """`pip install ...` arguments that put the right PyTorch build in place."""

    explicit = os.environ.get("TORCH_COMMAND")
    if explicit:
        return explicit

    if backend == gpu_backend.Backend.ROCM:
        if os.name != "nt":
            # Linux has first-party ROCm wheels on the regular PyTorch index.
            index = os.environ.get("TORCH_INDEX_URL", f"https://download.pytorch.org/whl/{ROCM_LINUX_CHANNEL}")
            return f"pip install torch torchvision --index-url {index}"

        index = gpu_backend.rocm_index_url()
        if index is None:
            arch = gpu_backend.amd_arch() or "unknown"
            raise RuntimeError(f"""AMD does not publish Windows ROCm PyTorch wheels for your GPU architecture ({arch}).
Run with --gpu-backend zluda instead (requires the AMD HIP SDK 6.2/6.4), or
set ROCM_INDEX_URL to a wheel index that covers your card.""")
        torch_pin = os.environ.get("TORCH_VERSION", TORCH_ROCM_VERSION)
        vision_pin = os.environ.get("TORCHVISION_VERSION", TORCHVISION_ROCM_VERSION)
        # Every transitive dependency (including the rocm-sdk runtime packages)
        # is served by the same index, so it is used as the *primary* one.
        return f"pip install --pre torch=={torch_pin} torchvision=={vision_pin} --index-url {index}"

    if backend == gpu_backend.Backend.ZLUDA:
        index = os.environ.get("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu118")
        torch_pin = os.environ.get("TORCH_VERSION", TORCH_ZLUDA_VERSION)
        vision_pin = os.environ.get("TORCHVISION_VERSION", TORCHVISION_ZLUDA_VERSION)
        return f"pip install torch=={torch_pin} torchvision=={vision_pin} --extra-index-url {index}"

    if backend == gpu_backend.Backend.DIRECTML:
        return "pip install torch-directml"

    if backend == gpu_backend.Backend.CPU:
        index = os.environ.get("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cpu")
        return f"pip install torch torchvision --index-url {index}"

    index = os.environ.get("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu130")
    return f"pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --extra-index-url {index}"


def prepare_zluda():
    """Fetch ZLUDA and verify the HIP SDK is in place (Windows + AMD only)."""

    from modules_forge import zluda_installer

    if os.name != "nt":
        raise RuntimeError("ZLUDA is only supported on Windows; use --gpu-backend rocm on Linux.")

    hip_sdk = gpu_backend.hip_sdk_path()
    if hip_sdk is None:
        raise RuntimeError("""ZLUDA needs the AMD HIP SDK, which was not found.
Install HIP SDK 6.2 (or 6.4) from https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html and re-launch.
HIP SDK 7.x is not supported by ZLUDA.""")

    version = gpu_backend.hip_sdk_version()
    if version is not None and version[0] != 6:
        print(f"Warning: ZLUDA is built against HIP SDK 6.x, but {version[0]}.{version[1]} was found. Expect failures.")

    print(f"HIP SDK: {hip_sdk}")

    if args.reinstall_zluda:
        zluda_installer.uninstall()

    try:
        zluda_installer.install(use_nightly=args.zluda_nightly)
    except Exception as e:
        raise RuntimeError(f"Couldn't install ZLUDA: {e}") from e

    startup_timer.record("install zluda")


def verify_torch_build(backend: str):
    """
    Warn when installing the requirements pulled a PyTorch build that does not
    match the selected backend -- this silently breaks GPU acceleration and is
    otherwise very hard to diagnose.
    """

    expected = {
        gpu_backend.Backend.ROCM: "rocm",
        gpu_backend.Backend.ZLUDA: "cu",
        gpu_backend.Backend.CUDA: "cu",
    }.get(backend)

    if expected is None:
        return

    try:
        installed = importlib.metadata.version("torch")
    except Exception:
        return

    local = installed.partition("+")[2]
    if local.startswith(expected):
        return

    print(f"""
Warning: the '{backend}' backend expects a '{expected}*' build of PyTorch, but '{installed}' is installed.
Something (usually an extension's install.py) replaced it. Re-run with --reinstall-torch to fix.
""")


def prepare_environment():
    backend = resolve_gpu_backend()

    torch_index_url = os.environ.get("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu130")
    torch_command = torch_install_command(backend)
    xformers_package = os.environ.get("XFORMERS_PACKAGE", f"xformers==0.0.35 --extra-index-url {torch_index_url}")

    packaging_package = os.environ.get("PACKAGING_PACKAGE", "packaging==26.2")
    gradio_package = os.environ.get("GRADIO_PACKAGE", "gradio==4.40.0 gradio_rangeslider==0.0.8")
    requirements_file = os.environ.get("REQS_FILE", "requirements.txt")

    try:
        # the existence of this file is a signal that webui needs to be restarted when it stops execution
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

    print(f"Compute backend: {backend}")
    if backend in gpu_backend.Backend.AMD:
        print(f"AMD GPU: {', '.join(gpu_backend.amd_gpu_names()) or 'unknown'} ({gpu_backend.amd_arch() or 'unknown arch'})")

    if args.reinstall_torch or not is_installed("torch") or (backend != gpu_backend.Backend.DIRECTML and not is_installed("torchvision")):
        # TODO: Yeet Nunchaku...
        if args.nunchaku and backend == gpu_backend.Backend.CUDA:
            torch_command = os.environ.get("TORCH_COMMAND", f"pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 --extra-index-url {torch_index_url}")
            print("(using an older version of PyTorch due to Nunchaku dependency...)")

        run(f'"{python}" -m {torch_command}', "Installing PyTorch", "Couldn't install PyTorch", live=True)
        startup_timer.record("install torch")

    if backend == gpu_backend.Backend.ZLUDA:
        prepare_zluda()

    if not args.skip_torch_cuda_test:
        TORCH_CHECK: str = """
import torch
cuda = hasattr(torch, "cuda") and torch.cuda.is_available()
xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
mps = hasattr(torch, "mps") and torch.mps.is_available()
assert cuda or xpu or mps
        """

        if backend == gpu_backend.Backend.DIRECTML:
            TORCH_CHECK = "import torch_directml\nassert torch_directml.device_count() > 0"

        if backend == gpu_backend.Backend.CPU:
            startup_timer.record("torch GPU test")
        else:
            success, err = check_run_python(TORCH_CHECK, return_error=True)
            if not success:
                message = str(err).lower()
                if "older driver" in message:
                    raise SystemError("Please update your GPU driver or manually install older version of PyTorch")
                if backend == gpu_backend.Backend.ROCM:
                    detail = err.decode("utf-8", "ignore") if isinstance(err, bytes) else err
                    family = gpu_backend.therock_family(gpu_backend.amd_arch())
                    raise RuntimeError(f"""PyTorch (ROCm) cannot see your Radeon GPU.
  * make sure the AMD Adrenalin driver is up to date
  * confirm your GPU belongs to the {family!r} wheel family
  * or fall back to ZLUDA with --gpu-backend zluda
{detail}""")
                raise RuntimeError("PyTorch is not able to access any compute device (GPU)")
            startup_timer.record("torch GPU test")

    if not is_installed("packaging"):
        run_pip(f"install {packaging_package}", "packaging")

    ver_PY = f"cp{sys.version_info.major}{sys.version_info.minor}"
    ver_SAGE = "2.2.0"
    ver_FLASH = "2.8.3"
    ver_TRITON = "3.7.1"
    ver_NUNCHAKU = "1.2.1"
    ver_TORCH, ver_CUDA = _torch_version()
    v_TORCH = ver_TORCH.rsplit(".", 1)[0]
    v_CUDA = f"{ver_CUDA[0:-1]}.{ver_CUDA[-1]}"

    if os.name == "nt":
        ver_TRITON += ".post27"

        sage_package = os.environ.get("SAGE_PACKAGE", f"https://github.com/woct0rdho/SageAttention/releases/download/v{ver_SAGE}-windows.post6/sageattention-{ver_SAGE}+{ver_CUDA}torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl")
        flash_package = os.environ.get("FLASH_PACKAGE", f"https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.52/flash_attn-{ver_FLASH}+{ver_CUDA}torch{v_TORCH}-{ver_PY}-{ver_PY}-win_amd64.whl")
        triton_package = os.environ.get("TRITION_PACKAGE", f"triton-windows=={ver_TRITON}")
        nunchaku_package = os.environ.get("NUNCHAKU_PACKAGE", f"https://github.com/nunchaku-ai/nunchaku/releases/download/v{ver_NUNCHAKU}/nunchaku-{ver_NUNCHAKU}+{v_CUDA}torch{v_TORCH}-{ver_PY}-{ver_PY}-win_amd64.whl")

    else:
        sage_package = os.environ.get("SAGE_PACKAGE", f"sageattention=={ver_SAGE}")
        flash_package = os.environ.get("FLASH_PACKAGE", f"https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-{ver_FLASH}+{ver_CUDA}torch{v_TORCH}-{ver_PY}-{ver_PY}-linux_x86_64.whl")
        triton_package = os.environ.get("TRITION_PACKAGE", f"triton=={ver_TRITON}")
        nunchaku_package = os.environ.get("NUNCHAKU_PACKAGE", f"https://github.com/nunchaku-ai/nunchaku/releases/download/v{ver_NUNCHAKU}/nunchaku-{ver_NUNCHAKU}+{v_CUDA}torch{v_TORCH}-{ver_PY}-{ver_PY}-linux_x86_64.whl")

    cuda_only_requested = [name for name, enabled in (("--xformers", args.xformers), ("--sage", args.sage), ("--flash", args.flash), ("--nunchaku", args.nunchaku), ("--onnxruntime-gpu", args.onnxruntime_gpu)) if enabled]

    if backend != gpu_backend.Backend.CUDA and cuda_only_requested:
        print(f"Ignoring {', '.join(cuda_only_requested)}: these packages are CUDA-only and cannot run on the '{backend}' backend.")
        args.xformers = args.sage = args.flash = args.nunchaku = args.onnxruntime_gpu = False

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

    if args.nunchaku and not is_installed("nunchaku"):
        try:
            run_pip(f"install {nunchaku_package}", "nunchaku")
        except RuntimeError:
            print("Failed to install nunchaku; Please manually install it")
        else:
            startup_timer.record("install nunchaku")

    if args.ngrok and not is_installed("ngrok"):
        run_pip("install ngrok", "ngrok")
        startup_timer.record("install ngrok")

    if backend == gpu_backend.Backend.DIRECTML and not is_installed("torch_directml"):
        run_pip("install torch-directml", "torch-directml")
        startup_timer.record("install torch-directml")

    if not is_installed("gradio"):
        run_pip(f"install {gradio_package}", "gradio")

    if not os.path.isfile(requirements_file):
        requirements_file = os.path.join(script_path, requirements_file)

    if not requirements_met(requirements_file):
        run_pip(f'install -r "{requirements_file}"', "requirements")
        startup_timer.record("install requirements")

    if args.onnxruntime_gpu and not is_installed("onnxruntime-gpu"):
        # https://onnxruntime.ai/docs/install/#nightly-for-cuda-13x
        _deps = "coloredlogs flatbuffers numpy packaging protobuf sympy"
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

    verify_torch_build(backend)

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
        ModelRef(arg_name="--controlnet-dirs", relative_path="controlnet"),
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
        _configure_yaml(base, config.get("controlnet", None), "--controlnet-dirs")


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
