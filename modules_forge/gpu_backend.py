"""
Compute backend detection & selection.

This module is imported by `modules/launch_utils.py` *before* PyTorch is
installed, therefore it must only use the standard library.

Supported backends
------------------
``cuda``     NVIDIA GPUs (the historical default of this repository)
``rocm``     AMD GPUs using native ROCm PyTorch wheels (Linux, and Windows via
             AMD's "TheRock" nightly wheels)
``zluda``    AMD GPUs on Windows using CUDA PyTorch wheels translated by ZLUDA
             (requires the AMD HIP SDK to be installed)
``directml`` Any DirectX 12 GPU through ``torch-directml`` (very limited)
``cpu``      No GPU acceleration
"""

import functools
import json
import os
import re
import subprocess
import sys

IS_WINDOWS = os.name == "nt"


class Backend:
    CUDA = "cuda"
    ROCM = "rocm"
    ZLUDA = "zluda"
    DIRECTML = "directml"
    CPU = "cpu"

    ALL = ("cuda", "rocm", "zluda", "directml", "cpu")
    AMD = ("rocm", "zluda")


# region GPU discovery


#: Marketing name (lower-case substring) -> LLVM ``gfx`` target.
#: Only used as a fallback when the HIP runtime cannot be queried.
AMD_ARCH_BY_NAME: tuple[tuple[str, str], ...] = (
    # RDNA 4
    ("rx 9070", "gfx1201"),
    ("rx 9060", "gfx1200"),
    # RDNA 3
    ("rx 7900", "gfx1100"),
    ("rx 7800", "gfx1101"),
    ("rx 7700", "gfx1101"),
    ("rx 7650", "gfx1102"),
    ("rx 7600", "gfx1102"),
    ("radeon 890m", "gfx1151"),
    ("radeon 880m", "gfx1150"),
    ("radeon 780m", "gfx1103"),
    ("radeon 760m", "gfx1103"),
    # RDNA 2
    ("rx 6950", "gfx1030"),
    ("rx 6900", "gfx1030"),
    ("rx 6800", "gfx1030"),
    ("rx 6750", "gfx1031"),
    ("rx 6700", "gfx1031"),
    ("rx 6650", "gfx1032"),
    ("rx 6600", "gfx1032"),
    ("rx 6500", "gfx1034"),
    ("rx 6400", "gfx1034"),
    ("radeon 680m", "gfx1035"),
    ("radeon 660m", "gfx1035"),
    ("pro w6800", "gfx1030"),
    ("pro w6600", "gfx1032"),
    # RDNA 1
    ("rx 5700", "gfx1010"),
    ("rx 5600", "gfx1012"),
    ("rx 5500", "gfx1012"),
    # GCN
    ("radeon vii", "gfx906"),
    ("vega 64", "gfx900"),
    ("vega 56", "gfx900"),
    ("rx 590", "gfx803"),
    ("rx 580", "gfx803"),
    ("rx 570", "gfx803"),
)


def _run(command: list[str] | str, timeout: int = 30) -> str:
    try:
        proc = subprocess.run(
            command,
            shell=isinstance(command, str),
            check=False,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""
    return proc.stdout.decode("utf-8", errors="ignore")


@functools.cache
def list_gpu_names() -> tuple[str, ...]:
    """Names of every display adapter present in the system."""

    names: list[str] = []

    if IS_WINDOWS:
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ]
        )
        names = [line.strip() for line in out.splitlines() if line.strip()]

        if not names:  # very old Windows 10 builds without CIM cmdlets
            out = _run("wmic path win32_VideoController get name")
            names = [line.strip() for line in out.splitlines()[1:] if line.strip()]
    else:
        out = _run("lspci -mm")
        for line in out.splitlines():
            if re.search(r'"(VGA compatible controller|Display controller|3D controller)"', line):
                fields = re.findall(r'"([^"]*)"', line)
                if len(fields) >= 4:
                    names.append(f"{fields[2]} {fields[3]}".strip())

    return tuple(names)


@functools.cache
def has_nvidia_gpu() -> bool:
    if _run(["nvidia-smi", "-L"]).strip():
        return True
    return any("nvidia" in name.lower() for name in list_gpu_names())


@functools.cache
def has_amd_gpu() -> bool:
    return bool(amd_gpu_names())


@functools.cache
def amd_gpu_names() -> tuple[str, ...]:
    found = []
    for name in list_gpu_names():
        lowered = name.lower()
        if "amd" in lowered or "radeon" in lowered or "advanced micro devices" in lowered:
            found.append(name)
    return tuple(found)


# The HIP runtime (``amdhip64*.dll``) ships with the AMD Adrenalin driver, so it
# is normally available even when neither the HIP SDK nor ROCm is installed.
# Querying it is far more reliable than matching marketing names.
_HIP_ARCH_PROBE = r"""
import ctypes, ctypes.util, os, sys

def load():
    if os.name == "nt":
        root = os.path.join(os.environ.get("windir", r"C:\Windows"), "System32")
        for dll in ("amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"):
            path = os.path.join(root, dll)
            if os.path.isfile(path):
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    continue
        return None
    for so in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
        try:
            return ctypes.CDLL(so)
        except OSError:
            continue
    return None

hip = load()
if hip is None:
    raise SystemExit(1)

# hipDeviceProp_t is large and version dependent; over-allocate and scan it for
# the NUL terminated "gfx...." string that describes the compute architecture.
BUFFER = ctypes.c_byte * 4096

hip.hipInit(0)
count = ctypes.c_int()
if hip.hipGetDeviceCount(ctypes.byref(count)) != 0:
    raise SystemExit(1)

for index in range(count.value):
    prop = BUFFER()
    getter = getattr(hip, "hipGetDevicePropertiesR0600", None) or hip.hipGetDeviceProperties
    if getter(ctypes.byref(prop), index) != 0:
        continue
    raw = bytes(prop)
    match = None
    start = 0
    while True:
        start = raw.find(b"gfx", start)
        if start == -1:
            break
        end = start
        while end < len(raw) and raw[end] not in (0, 0x3A):  # NUL or ':'
            end += 1
        candidate = raw[start:end].decode("ascii", "ignore")
        if len(candidate) >= 6 and candidate[3:].strip("0123456789abcdef") == "":
            match = candidate
            break
        start += 3
    print(f"{index}\t{match or ''}")
"""


@functools.cache
def hip_agents() -> tuple[str, ...]:
    """``gfx`` target of every HIP visible device, in device order."""

    out = _run([sys.executable, "-c", _HIP_ARCH_PROBE], timeout=60)
    agents = []
    for line in out.splitlines():
        _, _, arch = line.partition("\t")
        arch = arch.strip()
        if arch:
            agents.append(arch)
    return tuple(agents)


def _arch_from_name(name: str) -> str | None:
    lowered = name.lower()
    for needle, arch in AMD_ARCH_BY_NAME:
        if needle in lowered:
            return arch
    return None


@functools.cache
def amd_arch() -> str | None:
    """``gfx`` target of the AMD GPU this install should be built for."""

    override = os.environ.get("SD_AMD_ARCH") or os.environ.get("GFX_VERSION")
    if override:
        return override.split(":")[0].strip()

    agents = hip_agents()
    if agents:
        return agents[0]

    for name in amd_gpu_names():
        arch = _arch_from_name(name)
        if arch is not None:
            return arch

    return None


def gfx_version(arch: str) -> int:
    """``"gfx1030"`` -> ``0x1030``. Returns ``0`` when unparseable."""

    digits = ""
    for char in arch[3:]:
        if char in "0123456789abcdef":
            digits += char
        else:
            break
    try:
        return int(digits, 16)
    except ValueError:
        return 0


# endregion


# region ROCm wheel index


#: ``gfx`` family -> path on https://rocm.nightlies.amd.com
#: Mirrors the families AMD publishes PyTorch wheels for.
def therock_family(arch: str | None) -> str | None:
    if not arch:
        return None

    version = gfx_version(arch)

    if (version & 0xFFF0) == 0x1200:
        return "v2/gfx120X-all"
    if (version & 0xFFF0) == 0x1100:
        return "v2/gfx110X-all"
    if version == 0x1151:
        return "v2/gfx1151"
    if version == 0x1150:
        return "v2-staging/gfx1150"
    if version in (0x1152, 0x1153):
        return "v2-staging/gfx115X"
    if version in (0x1030, 0x1031, 0x1032, 0x1034):
        # RDNA 2 desktop: RX 6800/6900, RX 6700, RX 6600, RX 6500
        return "v2-staging/gfx103X-dgpu"

    return None


def rocm_index_url(arch: str | None = None) -> str | None:
    explicit = os.environ.get("ROCM_INDEX_URL")
    if explicit:
        return explicit.rstrip("/") + "/"

    family = therock_family(arch if arch is not None else amd_arch())
    if family is None:
        return None

    return f"https://rocm.nightlies.amd.com/{family}/"


# endregion


# region Selection


def installed_backend() -> str | None:
    """Backend implied by the PyTorch build that is currently installed."""

    try:
        import torch
    except Exception:
        return None

    if getattr(torch.version, "hip", None):
        return Backend.ROCM

    try:
        import torch_directml  # noqa: F401
    except Exception:
        pass
    else:
        if os.environ.get("SD_GPU_BACKEND") == Backend.DIRECTML:
            return Backend.DIRECTML

    if getattr(torch.version, "cuda", None):
        return Backend.ZLUDA if is_zluda_runtime() else Backend.CUDA

    return Backend.CPU


def is_zluda_runtime() -> bool:
    """True when the loaded CUDA runtime is actually ZLUDA."""

    if os.environ.get("SD_ZLUDA_ACTIVE") == "1":
        return True

    try:
        import torch
    except Exception:
        return False

    try:
        name = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        return False

    return "[ZLUDA]" in name


def select_backend(requested: str | None = None) -> str:
    """
    Resolve the backend to install/run for.

    ``requested`` comes from ``--gpu-backend`` / ``$SD_GPU_BACKEND``; ``auto``
    (or ``None``) picks the best option for the detected hardware.
    """

    requested = (requested or os.environ.get("SD_GPU_BACKEND") or "auto").strip().lower()

    if requested in Backend.ALL:
        return requested

    if requested not in ("auto", ""):
        raise ValueError(f"Unknown GPU backend {requested!r}; expected one of {', '.join(Backend.ALL)} or auto")

    # An existing install wins, so that re-launching never silently swaps the
    # whole PyTorch stack underneath the user.
    already = installed_backend()
    if already is not None and already != Backend.CPU:
        return already

    if has_nvidia_gpu():
        return Backend.CUDA

    if has_amd_gpu():
        if not IS_WINDOWS:
            return Backend.ROCM
        # Windows: prefer native ROCm wheels when AMD publishes them for this
        # architecture, otherwise fall back to ZLUDA (needs the HIP SDK).
        if rocm_index_url() is not None:
            return Backend.ROCM
        if hip_sdk_path() is not None:
            return Backend.ZLUDA
        return Backend.ROCM

    return Backend.CPU


@functools.cache
def hip_sdk_path() -> str | None:
    """Installation root of the AMD HIP SDK / ROCm, if present."""

    explicit = os.environ.get("HIP_PATH")
    if explicit and os.path.exists(explicit):
        return explicit

    if IS_WINDOWS:
        root = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "AMD", "ROCm")
        if not os.path.isdir(root):
            return None

        def key(name: str):
            try:
                return tuple(int(part) for part in name.split("."))
            except ValueError:
                return ()

        versions = sorted((name for name in os.listdir(root) if key(name)), key=key)
        if not versions:
            return None
        return os.path.join(root, versions[-1])

    return "/opt/rocm" if os.path.exists("/opt/rocm") else None


def hip_sdk_version() -> tuple[int, int] | None:
    path = hip_sdk_path()
    if path is None:
        return None
    name = os.path.basename(path.rstrip("/\\"))
    try:
        major, minor = (int(part) for part in name.split(".")[:2])
    except ValueError:
        return None
    return (major, minor)


def describe() -> str:
    payload = {
        "gpus": list(list_gpu_names()),
        "amd_arch": amd_arch(),
        "hip_agents": list(hip_agents()),
        "hip_sdk": hip_sdk_path(),
        "rocm_index": rocm_index_url(),
    }
    return json.dumps(payload, indent=2)


# endregion


if __name__ == "__main__":
    print(describe())
    print("selected backend:", select_backend())
