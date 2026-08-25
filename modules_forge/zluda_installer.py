"""
ZLUDA support (Windows + AMD only).

ZLUDA is a drop-in implementation of the CUDA driver API on top of AMD's HIP
runtime, which lets the regular CUDA builds of PyTorch run on Radeon cards.
It is the fallback for AMD GPUs that AMD does not publish native ROCm wheels
for, and requires the AMD HIP SDK (6.2 or 6.4) to be installed.

Reference: https://github.com/lshqqytiger/ZLUDA
"""

import ctypes
import os
import shutil
import site
import sys
import zipfile

from modules_forge import gpu_backend

#: ZLUDA ships CUDA libraries under their unsuffixed names; PyTorch looks them
#: up under the names the CUDA 11.x redistributables use.
DLL_MAPPING: dict[str, str] = {
    "cublas.dll": "cublas64_11.dll",
    "cusparse.dll": "cusparse64_11.dll",
    "cufft.dll": "cufft64_10.dll",
    "cufftw.dll": "cufftw64_10.dll",
    "nvrtc.dll": "nvrtc64_112_0.dll",
}

#: HIP SDK libraries that must be resident before ZLUDA is initialised.
HIPSDK_TARGETS: tuple[str, ...] = ("rocblas.dll", "rocsolver.dll", "rocsparse.dll", "hipfft.dll")

#: Pinned ZLUDA build. Override with `set ZLUDA_HASH=...` to move to another one.
DEFAULT_COMMIT = "5e717459179dc272b7d7d23391f0fad66c7459cf"

path: str = os.path.abspath(os.environ.get("ZLUDA", ".zluda"))

MIOpen_enabled: bool = False
hipBLASLt_enabled: bool = False
nightly: bool = False


class _Result(ctypes.Structure):
    _fields_ = [("return_code", ctypes.c_int), ("value", ctypes.c_ulonglong)]


def is_available() -> bool:
    return sys.platform == "win32" and gpu_backend.hip_sdk_path() is not None


def hip_sdk_major() -> int:
    version = gpu_backend.hip_sdk_version()
    return version[0] if version else 6


def download_url(use_nightly: bool = False) -> str:
    commit = os.environ.get("ZLUDA_HASH", DEFAULT_COMMIT)
    platform = "nightly-windows" if use_nightly else "windows"
    return f"https://github.com/lshqqytiger/ZLUDA/releases/download/rel.{commit}/ZLUDA-{platform}-rocm{hip_sdk_major()}-amd64.zip"


def is_installed() -> bool:
    return os.path.isfile(os.path.join(path, "nvcuda.dll"))


def install(use_nightly: bool = False):
    """Download and unpack ZLUDA into ``.zluda`` (no-op when already present)."""

    if is_installed():
        return

    import urllib.request

    url = download_url(use_nightly)
    archive = os.path.join(os.path.dirname(path) or ".", "_zluda.zip")

    print(f"Downloading ZLUDA from {url}")
    urllib.request.urlretrieve(url, archive)

    try:
        with zipfile.ZipFile(archive, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                info.filename = os.path.basename(info.filename)
                zf.extract(info, path)
    finally:
        try:
            os.remove(archive)
        except OSError:
            pass

    print(f"ZLUDA installed to {path}")


def uninstall():
    if os.path.exists(path):
        shutil.rmtree(path)


def _link_or_copy(src: str, dst: str):
    for attempt in (os.symlink, os.link, shutil.copyfile):
        try:
            attempt(src, dst)
        except Exception:
            continue
        else:
            return
    raise RuntimeError(f"Could not create {dst}")


def _torch_lib_dir() -> str | None:
    candidates = [p for p in site.getsitepackages() if p.endswith("site-packages")] or list(site.getsitepackages())
    for base in candidates:
        lib = os.path.join(base, "torch", "lib")
        if os.path.isdir(lib):
            return lib
    return None


def load():
    """
    Make the ZLUDA CUDA runtime the one PyTorch will bind to.

    Must be called *before* ``import torch``.
    """

    global MIOpen_enabled, hipBLASLt_enabled, nightly

    hip_root = gpu_backend.hip_sdk_path()
    if hip_root is None:
        raise RuntimeError("The AMD HIP SDK was not found. Install HIP SDK 6.2 (or 6.4) from https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html")

    if not is_installed():
        raise RuntimeError(f"ZLUDA is not installed in {path}")

    core = ctypes.windll.LoadLibrary(os.path.join(path, "nvcuda.dll"))
    core.zluda_get_hip_object.restype = _Result
    core.zluda_get_hip_object.argtypes = [ctypes.c_void_p, ctypes.c_int]
    ctypes.windll.LoadLibrary(os.path.join(path, "nvml.dll"))

    try:
        core.zluda_get_nightly_flag.restype = ctypes.c_int
        core.zluda_get_nightly_flag.argtypes = []
        nightly = core.zluda_get_nightly_flag() == 1
    except AttributeError:
        nightly = False

    hip_bin = os.path.join(hip_root, "bin")
    blaslt_library = os.environ.get("HIPBLASLT_TENSILE_LIBPATH", os.path.join(hip_bin, "hipblaslt", "library"))
    arch = gpu_backend.amd_arch() or ""

    hipBLASLt_enabled = nightly and os.path.isfile(os.path.join(hip_bin, "hipblaslt.dll")) and os.path.isfile(os.path.join(blaslt_library, f"TensileLibrary_lazy_{arch}.dat"))
    MIOpen_enabled = nightly and os.path.isfile(os.path.join(hip_bin, "MIOpen.dll"))

    for source, alias in DLL_MAPPING.items():
        if not os.path.exists(os.path.join(path, alias)):
            _link_or_copy(os.path.join(path, source), os.path.join(path, alias))

    if hipBLASLt_enabled and not os.path.exists(os.path.join(path, "cublasLt64_11.dll")):
        _link_or_copy(os.path.join(path, "cublasLt.dll"), os.path.join(path, "cublasLt64_11.dll"))

    if MIOpen_enabled and not os.path.exists(os.path.join(path, "cudnn64_9.dll")):
        _link_or_copy(os.path.join(path, "cudnn.dll"), os.path.join(path, "cudnn64_9.dll"))

    os.environ["PATH"] = path + os.pathsep + hip_bin + os.pathsep + os.environ.get("PATH", "")
    for directory in (path, hip_bin):
        try:
            os.add_dll_directory(directory)
        except (AttributeError, OSError):
            pass

    os.environ.setdefault("ZLUDA_COMGR_LOG_LEVEL", "1")

    torch_lib = _torch_lib_dir()
    if torch_lib is not None:
        os.environ["ZLUDA_NVRTC_LIB"] = os.path.join(torch_lib, "nvrtc64_112_0.dll")

    # Preloading by full path registers the module under its base name, so the
    # later lookups performed by `torch` resolve to ZLUDA instead of the CUDA
    # redistributables that ship inside the wheel.
    for library in HIPSDK_TARGETS:
        candidate = os.path.join(hip_bin, library)
        if os.path.isfile(candidate):
            ctypes.windll.LoadLibrary(candidate)

    for alias in DLL_MAPPING.values():
        ctypes.windll.LoadLibrary(os.path.join(path, alias))

    if hipBLASLt_enabled:
        os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "0")
        ctypes.windll.LoadLibrary(os.path.join(hip_bin, "hipblaslt.dll"))
        ctypes.windll.LoadLibrary(os.path.join(path, "cublasLt64_11.dll"))
    else:
        # cublasLt is not implemented without the nightly build; the ADDMM path
        # that uses it would otherwise crash.
        os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"

    if MIOpen_enabled:
        ctypes.windll.LoadLibrary(os.path.join(hip_bin, "MIOpen.dll"))
        ctypes.windll.LoadLibrary(os.path.join(path, "cudnn64_9.dll"))

    os.environ["SD_ZLUDA_ACTIVE"] = "1"

    print(f"ZLUDA: path='{path}' nightly={nightly} hipBLASLt={hipBLASLt_enabled} MIOpen={MIOpen_enabled}")
