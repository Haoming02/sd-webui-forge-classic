# https://github.com/comfyanonymous/ComfyUI/blob/v0.7.0/comfy/model_management.py
# Cherry-picked some good parts from ComfyUI with some bad parts fixed

"""
This file is part of ComfyUI.
Copyright (C) 2024 Comfy

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import gc
import importlib
import logging
import os
import platform
import sys
import time
import weakref
from contextlib import nullcontext
from enum import Enum
from typing import TYPE_CHECKING

import psutil
import torch

from backend.args import args
from backend.logging import setup_logger

if TYPE_CHECKING:
    from backend.patcher.base import ModelPatcher

logger = logging.getLogger("memory_management")
setup_logger(logger)

cpu = torch.device("cpu")


class VRAMState(Enum):
    DISABLED = 0  # No vram present: no need to move models to vram
    NO_VRAM = 1  # Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5  # No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.


class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2


# Determine VRAM State
lowvram_available = True
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

VAE_ALWAYS_TILED: bool = False

FLOAT8_TYPES: list[torch.dtype] = []

for dtype in ("e4m3fn", "e4m3fnuz", "e5m2", "e5m2fnuz", "e8m0fnu"):
    try:
        FLOAT8_TYPES.append(getattr(torch, f"float8_{dtype}"))
    except Exception:
        pass

try:
    torch_version: str = torch.__version__
    _ver: list[str] = torch_version.split(".", 2)
    torch_version_numeric: tuple[int, int] = (int(_ver[0]), int(_ver[1]))
except Exception:
    logger.warning("Could not determine PyTorch version...")
    torch_version = ""
    torch_version_numeric = None


def mac_version():
    try:
        return tuple(int(n) for n in platform.mac_ver()[0].split("."))
    except Exception:
        return None


if args.deterministic:
    logger.info("Using deterministic algorithms for PyTorch")
    torch.use_deterministic_algorithms(True, warn_only=True)

directml_enabled = False
if args.directml is not None:
    logger.warning("torch-directml barely works; please don't use it, there are better options...")
    import torch_directml

    directml_enabled = True
    device_index: int = max(0, args.directml)
    directml_device = torch_directml.device(device_index)
    logger.info("Using directml with device: {}".format(torch_directml.device_name(device_index)))
    lowvram_available = False

try:
    import intel_extension_for_pytorch as ipex  # noqa: F401

    _ = torch.xpu.device_count()
    xpu_available = torch.xpu.is_available()
except Exception:
    xpu_available = False

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except Exception:
    pass


if args.cpu:
    cpu_state = CPUState.CPU


def is_intel_xpu() -> bool:
    return cpu_state is CPUState.GPU and xpu_available


def is_nvidia() -> bool:
    return cpu_state is CPUState.GPU and torch.version.cuda


def is_amd() -> bool:
    return cpu_state is CPUState.GPU and torch.version.hip


def get_torch_device() -> torch.device:
    if directml_enabled:
        return directml_device
    if cpu_state is CPUState.MPS:
        return torch.device("mps")
    if cpu_state is CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        else:
            return torch.device(torch.cuda.current_device())


def get_total_memory(dev: torch.device = None, torch_total_too: bool = False):
    dev = dev or get_torch_device()

    if hasattr(dev, "type") and (dev.type == "cpu" or dev.type == "mps"):
        mem_total = psutil.virtual_memory().total
        mem_total_torch = mem_total
    else:
        if directml_enabled:
            mem_total = 1024 * 1024 * 1024  # TODO
            mem_total_torch = mem_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_total_xpu = torch.xpu.get_device_properties(dev).total_memory
            mem_total_torch = mem_reserved
            mem_total = mem_total_xpu
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_reserved = stats["reserved_bytes.all.current"]
            _, mem_total_cuda = torch.cuda.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_cuda

    if torch_total_too:
        return (mem_total, mem_total_torch)
    else:
        return mem_total


total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
total_ram = psutil.virtual_memory().total / (1024 * 1024)
logger.info("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))

try:
    logger.info("PyTorch Version: {}".format(torch_version))
    if (mac_ver := mac_version()) is not None:
        logger.info("Mac Version {}".format(mac_ver))
except Exception:
    pass

OOM_EXCEPTION = getattr(torch, "OutOfMemoryError", Exception)

if args.disable_xformers:
    XFORMERS_IS_AVAILABLE = False
else:
    try:
        import xformers
        import xformers.ops  # noqa: F401

        XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
    except Exception:
        XFORMERS_IS_AVAILABLE = False

XFORMERS_ENABLED_VAE = XFORMERS_IS_AVAILABLE or args.force_xformers_vae

if args.disable_sage:
    SAGE_IS_AVAILABLE = False
else:
    try:
        from sageattention import sageattn  # noqa: F401
    except Exception:
        SAGE_IS_AVAILABLE = False
    else:
        SAGE_IS_AVAILABLE = True

if args.disable_flash:
    FLASH_IS_AVAILABLE = False
else:
    try:
        from flash_attn import flash_attn_func  # noqa: F401
    except Exception:
        FLASH_IS_AVAILABLE = False
    else:
        FLASH_IS_AVAILABLE = True

try:
    import bitsandbytes  # noqa: F401
except Exception:
    BNB_IS_AVAILABLE = False
else:
    BNB_IS_AVAILABLE = True


def amd_min_version(device: torch.device = None, min_rdna_version: int = 0) -> bool:
    if not is_amd():
        return False

    if is_device_cpu(device):
        return False

    arch = torch.cuda.get_device_properties(device).gcnArchName
    if arch.startswith("gfx") and len(arch) == 7:
        try:
            cmp_rdna_version = int(arch[4]) + 2
        except Exception:
            cmp_rdna_version = 0
        if cmp_rdna_version >= min_rdna_version:
            return True

    return False


MIN_WEIGHT_MEMORY_RATIO = 0.4
if is_nvidia():
    MIN_WEIGHT_MEMORY_RATIO = 0.0

ENABLE_PYTORCH_ATTENTION = False
if args.use_pytorch_cross_attention:
    ENABLE_PYTORCH_ATTENTION = True
    XFORMERS_IS_AVAILABLE = False
    SAGE_IS_AVAILABLE = False
    FLASH_IS_AVAILABLE = False


if is_nvidia() and torch_version_numeric[0] >= 2:
    ENABLE_PYTORCH_ATTENTION = True
elif is_intel_xpu():
    ENABLE_PYTORCH_ATTENTION = True


SUPPORT_FP8_OPS: bool = None

if is_amd():
    AMD_RDNA2_AND_OLDER_ARCH = ("gfx1030", "gfx1031", "gfx1010", "gfx1011", "gfx1012", "gfx906", "gfx900", "gfx803")

    try:
        arch = torch.cuda.get_device_properties(get_torch_device()).gcnArchName
        if not (any((a in arch) for a in AMD_RDNA2_AND_OLDER_ARCH)):
            if os.getenv("ENABLE_MIOPEN") != "1":
                torch.backends.cudnn.enabled = False

        try:
            rocm_version = tuple(map(int, str(torch.version.hip).split(".")[:2]))
        except Exception:
            rocm_version = (6, -1)

        logger.info("AMD Arch: {}".format(arch))
        logger.info("ROCm Version: {}".format(rocm_version))
        if importlib.util.find_spec("triton") is not None:
            if torch_version_numeric >= (2, 7):
                if any((a in arch) for a in ["gfx90a", "gfx942", "gfx1100", "gfx1101", "gfx1151"]):
                    ENABLE_PYTORCH_ATTENTION = True
            if rocm_version >= (7, 0):
                if any((a in arch) for a in ["gfx1201"]):
                    ENABLE_PYTORCH_ATTENTION = True
        if torch_version_numeric >= (2, 7) and rocm_version >= (6, 4):
            if any((a in arch) for a in ["gfx1200", "gfx1201", "gfx950"]):
                SUPPORT_FP8_OPS = True

    except Exception:
        pass


if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


PRIORITIZE_FP16 = False

try:
    if args.fast_fp16 and (is_nvidia() or is_amd()):
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        logger.info("allow_fp16_accumulation: {}".format(torch.backends.cuda.matmul.allow_fp16_accumulation))
        PRIORITIZE_FP16 = True
except Exception:
    pass

if args.autotune and torch.cuda.is_available() and torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True
    logger.info("benchmark: {}".format(torch.backends.cudnn.benchmark))

try:
    if torch_version_numeric >= (2, 5):
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
except Exception:
    pass

if args.lowvram:
    set_vram_to = VRAMState.LOW_VRAM
    lowvram_available = True
elif args.novram:
    set_vram_to = VRAMState.NO_VRAM
elif args.highvram or args.gpu_only:
    vram_state = VRAMState.HIGH_VRAM

if args.force_fp32:
    logger.info("Forcing fp32")
    FORCE_FP32 = True
else:
    FORCE_FP32 = False

if lowvram_available:
    if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
        vram_state = set_vram_to

if cpu_state is not CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state is CPUState.MPS:
    vram_state = VRAMState.SHARED

logger.info(f"VRAM State: {vram_state.name}")

DISABLE_SMART_MEMORY = args.disable_smart_memory
if DISABLE_SMART_MEMORY:
    logger.info("Disabled Smart Memory Management")


def get_torch_device_name(device: torch.device) -> str:
    if hasattr(device, "type"):
        if device.type == "cuda":
            try:
                allocator_backend = f"- {torch.cuda.get_allocator_backend()}"
            except Exception:
                allocator_backend = ""
            return "{} ({}) {}".format(torch.cuda.get_device_name(device), device, allocator_backend)
        elif device.type == "xpu":
            return "{} ({})".format(torch.xpu.get_device_name(device), device)
        else:
            return "{}".format(device.type)
    elif is_intel_xpu():
        return "{} ({})".format(torch.xpu.get_device_name(device), device)
    else:
        return "{} (CUDA {})".format(torch.cuda.get_device_name(device), device)


try:
    torch_device_name: str = get_torch_device_name(get_torch_device())
    logger.info("Device: {}".format(torch_device_name))
except Exception:
    torch_device_name = ""
    logger.warning("Could not determine default device...")

if "rtx" in torch_device_name.lower() and not args.cuda_malloc:
    logger.warning("Hint: your device supports --cuda-malloc for potential speed improvements")


def bake_gguf_model(model):
    if getattr(model, "gguf_baked", False):
        return

    for p in model.parameters():
        gguf_cls = getattr(p, "gguf_cls", None)
        if gguf_cls is not None:
            gguf_cls.bake(p)

    global signal_empty_cache
    signal_empty_cache = True

    model.gguf_baked = True
    return model


current_loaded_models: list["LoadedModel"] = []


def module_size(module: torch.nn.Module) -> int:
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem


class LoadedModel:
    def __init__(self, model: "ModelPatcher"):
        self._set_model(model)
        self.device = model.load_device
        self.real_model = None
        self.currently_used = True
        self.model_finalizer = None
        self._patcher_finalizer = None

    def _set_model(self, model):
        self._model = weakref.ref(model)
        if model.parent is not None:
            self._parent_model = weakref.ref(model.parent)
            self._patcher_finalizer = weakref.finalize(model, self._switch_parent)

    def _switch_parent(self):
        model = self._parent_model()
        if model is not None:
            self._set_model(model)

    @property
    def model(self) -> "ModelPatcher":
        return self._model()

    def model_memory(self):
        return self.model.model_size()

    def model_loaded_memory(self):
        return self.model.loaded_size()

    def model_offloaded_memory(self):
        return self.model.model_size() - self.model.loaded_size()

    def model_memory_required(self, device):
        if device == self.model.current_loaded_device():
            return self.model_offloaded_memory()
        else:
            return self.model_memory()

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        # if self.model.loaded_size() > 0:
        use_more_vram = lowvram_model_memory
        if use_more_vram == 0:
            use_more_vram = 1e32
        self.model_use_more_vram(use_more_vram, force_patch_weights=force_patch_weights)

        real_model = self.model.model

        if is_intel_xpu() and not args.disable_ipex_optimize and "ipex" in globals() and real_model is not None:
            with torch.no_grad():
                real_model = ipex.optimize(real_model.eval(), inplace=True, graph_mode=True, concat_linear=True)

            global signal_empty_cache
            signal_empty_cache = True

        bake_gguf_model(real_model)

        self.model.refresh_loras()

        self.real_model = weakref.ref(real_model)
        self.model_finalizer = weakref.finalize(real_model, cleanup_models)
        return real_model

    def should_reload_model(self, force_patch_weights=False):
        if force_patch_weights and self.model.lowvram_patch_counter() > 0:
            return True
        return False

    def model_unload(self, memory_to_free=None, unpatch_weights=True):
        if memory_to_free is not None:
            if memory_to_free < self.model.loaded_size():
                freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
                if freed >= memory_to_free:
                    return False
        self.model.detach(unpatch_weights)
        self.model_finalizer.detach()
        self.model_finalizer = None
        self.real_model = None
        return True

    def model_use_more_vram(self, extra_memory, force_patch_weights=False):
        return self.model.partially_load(self.device, extra_memory, force_patch_weights=force_patch_weights)

    def __eq__(self, other):
        return self.model is other.model

    def __del__(self):
        if self._patcher_finalizer is not None:
            self._patcher_finalizer.detach()

    def is_dead(self):
        return self.real_model() is not None and self.model is None


def use_more_memory(extra_memory, loaded_models, device):
    for m in loaded_models:
        if m.device == device:
            extra_memory -= m.model_use_more_vram(extra_memory)
            if extra_memory <= 0:
                break


def offloaded_memory(loaded_models, device):
    offloaded_mem = 0
    for m in loaded_models:
        if m.device == device:
            offloaded_mem += m.model_offloaded_memory()
    return offloaded_mem


WINDOWS: bool = any(platform.win32_ver())

if args.reserve_vram is not None:
    EXTRA_RESERVED_VRAM = args.reserve_vram * 1024 * 1024 * 1024
    logger.info("Reserving {:0.0f} MB VRAM".format(EXTRA_RESERVED_VRAM / (1024 * 1024)))
else:
    EXTRA_RESERVED_VRAM = 400 * 1024 * 1024
    if WINDOWS:
        EXTRA_RESERVED_VRAM = 600 * 1024 * 1024
        if total_vram > (15 * 1024):
            EXTRA_RESERVED_VRAM += 100 * 1024 * 1024

SETTING_RESERVED_VRAM = -1


def set_reserved_memory(val: float):
    global SETTING_RESERVED_VRAM
    SETTING_RESERVED_VRAM = (1.0 - val) * total_vram * 1024 * 1024
    if SETTING_RESERVED_VRAM == 0.0:
        return
    logger.info("Manually Reserving {:0.0f} MB VRAM".format(SETTING_RESERVED_VRAM / (1024 * 1024)))


def extra_reserved_memory() -> float:
    return max(SETTING_RESERVED_VRAM, EXTRA_RESERVED_VRAM)


def minimum_inference_memory() -> float:
    return (1024 * 1024 * 1024) * 0.8 + extra_reserved_memory()


def free_memory(memory_required: float, device: torch.device, keep_loaded: list["LoadedModel"] = []):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.xpu.is_available():
        torch.xpu.synchronize()

    cleanup_models_gc()
    unloaded_model = []
    can_unload = []
    unloaded_models = []

    for i in range(len(current_loaded_models) - 1, -1, -1):
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded and not shift_model.is_dead():
                can_unload.append((-shift_model.model_offloaded_memory(), sys.getrefcount(shift_model.model), shift_model.model_memory(), i))
                shift_model.currently_used = False

    for x in sorted(can_unload):
        i = x[-1]
        memory_to_free = None
        if not DISABLE_SMART_MEMORY:
            free_mem = get_free_memory(device)
            if free_mem > memory_required:
                break
            memory_to_free = memory_required - free_mem
        logger.debug(f"Unloading {current_loaded_models[i].model.model.__class__.__name__}")
        if current_loaded_models[i].model_unload(memory_to_free):
            unloaded_model.append(i)

    for i in sorted(unloaded_model, reverse=True):
        unloaded_models.append(current_loaded_models.pop(i))

    if len(unloaded_model) > 0:
        soft_empty_cache()
    else:
        if vram_state is not VRAMState.HIGH_VRAM:
            mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()
    return unloaded_models


def load_models_gpu(models: list["ModelPatcher"], memory_required: float = 0, force_patch_weights: bool = False, minimum_memory_required: float = None, force_full_load: bool = False):
    execution_start_time = time.perf_counter()
    cleanup_models_gc(target=models)

    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required + extra_reserved_memory())
    if minimum_memory_required is None:
        minimum_memory_required = extra_mem
    else:
        minimum_memory_required = max(inference_memory, minimum_memory_required + extra_reserved_memory())

    models_temp = set()
    for m in models:
        models_temp.add(m)
        for mm in m.model_patches_models():
            models_temp.add(mm)

    models = models_temp

    models_to_load: list["LoadedModel"] = []

    for x in models:
        loaded_model = LoadedModel(x)
        try:
            loaded_model_index = current_loaded_models.index(loaded_model)
        except Exception:
            loaded_model_index = None

        if loaded_model_index is not None:
            loaded = current_loaded_models[loaded_model_index]
            loaded.currently_used = True
            models_to_load.append(loaded)
        else:
            if hasattr(x, "model"):
                logger.info(f"Requested to load {x.model.__class__.__name__}")
            models_to_load.append(loaded_model)

    for loaded_model in models_to_load:
        to_unload = []
        for i in range(len(current_loaded_models)):
            if loaded_model.model.is_clone(current_loaded_models[i].model):
                to_unload = [i] + to_unload
        for i in to_unload:
            model_to_unload = current_loaded_models.pop(i)
            model_to_unload.model.detach(unpatch_all=False)
            model_to_unload.model_finalizer.detach()

    total_memory_required = {}
    for loaded_model in models_to_load:
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory(total_memory_required[device] * 1.1 + extra_mem, device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_mem = get_free_memory(device)
            if free_mem < minimum_memory_required:
                models_l = free_memory(minimum_memory_required, device)
                logger.debug("{} models unloaded.".format(len(models_l)))

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            vram_set_state = VRAMState.DISABLED
        else:
            vram_set_state = vram_state
        lowvram_model_memory = 0
        if lowvram_available and vram_set_state in (VRAMState.LOW_VRAM, VRAMState.NORMAL_VRAM) and not force_full_load:
            loaded_memory = loaded_model.model_loaded_memory()
            current_free_mem = get_free_memory(torch_dev) + loaded_memory

            lowvram_model_memory = max(0, (current_free_mem - minimum_memory_required), min(current_free_mem * MIN_WEIGHT_MEMORY_RATIO, current_free_mem - minimum_inference_memory()))
            lowvram_model_memory = lowvram_model_memory - loaded_memory

            if lowvram_model_memory == 0:
                lowvram_model_memory = 0.1

        if vram_set_state is VRAMState.NO_VRAM:
            lowvram_model_memory = 0.1

        loaded_model.model_load(lowvram_model_memory, force_patch_weights=force_patch_weights)
        current_loaded_models.insert(0, loaded_model)

    if (moving_time := time.perf_counter() - execution_start_time) > 0.1:
        logger.info(f"Moving model(s) has taken {moving_time:.2f} seconds")


def load_model_gpu(model: "ModelPatcher"):
    return load_models_gpu([model])


def loaded_models(only_currently_used: bool = False) -> list["LoadedModel"]:
    output = []
    for m in current_loaded_models:
        if only_currently_used and not m.currently_used:
            continue
        output.append(m.model)
    return output


def cleanup_models_gc(*, target: list["ModelPatcher"] = []):
    _gc: bool = False
    _del: list[int] = []

    for i in range(len(current_loaded_models)):
        cur = current_loaded_models[i]
        if not cur.is_dead():
            continue
        if any(mdl.model is cur.real_model() for mdl in target):
            _del.append(i)
            break

        logger.info("Potential memory leak detected with model {}...".format(cur.real_model().__class__.__name__))
        _gc = True

    if not _gc and len(_del) == 0:
        return

    for i in reversed(_del):
        m = current_loaded_models.pop(i)
        del m

    gc.collect()
    soft_empty_cache()

    for mdl in current_loaded_models:
        if mdl.is_dead():
            logger.warning("Memory Leak with model {} !".format(mdl.real_model().__class__.__name__))


def cleanup_models():
    to_delete = []
    for i in range(len(current_loaded_models)):
        if current_loaded_models[i].real_model() is None:
            to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        del x


def dtype_size(dtype: torch.dtype) -> int:
    return getattr(dtype, "itemsize", 4)


def unet_offload_device():
    if vram_state is VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return cpu


def unet_initial_load_device(parameters: int, dtype: torch.dtype) -> torch.device:
    torch_dev = get_torch_device()
    if vram_state in (VRAMState.HIGH_VRAM, VRAMState.SHARED):
        return torch_dev

    cpu_dev = torch.device("cpu")
    if DISABLE_SMART_MEMORY or vram_state is VRAMState.NO_VRAM:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters
    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)

    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev


def maximum_vram_for_weights(device: torch.device = None) -> float:
    return get_total_memory(device) * 0.88 - minimum_inference_memory()


def unet_dtype(device: torch.device = None, model_params: int = 0, supported_dtypes: list[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32], weight_dtype: torch.dtype = None) -> torch.dtype:
    if model_params < 0:
        model_params = 1e32
    if args.fp32_unet:
        return torch.float32
    if args.bf16_unet:
        return torch.bfloat16
    if args.fp16_unet:
        return torch.float16
    if args.fp8_e4m3fn_unet:
        return torch.float8_e4m3fn
    if args.fp8_e5m2_unet:
        return torch.float8_e5m2
    if args.fp8_e8m0fnu_unet:
        return torch.float8_e8m0fnu

    if weight_dtype in FLOAT8_TYPES:
        if supports_fp8_compute(device):
            return weight_dtype

        free_model_memory = maximum_vram_for_weights(device)
        if model_params * 2 > free_model_memory:
            return weight_dtype

    if PRIORITIZE_FP16 or weight_dtype == torch.float16:
        if torch.float16 in supported_dtypes and should_use_fp16(device=device, model_params=model_params):
            return torch.float16

    for dt in supported_dtypes:
        if dt == torch.float16 and should_use_fp16(device=device, model_params=model_params):
            if torch.float16 in supported_dtypes:
                return torch.float16
        if dt == torch.bfloat16 and should_use_bf16(device, model_params=model_params):
            if torch.bfloat16 in supported_dtypes:
                return torch.bfloat16

    for dt in supported_dtypes:
        if dt == torch.float16 and should_use_fp16(device=device, model_params=model_params, manual_cast=True):
            if torch.float16 in supported_dtypes:
                return torch.float16
        if dt == torch.bfloat16 and should_use_bf16(device, model_params=model_params, manual_cast=True):
            if torch.bfloat16 in supported_dtypes:
                return torch.bfloat16

    return torch.float32


def inference_cast(weight_dtype: torch.device, inference_device: torch.device, supported_dtypes: list[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32]) -> torch.dtype:
    if weight_dtype == torch.float32:
        return weight_dtype

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=False)
    if fp16_supported and weight_dtype == torch.float16:
        return weight_dtype

    bf16_supported = should_use_bf16(inference_device)
    if bf16_supported and weight_dtype == torch.bfloat16:
        return weight_dtype

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=True)
    if PRIORITIZE_FP16 and fp16_supported and torch.float16 in supported_dtypes:
        return torch.float16

    for dt in supported_dtypes:
        if dt == torch.float16 and fp16_supported:
            return torch.float16
        if dt == torch.bfloat16 and bf16_supported:
            return torch.bfloat16

    return torch.float32


def text_encoder_offload_device() -> torch.device:
    return get_torch_device() if args.gpu_only else cpu


def text_encoder_device() -> torch.device:
    if args.text_enc_device is not None:
        return torch.device(args.text_enc_device)
    if args.gpu_only:
        return get_torch_device()
    if args.cpu_text_enc:
        return cpu
    elif vram_state in (VRAMState.HIGH_VRAM, VRAMState.NORMAL_VRAM):
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return cpu
    else:
        return cpu


def text_encoder_initial_device(load_device: torch.device, offload_device: torch.device, model_size: int = 0) -> torch.device:
    if load_device == offload_device or model_size <= 1024 * 1024 * 1024:
        return offload_device

    if is_device_mps(load_device):
        return load_device

    mem_l = get_free_memory(load_device)
    mem_o = get_free_memory(offload_device)
    if mem_l > (mem_o * 0.5) and model_size * 1.2 < mem_l:
        return load_device
    else:
        return offload_device


def text_encoder_dtype(device=None) -> torch.dtype:
    if args.fp8_e4m3fn_text_enc:
        return torch.float8_e4m3fn
    if args.fp8_e5m2_text_enc:
        return torch.float8_e5m2
    if args.fp16_text_enc:
        return torch.float16
    if args.bf16_text_enc:
        return torch.bfloat16
    if args.fp32_text_enc:
        return torch.float32

    return torch.float16


def intermediate_device() -> torch.device:
    return get_torch_device() if args.gpu_only else cpu


def vae_device() -> torch.device:
    if args.vae_device is not None:
        return torch.device(args.vae_device)
    return cpu if args.cpu_vae else get_torch_device()


def vae_offload_device() -> torch.device:
    return get_torch_device() if args.gpu_only else cpu


def vae_dtype(device=None, allowed_dtypes=None) -> torch.dtype:
    if args.fp16_vae:
        return torch.float16
    if args.bf16_vae:
        return torch.bfloat16
    if args.fp32_vae:
        return torch.float32

    if should_use_bf16(vae_device()):
        return torch.bfloat16

    return torch.float32


def get_autocast_device(dev: torch.device) -> str:
    return getattr(dev, "type", "cuda")


def supports_dtype(device: torch.device, dtype: torch.dtype) -> bool:
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return False
    if dtype == torch.float16:
        return True
    if dtype == torch.bfloat16:
        return True
    return False


def supports_cast(device: torch.device, dtype: torch.dtype) -> bool:
    if dtype == torch.float32:
        return True
    if dtype == torch.float16:
        return True
    if directml_enabled:
        return False
    if dtype == torch.bfloat16:
        return True
    if is_device_mps(device):
        return False
    if dtype == torch.float8_e4m3fn:
        return True
    if dtype == torch.float8_e5m2:
        return True
    return False


def pick_weight_dtype(dtype: torch.dtype, fallback_dtype: torch.dtype, device: torch.device = None) -> torch.dtype:
    if dtype is None:
        dtype = fallback_dtype
    elif dtype_size(dtype) > dtype_size(fallback_dtype):
        dtype = fallback_dtype

    if not supports_cast(device, dtype):
        dtype = fallback_dtype

    return dtype


def device_supports_non_blocking(device: torch.device) -> bool:
    if args.force_non_blocking:
        return True
    if is_device_mps(device):
        return False
    if is_intel_xpu():
        return False
    if args.deterministic:
        return False
    if directml_enabled:
        return False
    return True


def cast_to(weight: torch.Tensor, dtype: torch.dtype = None, device: torch.device = None, non_blocking: bool = False, copy: bool = False, context=nullcontext()):
    if device is None or weight.device == device:
        if not copy and (dtype is None or weight.dtype == dtype):
            return weight
        with context:
            return weight.to(dtype=dtype, copy=copy)

    with context:
        r = torch.empty_like(weight, dtype=dtype, device=device)
        r.copy_(weight, non_blocking=non_blocking)
        return r


def cast_to_device(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype, copy: bool = False):
    non_blocking = device_supports_non_blocking(device)
    return cast_to(tensor, dtype=dtype, device=device, non_blocking=non_blocking, copy=copy)


def xformers_enabled() -> bool:
    if cpu_state is not CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    if directml_enabled:
        return False
    return XFORMERS_IS_AVAILABLE


def xformers_enabled_vae() -> bool:
    if cpu_state is not CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    if directml_enabled:
        return False
    return XFORMERS_ENABLED_VAE


def sage_enabled() -> bool:
    if cpu_state is not CPUState.GPU:
        return False
    if not is_nvidia():
        return False
    return SAGE_IS_AVAILABLE


def flash_enabled() -> bool:
    if cpu_state is not CPUState.GPU:
        return False
    if not is_nvidia():
        return False
    return FLASH_IS_AVAILABLE


def bnb_enabled() -> bool:
    return BNB_IS_AVAILABLE


def pytorch_attention_enabled() -> bool:
    return ENABLE_PYTORCH_ATTENTION


def pytorch_attention_enabled_vae() -> bool:
    return ENABLE_PYTORCH_ATTENTION and not is_amd()


def pytorch_attention_flash_attention() -> bool:
    if ENABLE_PYTORCH_ATTENTION:
        if is_nvidia():
            return True
        if is_intel_xpu():
            return True
        if is_amd():
            return True
    return False


def force_upcast_attention_dtype() -> dict[torch.dtype, torch.dtype]:
    upcast: bool = args.force_upcast_attention

    macos_version = mac_version()
    if macos_version is not None and macos_version >= (14, 5):
        upcast = True

    return {torch.float16: torch.float32} if upcast else {}


def get_free_memory(dev: torch.device = None, torch_free_too: bool = False) -> int | tuple[int, int]:
    dev = dev or get_torch_device()

    if hasattr(dev, "type") and (dev.type == "cpu" or dev.type == "mps"):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024  # TODO
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_xpu = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_xpu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total


def cpu_mode() -> bool:
    return cpu_state is CPUState.CPU


def mps_mode() -> bool:
    return cpu_state is CPUState.MPS


def is_device_type(device: torch.device, type: str) -> bool:
    return getattr(device, "type", False) == type


def is_device_cpu(device: torch.device) -> bool:
    return is_device_type(device, "cpu")


def is_device_mps(device: torch.device) -> bool:
    return is_device_type(device, "mps")


def is_device_xpu(device: torch.device) -> bool:
    return is_device_type(device, "xpu")


def is_device_cuda(device: torch.device) -> bool:
    return is_device_type(device, "cuda")


def is_directml_enabled() -> bool:
    return directml_enabled


def should_use_fp16(device: torch.device = None, model_params: int = 0, prioritize_performance: bool = True, manual_cast: bool = False) -> bool:
    if device is not None and is_device_cpu(device):
        return False

    if args.force_fp16:
        return True

    if FORCE_FP32:
        return False

    if is_directml_enabled():
        return True

    if (device is not None and is_device_mps(device)) or mps_mode():
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        if torch_version_numeric < (2, 3):
            return True
        else:
            return torch.xpu.get_device_properties(device).has_fp16

    if torch.version.hip:
        return True

    props = torch.cuda.get_device_properties(device)
    if props.major >= 8:
        return True

    if props.major < 6:
        return False

    nvidia_10_series = ("1080", "1070", "titan x", "p3000", "p3200", "p4000", "p4200", "p5000", "p5200", "p6000", "1060", "1050", "p40", "p100", "p6", "p4")
    for x in nvidia_10_series:
        if x in props.name.lower():
            if WINDOWS or manual_cast:
                return True
            else:
                return False

    if manual_cast:
        free_model_memory = maximum_vram_for_weights(device)
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    if props.major < 7:
        return False

    nvidia_16_series = ("1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX", "T2000", "T1000", "T1200")
    for x in nvidia_16_series:
        if x in props.name:
            return False

    return True


def should_use_bf16(device: torch.device = None, model_params: int = 0, prioritize_performance: bool = True, manual_cast: bool = False) -> bool:
    if device is not None and is_device_cpu(device):
        return False

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if (device is not None and is_device_mps(device)) or mps_mode():
        if mac_version() < (14,):
            return False
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        if torch_version_numeric < (2, 3):
            return True
        else:
            return torch.xpu.is_bf16_supported()

    if is_amd():
        arch = torch.cuda.get_device_properties(device).gcnArchName
        if any((a in arch) for a in AMD_RDNA2_AND_OLDER_ARCH):
            if manual_cast:
                return True
            return False

    props = torch.cuda.get_device_properties(device)

    if props.major >= 8:
        return True

    bf16_works = torch.cuda.is_bf16_supported()

    if bf16_works and manual_cast:
        free_model_memory = maximum_vram_for_weights(device)
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    return False


def supports_fp8_compute(device: torch.device = None) -> bool:
    if SUPPORT_FP8_OPS is True:
        return True

    if not is_nvidia():
        return False

    props = torch.cuda.get_device_properties(device)
    if props.major >= 9:
        return True
    if props.major < 8:
        return False
    if props.minor < 9:
        return False

    if WINDOWS:
        if torch_version_numeric < (2, 4):
            return False
    else:
        if torch_version_numeric < (2, 3):
            return False

    return True


def extended_fp16_support() -> bool:
    return torch_version_numeric >= (2, 7)


LORA_COMPUTE_DTYPES: dict[torch.device, torch.dtype] = {}


def lora_compute_dtype(device: torch.device) -> torch.dtype:
    if device in LORA_COMPUTE_DTYPES:
        return LORA_COMPUTE_DTYPES[device]

    if should_use_fp16(device):
        dtype = torch.float16
    else:
        dtype = torch.float32

    LORA_COMPUTE_DTYPES[device] = dtype
    return dtype


signal_empty_cache = False


def soft_empty_cache(force=False):
    if cpu_state is CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    global signal_empty_cache
    signal_empty_cache = False


def unload_model(model: "ModelPatcher") -> bool:
    index = None
    for i, p in enumerate(current_loaded_models):
        if p.model == model:
            index = i
            break

    if index is not None:
        mdl = current_loaded_models.pop(index)
        del mdl
        return True

    return False


def unload_all_models():
    free_memory(1e30, get_torch_device())


# region Streams


STREAMS = {}

if args.cuda_stream is None:
    NUM_STREAMS = 0
else:
    NUM_STREAMS = int(args.cuda_stream)

if NUM_STREAMS > 0:
    logger.info("Using async weight offloading with {} streams".format(NUM_STREAMS))


def current_stream(device: torch.device):
    if device is None:
        return None
    if is_device_cuda(device):
        return torch.cuda.current_stream()
    elif is_device_xpu(device):
        return torch.xpu.current_stream()
    else:
        return None


stream_counters: dict[torch.device, int] = {}


def get_offload_stream(device: torch.device):
    if NUM_STREAMS == 0:
        return None
    if torch.compiler.is_compiling():
        return None

    stream_counter = stream_counters.get(device, 0)

    if device in STREAMS:
        ss = STREAMS[device]
        ss[stream_counter].wait_stream(current_stream(device))
        stream_counter = (stream_counter + 1) % len(ss)
        stream_counters[device] = stream_counter
        return ss[stream_counter]
    elif is_device_cuda(device):
        ss = []
        for _ in range(NUM_STREAMS):
            s1 = torch.cuda.Stream(device=device, priority=0)
            ss.append(s1)
        STREAMS[device] = ss
        s = ss[stream_counter]
        stream_counters[device] = stream_counter
        return s
    elif is_device_xpu(device):
        ss = []
        for _ in range(NUM_STREAMS):
            s1 = torch.xpu.Stream(device=device, priority=0)
            ss.append(s1)
        STREAMS[device] = ss
        s = ss[stream_counter]
        stream_counters[device] = stream_counter
        return s

    return None


def sync_stream(device: torch.device, stream):
    if stream is None or current_stream(device) is None:
        return
    current_stream(device).wait_stream(stream)


# region Pin


PINNED_MEMORY = {}
PINNING_ALLOWED_TYPES = "Parameter"

TOTAL_PINNED_MEMORY = 0
MAX_PINNED_MEMORY = -1

if args.pin_shared_memory:
    if is_nvidia() or is_amd():
        if WINDOWS:
            MAX_PINNED_MEMORY = get_total_memory(torch.device("cpu")) * 0.45  # Windows limit is apparently 50%
        else:
            MAX_PINNED_MEMORY = get_total_memory(torch.device("cpu")) * 0.95
        logger.info("Pinned Memory: {} MB".format(round(MAX_PINNED_MEMORY / (1024 * 1024))))


def discard_cuda_async_error():
    try:
        a = torch.tensor([1], dtype=torch.uint8, device=get_torch_device())
        b = torch.tensor([1], dtype=torch.uint8, device=get_torch_device())
        _ = a + b
        torch.cuda.synchronize()
    except torch.AcceleratorError:
        pass


def pin_memory(tensor):
    global TOTAL_PINNED_MEMORY
    if MAX_PINNED_MEMORY <= 0:
        return False

    if type(tensor).__name__ != PINNING_ALLOWED_TYPES:
        return False

    if not is_device_cpu(tensor.device):
        return False

    if tensor.is_pinned():
        return False

    if not tensor.is_contiguous():
        return False

    size = tensor.numel() * tensor.element_size()
    if (TOTAL_PINNED_MEMORY + size) > MAX_PINNED_MEMORY:
        return False

    ptr = tensor.data_ptr()
    if ptr == 0:
        return False

    if torch.cuda.cudart().cudaHostRegister(ptr, size, 1) == 0:
        PINNED_MEMORY[ptr] = size
        TOTAL_PINNED_MEMORY += size
        return True
    else:
        discard_cuda_async_error()

    return False


def unpin_memory(tensor):
    global TOTAL_PINNED_MEMORY
    if MAX_PINNED_MEMORY <= 0:
        return False

    if not is_device_cpu(tensor.device):
        return False

    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()

    size_stored = PINNED_MEMORY.get(ptr, None)
    if size_stored is None:
        return False

    if size != size_stored:
        return False

    if torch.cuda.cudart().cudaHostUnregister(ptr) == 0:
        TOTAL_PINNED_MEMORY -= PINNED_MEMORY.pop(ptr)
        if len(PINNED_MEMORY) == 0:
            TOTAL_PINNED_MEMORY = 0
        return True
    else:
        discard_cuda_async_error()

    return False


# region Conv3d


NVIDIA_CONV3D_WORKAROUND = False
try:
    if is_nvidia():
        cudnn_version = torch.backends.cudnn.version()
        if (91002 <= cudnn_version < 91500) and ((2, 9) <= torch_version_numeric <= (2, 10)):
            NVIDIA_CONV3D_WORKAROUND = True
except Exception:
    pass
