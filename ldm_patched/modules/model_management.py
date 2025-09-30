"""
Credit: ComfyUI
https://github.com/comfyanonymous/ComfyUI

- Edited by. Forge Official
- Edited by. Haoming02
"""

import time
from enum import Enum
from functools import lru_cache

import psutil
import torch

from ldm_patched.modules.args_parser import args
from ldm_patched.modules.model_patcher import ModelPatcher
from modules_forge import stream


class VRAMState(Enum):
    DISABLED = 0
    NO_VRAM = 1
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5


class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2


# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except Exception:
    OOM_EXCEPTION = Exception

total_vram = 0

xpu_available = False

if args.pytorch_deterministic:
    print("Using deterministic algorithms for pytorch")
    torch.use_deterministic_algorithms(True, warn_only=True)

directml_enabled = False
if args.directml:
    import torch_directml

    directml_enabled = True
    device_index = args.gpu_device_id
    if device_index is None or device_index < 0:
        directml_device = torch_directml.device()
    else:
        directml_device = torch_directml.device(device_index)
    print("Using directml with device:", torch_directml.device_name(device_index))

try:
    import intel_extension_for_pytorch as ipex  # noqa

    if torch.xpu.is_available():
        xpu_available = True
except Exception:
    pass

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except Exception:
    pass

if args.always_cpu:
    cpu_state = CPUState.CPU


def is_intel_xpu():
    global cpu_state
    global xpu_available
    if cpu_state is CPUState.GPU:
        if xpu_available:
            return True
    return False


@lru_cache(maxsize=1, typed=False)
def get_torch_device():
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state is CPUState.MPS:
        return torch.device("mps")
    if cpu_state is CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu")
        else:
            return torch.device(torch.cuda.current_device())


def get_total_memory(dev=None, torch_total_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

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
            mem_total = torch.xpu.get_device_properties(dev).total_memory
            mem_total_torch = mem_reserved
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
print(f"Total VRAM {round(total_vram / 1024)} GB; Total RAM {round(total_ram / 1024)} GB")

if not args.always_normal_vram and not args.always_cpu:
    if total_vram <= 4096:
        print(
            """
            Trying to enable lowvram mode because your GPU seems to have 4GB or less VRAM.
            If you don't want this use: --always-normal-vram
            """.strip()
        )
        set_vram_to = VRAMState.LOW_VRAM


if args.fast_fp16:
    _ver = str(torch.version.__version__)
    if int(_ver[0]) >= 2 and int(_ver[2]) >= 7:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        print("allow_fp16_accumulation:", torch.backends.cuda.matmul.allow_fp16_accumulation)
    else:
        print("This version of pytorch does not support fp16_accumulation")

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
if args.disable_xformers:
    XFORMERS_IS_AVAILABLE = False
else:
    try:
        import xformers
        import xformers.ops  # noqa

        XFORMERS_IS_AVAILABLE = True
        try:
            XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
        except Exception:
            pass
        try:
            XFORMERS_VERSION = xformers.version.__version__
            print("xformers version:", XFORMERS_VERSION)
            if XFORMERS_VERSION.startswith("0.0.18"):
                from modules.errors import print_error_explanation

                print_error_explanation(
                    """
                    WARNING: This version of xformers has a major bug where you will get black images when generating high resolution images.
                    Please downgrade or upgrade xformers to a different version.
                    """.strip()
                )
                XFORMERS_ENABLED_VAE = False
        except Exception:
            pass
    except Exception:
        XFORMERS_IS_AVAILABLE = False

if args.disable_sage:
    SAGE_IS_AVAILABLE = False
else:
    try:
        from sageattention import sageattn  # noqa
    except ImportError:
        SAGE_IS_AVAILABLE = False
    else:
        SAGE_IS_AVAILABLE = True

if args.disable_flash:
    FLASH_IS_AVAILABLE = False
else:
    try:
        from flash_attn import flash_attn_func  # noqa
    except ImportError:
        FLASH_IS_AVAILABLE = False
    else:
        FLASH_IS_AVAILABLE = True


def is_nvidia():
    global cpu_state
    if cpu_state is CPUState.GPU:
        if torch.version.cuda:
            return True
    return False


ENABLE_PYTORCH_ATTENTION = False
if args.attention_pytorch:
    ENABLE_PYTORCH_ATTENTION = True
    SAGE_IS_AVAILABLE = False
    FLASH_IS_AVAILABLE = False
    XFORMERS_IS_AVAILABLE = False

VAE_DTYPE = torch.float32

try:
    if is_nvidia():
        torch_version = torch.version.__version__
        if int(torch_version[0]) >= 2:
            ENABLE_PYTORCH_ATTENTION = True
            if torch.cuda.is_bf16_supported() and torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8:
                VAE_DTYPE = torch.bfloat16
    if is_intel_xpu():
        ENABLE_PYTORCH_ATTENTION = True
except Exception:
    pass

if is_intel_xpu():
    VAE_DTYPE = torch.bfloat16

if args.vae_in_cpu:
    VAE_DTYPE = torch.float32

if args.vae_in_fp16:
    VAE_DTYPE = torch.float16
elif args.vae_in_bf16:
    VAE_DTYPE = torch.bfloat16
elif args.vae_in_fp32:
    VAE_DTYPE = torch.float32


VAE_ALWAYS_TILED = False


if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

if args.always_low_vram:
    set_vram_to = VRAMState.LOW_VRAM
elif args.always_no_vram:
    set_vram_to = VRAMState.NO_VRAM
elif args.always_high_vram or args.always_gpu:
    vram_state = VRAMState.HIGH_VRAM

FORCE_FP32 = False
if args.all_in_fp32:
    print("Forcing FP32")
    FORCE_FP32 = True

FORCE_FP16 = False
if args.all_in_fp16:
    print("Forcing FP16")
    FORCE_FP16 = True

if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
    vram_state = set_vram_to

if cpu_state is not CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state is CPUState.MPS:
    vram_state = VRAMState.SHARED

print(f"Set vram state to: {vram_state.name}")

ALWAYS_VRAM_OFFLOAD = args.always_offload_from_vram
if ALWAYS_VRAM_OFFLOAD:
    print("Always offload VRAM")

PIN_SHARED_MEMORY = args.pin_shared_memory
if PIN_SHARED_MEMORY:
    print("Always pin shared GPU memory")


def get_torch_device_name(device):
    if hasattr(device, "type"):
        if device.type == "cuda":
            try:
                allocator_backend = torch.cuda.get_allocator_backend()
            except Exception:
                allocator_backend = ""
            return "{} {} : {}".format(device, torch.cuda.get_device_name(device), allocator_backend)
        else:
            return "{}".format(device.type)
    elif is_intel_xpu():
        return "{} {}".format(device, torch.xpu.get_device_name(device))
    else:
        return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))


try:
    torch_device_name = get_torch_device_name(get_torch_device())
    print("Device:", torch_device_name)
except Exception:
    torch_device_name = ""
    print("Could not identify torch device")

if "rtx" in torch_device_name.lower():
    if not (args.cuda_malloc or args.cuda_stream or args.pin_shared_memory):
        print("Hint: your device supports --cuda-malloc for potential speed improvements")
        print("Hint: your device supports --cuda-stream for potential speed improvements")
        print("Hint: your device supports --pin-shared-memory for potential speed improvements")


current_loaded_models: list["LoadedModel"] = []


def module_size(module, exclude_device=None):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]

        if exclude_device is not None and exclude_device == t.device:
            continue

        module_mem += t.nelement() * t.element_size()
    return module_mem


class LoadedModel:
    def __init__(self, model: ModelPatcher, memory_required: int):
        self.model = model
        self.memory_required = memory_required
        self.model_accelerated = False
        self.device = model.load_device

    def model_memory(self) -> int:
        return self.model.model_size()

    def model_memory_required(self, device: torch.device) -> int:
        return module_size(self.model.model, exclude_device=device)

    def model_load(self, async_kept_memory: int = -1):
        patch_model_to = None
        disable_async_load = async_kept_memory < 0

        if disable_async_load:
            patch_model_to = self.device

        self.model.model_patches_to(device=self.device, dtype=self.model.model_dtype())

        try:
            self.real_model = self.model.patch_model(device_to=patch_model_to)
        except Exception as e:
            self.model.unpatch_model(self.model.offload_device)
            self.model_unload()
            soft_empty_cache()
            raise e

        if not disable_async_load:
            flag = "ASYNC" if stream.using_stream else "SYNC"
            print(
                f"[Memory Management] Requested {flag} Preserved Memory (MB) = ",
                async_kept_memory / (1024 * 1024),
            )
            real_async_memory = 0
            mem_counter = 0
            for m in self.real_model.modules():
                if hasattr(m, "ldm_patched_cast_weights"):
                    m.prev_ldm_patched_cast_weights = m.ldm_patched_cast_weights
                    m.ldm_patched_cast_weights = True
                    module_mem = module_size(m)
                    if mem_counter + module_mem < async_kept_memory:
                        m.to(self.device)
                        mem_counter += module_mem
                    else:
                        real_async_memory += module_mem
                        m.to(self.model.offload_device)
                        if PIN_SHARED_MEMORY and is_device_cpu(self.model.offload_device):
                            m._apply(lambda x: x.pin_memory())
                elif hasattr(m, "weight"):
                    m.to(self.device)
                    mem_counter += module_size(m)
                    print(f"[Memory Management] {flag} Loader Disabled for ", m)
            print(
                f"[Memory Management] Parameters Loaded to {flag} Stream (MB) = ",
                real_async_memory / (1024 * 1024),
            )
            print(
                "[Memory Management] Parameters Loaded to GPU (MB) = ",
                mem_counter / (1024 * 1024),
            )

            self.model_accelerated = True

        if is_intel_xpu() and not args.disable_ipex_hijack:
            self.real_model = torch.xpu.optimize(
                self.real_model.eval(),
                inplace=True,
                auto_kernel_selection=True,
                graph_mode=True,
            )

        return self.real_model

    def model_unload(self, *, avoid_model_moving: bool = False):
        if self.model_accelerated:
            for m in self.real_model.modules():
                if hasattr(m, "prev_ldm_patched_cast_weights"):
                    m.ldm_patched_cast_weights = m.prev_ldm_patched_cast_weights
                    del m.prev_ldm_patched_cast_weights

            self.model_accelerated = False

        if avoid_model_moving:
            self.model.unpatch_model()
        else:
            self.model.unpatch_model(device_to=self.model.offload_device)
            self.model.model_patches_to(self.model.offload_device)

    def __eq__(self, other: "LoadedModel"):
        return self.model is other.model


def minimum_inference_memory():
    return 1024 * 1024 * 1024


def unload_model_clones(model):
    to_unload = [i for i in range(len(current_loaded_models)) if model.is_clone(current_loaded_models[i].model)]

    for i in reversed(to_unload):
        m = current_loaded_models.pop(i)
        m.model_unload(avoid_model_moving=True)
        del m

    if len(to_unload) > 0:
        print(f"Reusing {len(to_unload)} loaded model{'s' if len(to_unload) > 1 else ''}")
        soft_empty_cache()


def free_memory(memory_required, device, keep_loaded=[]):
    offload_everything = ALWAYS_VRAM_OFFLOAD or vram_state is VRAMState.NO_VRAM
    unloaded_model = False
    for i in range(len(current_loaded_models) - 1, -1, -1):
        if not offload_everything:
            if get_free_memory(device) > memory_required:
                break
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded:
                m = current_loaded_models.pop(i)
                m.model_unload()
                del m
                unloaded_model = True

    if unloaded_model:
        soft_empty_cache()
    else:
        if vram_state != VRAMState.HIGH_VRAM:
            mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()


def load_models_gpu(models, memory_required=0):
    execution_start_time = time.perf_counter()
    extra_mem = max(minimum_inference_memory(), memory_required)

    models_to_load = []
    models_already_loaded = []

    for x in models:
        load_model = LoadedModel(x, memory_required=memory_required)

        if load_model in current_loaded_models:
            index = current_loaded_models.index(load_model)
            loaded_model = current_loaded_models.pop(index)
            current_loaded_models.insert(0, loaded_model)
            models_already_loaded.append(loaded_model)
            del load_model
        else:
            if hasattr(x, "model"):
                print(f"Loading Model: {x.model.__class__.__name__}")
            models_to_load.append(load_model)

    if len(models_to_load) == 0:
        devs = set(map(lambda a: a.device, models_already_loaded))
        for d in devs:
            if d != torch.device("cpu"):
                free_memory(extra_mem, d, models_already_loaded)

        if (moving_time := time.perf_counter() - execution_start_time) > 0.1:
            print(f"Memory cleanup has taken {moving_time:.2f} seconds")

        return

    print(f"Begin to load {len(models_to_load)} model{'s' if len(models_to_load) > 1 else ''}")

    total_memory_required = {}
    for loaded_model in models_to_load:
        unload_model_clones(loaded_model.model)
        mem = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)
        total_memory_required[loaded_model.device] = mem

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory(
                total_memory_required[device] * 1.25 + extra_mem,
                device,
                models_already_loaded,
            )

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            vram_set_state = VRAMState.DISABLED
        else:
            vram_set_state = vram_state

        async_kept_memory = -1

        if vram_set_state in (VRAMState.LOW_VRAM, VRAMState.NORMAL_VRAM):
            model_memory = loaded_model.model_memory_required(torch_dev)
            current_free_mem = get_free_memory(torch_dev)
            minimal_inference_memory = minimum_inference_memory()
            estimated_remaining_memory = current_free_mem - model_memory - minimal_inference_memory

            print("[Memory Management] Current Free GPU Memory (MB) = ", current_free_mem / (2**20))
            print("[Memory Management] Model Memory (MB) = ", model_memory / (2**20))
            print("[Memory Management] Minimal Inference Memory (MB) = ", minimal_inference_memory / (2**20))
            print("[Memory Management] Estimated Remaining GPU Memory (MB) = ", estimated_remaining_memory / (2**20))

            if estimated_remaining_memory < 0:
                vram_set_state = VRAMState.LOW_VRAM
                async_kept_memory = (current_free_mem - minimal_inference_memory) / 1.25
                async_kept_memory = int(max(0, async_kept_memory))

        if vram_set_state is VRAMState.NO_VRAM:
            async_kept_memory = 0

        loaded_model.model_load(async_kept_memory)
        current_loaded_models.insert(0, loaded_model)

    moving_time = time.perf_counter() - execution_start_time
    print(f"Moving model(s) has taken {moving_time:.2f} seconds")


def load_model_gpu(model):
    return load_models_gpu([model])


def cleanup_models():
    unload_all_models()


def dtype_size(dtype: torch.dtype):
    return getattr(dtype, "itemsize", 4)


@lru_cache(maxsize=1, typed=False)
def unet_dtype(device: torch.device = None, model_params: int = 0):
    if args.unet_in_bf16:
        return torch.bfloat16
    if args.unet_in_fp16:
        return torch.float16
    if prefer_fp8() or args.unet_in_fp8_e4m3fn:
        return torch.float8_e4m3fn
    if args.unet_in_fp8_e5m2:
        return torch.float8_e5m2
    if should_use_fp16(device=device, model_params=model_params, manual_cast=True):
        return torch.float16
    return torch.float32


def unet_initial_load_device(parameters: int, dtype: torch.dtype):
    torch_dev = get_torch_device()
    if vram_state is VRAMState.HIGH_VRAM:
        return torch_dev

    cpu_dev = torch.device("cpu")
    if ALWAYS_VRAM_OFFLOAD:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters
    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)

    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev


def unet_offload_device():
    return get_torch_device() if vram_state is VRAMState.HIGH_VRAM else torch.device("cpu")


def unet_manual_cast(weight_dtype: torch.dtype, inference_device: torch.device):
    """`None` means no manual cast"""
    if weight_dtype is torch.float32:
        return None

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=False)
    if fp16_supported and weight_dtype is torch.float16:
        return None

    if fp16_supported:
        return torch.float16
    else:
        return torch.float32


def text_encoder_offload_device():
    return get_torch_device() if args.always_gpu else torch.device("cpu")


def text_encoder_device():
    if not (args.clip_in_gpu or args.always_gpu):
        return torch.device("cpu")

    if args.always_gpu:
        return get_torch_device()
    elif vram_state is VRAMState.HIGH_VRAM or vram_state is VRAMState.NORMAL_VRAM:
        if is_intel_xpu():
            return torch.device("cpu")
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def text_encoder_dtype(device=None):
    if not (args.clip_in_gpu or args.always_gpu):
        return torch.float16

    if args.clip_in_fp8_e4m3fn:
        return torch.float8_e4m3fn
    elif args.clip_in_fp8_e5m2:
        return torch.float8_e5m2
    elif args.clip_in_fp16:
        return torch.float16
    elif args.clip_in_fp32:
        return torch.float32

    if is_device_cpu(device):
        return torch.float16

    return torch.float16


def vae_offload_device():
    return get_torch_device() if args.always_gpu else torch.device("cpu")


def vae_device():
    return torch.device("cpu") if args.vae_in_cpu else get_torch_device()


def check_fp16():
    if get_torch_device().type in ("mps", "cpu"):
        return False

    from modules.shared import opts

    return opts.prefer_vae_precision_float16


def vae_dtype():
    if VAE_DTYPE is torch.bfloat16 and check_fp16():
        return torch.float16
    return VAE_DTYPE


def get_autocast_device(dev):
    return getattr(dev, "type", "cuda")


def intermediate_device():
    return get_torch_device() if args.always_gpu else torch.device("cpu")


def prefer_fp8():
    if not torch.cuda.is_available():
        return False

    if not is_nvidia():
        return False

    from modules.shared import opts

    return opts.fp8_storage


def support_fp8():
    if not prefer_fp8():
        return False
    if not is_nvidia():
        return False

    if int(torch_version[0]) < 2 or int(torch_version[2]) < 4:
        return False

    device = get_torch_device()
    props = torch.cuda.get_device_properties(device)

    if props.major >= 9:
        return True
    elif props.major == 8 and props.minor >= 9:
        return True
    else:
        return False


def supports_dtype(device, dtype):  # TODO
    if dtype is torch.float32:
        return True
    if is_device_cpu(device):
        return False
    if dtype is torch.float16:
        return True
    if dtype is torch.bfloat16:
        return True
    return False


def device_supports_non_blocking(device):
    if is_device_mps(device):
        return False
    if is_intel_xpu():
        return False
    if args.pytorch_deterministic:
        return False
    if directml_enabled:
        return False
    return True


def __cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False):
    if device is None or weight.device == device:
        if not copy:
            if dtype is None or weight.dtype == dtype:
                return weight
        return weight.to(dtype=dtype, copy=copy)

    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight, non_blocking=non_blocking)
    return r


def cast_to_device(tensor, device, dtype, copy=False):
    non_blocking = device_supports_non_blocking(device)
    return __cast_to(tensor, dtype=dtype, device=device, non_blocking=non_blocking, copy=copy)


def xformers_enabled():
    if cpu_state != CPUState.GPU:
        return False
    if not is_nvidia():
        return False
    return XFORMERS_IS_AVAILABLE


def sage_enabled():
    if cpu_state != CPUState.GPU:
        return False
    if not is_nvidia():
        return False
    return SAGE_IS_AVAILABLE


def flash_enabled():
    if cpu_state != CPUState.GPU:
        return False
    if not is_nvidia():
        return False
    return FLASH_IS_AVAILABLE


def xformers_enabled_vae():
    if not xformers_enabled():
        return False

    return XFORMERS_ENABLED_VAE


def pytorch_attention_enabled():
    return ENABLE_PYTORCH_ATTENTION


def get_free_memory(dev=None, torch_free_too=False):
    if dev is None:
        dev = get_torch_device()

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
            mem_allocated = stats["allocated_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = torch.xpu.get_device_properties(dev).total_memory - mem_allocated
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = max(mem_reserved - mem_active, 0)
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total


def cpu_mode():
    return cpu_state is CPUState.CPU


def mps_mode():
    return cpu_state is CPUState.MPS


def is_device_cpu(device):
    return getattr(device, "type", "cuda") == "cpu"


def is_device_mps(device):
    return getattr(device, "type", "cuda") == "mps"


@lru_cache(maxsize=4, typed=False)
def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if is_device_cpu(device):
        return False
    if FORCE_FP16:
        return True
    if is_device_mps(device):
        return False
    if FORCE_FP32:
        return False
    if directml_enabled:
        return False
    if cpu_mode() or mps_mode():
        return False
    if is_intel_xpu():
        return True
    if torch.version.hip:
        return True

    props = torch.cuda.get_device_properties("cuda")
    if props.major >= 7:
        return True

    return False


def soft_empty_cache(force=False):
    if cpu_state is CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        if force or is_nvidia():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def unload_all_models():
    free_memory(float("inf"), get_torch_device())
    if vram_state != VRAMState.HIGH_VRAM:
        free_memory(float("inf"), torch.device("cpu"))
