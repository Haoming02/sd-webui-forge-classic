# Cherry-picked some good parts from ComfyUI with some bad parts fixed

import platform
import time
from enum import Enum

import psutil
import torch

from backend import stream, utils
from backend.args import args
from backend.patcher.base import ModelPatcher

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
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

total_vram = 0

xpu_available = False

if args.pytorch_deterministic:
    print("Using deterministic algorithms for pytorch")
    torch.use_deterministic_algorithms(True, warn_only=True)

directml_enabled = False
if args.directml is not None:
    import torch_directml

    directml_enabled = True
    device_index = args.directml
    if device_index < 0:
        directml_device = torch_directml.device()
    else:
        directml_device = torch_directml.device(device_index)
    print("Using directml with device: {}".format(torch_directml.device_name(device_index)))

try:
    import intel_extension_for_pytorch as ipex

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
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False


def get_torch_device():
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
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
            mem_total_torch = mem_reserved
            mem_total = torch.xpu.get_device_properties(dev).total_memory
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
print("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))

try:
    print("pytorch version: {}".format(torch.version.__version__))
except Exception:
    pass

OOM_EXCEPTION = getattr(torch, "OutOfMemoryError", Exception)

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
    if cpu_state == CPUState.GPU:
        if torch.version.cuda:
            return True
    return False


ENABLE_PYTORCH_ATTENTION = False
if args.attention_pytorch:
    ENABLE_PYTORCH_ATTENTION = True
    XFORMERS_IS_AVAILABLE = False

VAE_DTYPES = [torch.float32]

try:
    if is_nvidia():
        torch_version = torch.version.__version__
        if int(torch_version[0]) >= 2:
            if ENABLE_PYTORCH_ATTENTION is False and args.attention_split is False:
                ENABLE_PYTORCH_ATTENTION = True
            if torch.cuda.is_bf16_supported() and torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8:
                VAE_DTYPES = [torch.bfloat16] + VAE_DTYPES
    if is_intel_xpu():
        if args.attention_split is False:
            ENABLE_PYTORCH_ATTENTION = True
except Exception:
    pass

if is_intel_xpu():
    VAE_DTYPES = [torch.bfloat16] + VAE_DTYPES

if args.vae_in_cpu:
    VAE_DTYPES = [torch.float32]

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
FORCE_FP16 = False
if args.all_in_fp32:
    print("Forcing FP32, if this improves things please report it.")
    FORCE_FP32 = True

if args.all_in_fp16:
    print("Forcing FP16.")
    FORCE_FP16 = True

if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
    vram_state = set_vram_to

if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
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
    print("Device: {}".format(torch_device_name))
except Exception:
    torch_device_name = ""
    print("Could not pick default device.")

if "rtx" in torch_device_name.lower():
    if not args.cuda_malloc:
        print("Hint: your device supports --cuda-malloc for potential speed improvements.")


current_loaded_models: list["LoadedModel"] = []


def state_dict_size(sd, exclude_device=None):
    module_mem = 0
    for k in sd:
        t = sd[k]

        if exclude_device is not None:
            if t.device == exclude_device:
                continue

        module_mem += t.nelement() * t.element_size()
    return module_mem


def state_dict_parameters(sd):
    module_mem = 0
    for k, v in sd.items():
        module_mem += v.nelement()
    return module_mem


def state_dict_dtype(state_dict: dict):
    if state_dict.pop("scaled_fp8", None) is not None:
        return torch.float8_e4m3fn
    if state_dict.pop("transformer.scaled_fp8", None) is not None:
        return torch.float8_e4m3fn

    for k, v in state_dict.items():
        if hasattr(v, "gguf_cls"):
            return "gguf"
        if "bitsandbytes__nf4" in k:
            return "nf4"
        if "bitsandbytes__fp4" in k:
            return "fp4"

    dtype_counts = {}

    for tensor in state_dict.values():
        dtype = tensor.dtype
        if dtype in dtype_counts:
            dtype_counts[dtype] += 1
        else:
            dtype_counts[dtype] = 1

    major_dtype = None
    max_count = 0

    for dtype, count in dtype_counts.items():
        if count > max_count:
            max_count = count
            major_dtype = dtype

    return major_dtype


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


def module_size(module: torch.nn.Module, exclude_device: torch.device = None, include_device: torch.device = None, return_split=False):
    module_mem = 0
    weight_mem = 0
    weight_patterns = "weight"

    for k, t in module.state_dict().items():
        if exclude_device is not None and t.device == exclude_device:
            continue
        if include_device is not None and t.device != include_device:
            continue

        module_mem += t.nelement() * t.element_size()

        if return_split and k == weight_patterns:
            weight_mem += t.nelement() * t.element_size()

    if return_split:
        return module_mem, weight_mem, module_mem - weight_mem

    return module_mem


def module_move(module, device, recursive=True, excluded_patterns=[]):
    if recursive:
        return module.to(device=device)

    for k, p in module.named_parameters(recurse=False, remove_duplicate=True):
        if k in excluded_patterns:
            continue
        setattr(module, k, utils.tensor2parameter(p.to(device=device)))

    return module


def build_module_profile(model: ModelPatcher, swap_memory):
    all_modules = []
    gpu_modules = []
    extras_modules = []
    mem_counter = 0

    for m in model.to_load_list():
        if hasattr(m, "parameters_manual_cast"):
            m.total_mem, m.weight_mem, m.extra_mem = module_size(m, return_split=True)
            all_modules.append(m)
        else:
            m.total_mem, m.weight_mem, m.extra_mem = module_size(m, return_split=True)
            gpu_modules.append(m)
            mem_counter += m.total_mem

    for m in sorted(all_modules.copy(), key=lambda x: x.extra_mem):
        if mem_counter + m.extra_mem < swap_memory:
            all_modules.remove(m)
            extras_modules.append(m)
            mem_counter += m.extra_mem

    # for m in sorted(extras_modules.copy(), key=lambda x: x.weight_mem):
    #     if mem_counter + m.weight_mem < swap_memory:
    #         extras_modules.remove(m)
    #         gpu_modules.append(m)
    #         mem_counter += m.weight_mem

    return gpu_modules, extras_modules, all_modules


class LoadedModel:
    def __init__(self, model: ModelPatcher):
        self.model = model
        self.model_accelerated = False
        self.device = model.load_device
        self.inclusive_memory = 0
        self.exclusive_memory = 0

    def compute_inclusive_exclusive_memory(self):
        self.inclusive_memory = module_size(self.model.model, include_device=self.device)
        self.exclusive_memory = module_size(self.model.model, exclude_device=self.device)

    def model_load(self, cpu_swap_memory=-1):
        patch_model_to = None
        full_load = cpu_swap_memory < 0

        if full_load:
            patch_model_to = self.device

        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        try:
            self.real_model = self.model.forge_patch_model(patch_model_to)
            self.model.current_device = self.model.load_device
        except Exception as e:
            self.model.forge_unpatch_model(self.model.offload_device)
            self.model_unload()
            raise e

        if not full_load:
            gpu_modules, gpu_modules_only_extras, cpu_modules = build_module_profile(self.model, cpu_swap_memory)
            pin_memory = PIN_SHARED_MEMORY and is_device_cpu(self.model.offload_device)

            mem_counter = 0
            swap_counter = 0

            for m in gpu_modules:
                m.to(self.device)
                mem_counter += m.total_mem

            for m in cpu_modules:
                m.prev_parameters_manual_cast = m.parameters_manual_cast
                m.parameters_manual_cast = True
                m.to(self.model.offload_device)
                if pin_memory:
                    m._apply(lambda x: x.pin_memory())
                swap_counter += m.total_mem

            for m in gpu_modules_only_extras:
                m.prev_parameters_manual_cast = m.parameters_manual_cast
                m.parameters_manual_cast = True
                module_move(m, device=self.device, recursive=False, excluded_patterns=["weight"])
                if hasattr(m, "weight") and m.weight is not None:
                    if pin_memory:
                        m.weight = utils.tensor2parameter(m.weight.to(self.model.offload_device).pin_memory())
                    else:
                        m.weight = utils.tensor2parameter(m.weight.to(self.model.offload_device))
                mem_counter += m.extra_mem
                swap_counter += m.weight_mem

            swap_flag = "Shared" if PIN_SHARED_MEMORY else "CPU"
            method_flag = "asynchronous" if stream.should_use_stream() else "blocked"
            print(f"{swap_flag} Swap Loaded ({method_flag} method): {swap_counter / (1024 * 1024):.2f} MB, GPU Loaded: {mem_counter / (1024 * 1024):.2f} MB")

            self.model_accelerated = True

            global signal_empty_cache
            signal_empty_cache = True

        bake_gguf_model(self.real_model)

        self.model.refresh_loras()

        if is_intel_xpu() and not args.disable_ipex_hijack:
            self.real_model = torch.xpu.optimize(self.real_model.eval(), inplace=True, auto_kernel_selection=True, graph_mode=True)

        return self.real_model

    def model_unload(self, avoid_model_moving=False):
        if self.model_accelerated:
            for m in self.real_model.modules():
                if hasattr(m, "prev_parameters_manual_cast"):
                    m.parameters_manual_cast = m.prev_parameters_manual_cast
                    del m.prev_parameters_manual_cast

            self.model_accelerated = False

        if avoid_model_moving:
            self.model.forge_unpatch_model()
        else:
            self.model.forge_unpatch_model(self.model.offload_device)
            self.model.model_patches_to(self.model.offload_device)

    def __eq__(self, other: "LoadedModel"):
        return self.model is other.model


WINDOWS = any(platform.win32_ver())

EXTRA_RESERVED_VRAM = 400 * 1024 * 1024
if WINDOWS:
    EXTRA_RESERVED_VRAM = 600 * 1024 * 1024
    if total_vram > (15 * 1024):
        EXTRA_RESERVED_VRAM = 800 * 1024 * 1024


current_inference_memory = 1024 * 1024 * 1024
"""set by refresh_memory_management_settings in main_entry.py"""


def minimum_inference_memory():
    return current_inference_memory * 0.8 + EXTRA_RESERVED_VRAM


def unload_model_clones(model):
    to_unload = [i for i in range(len(current_loaded_models)) if model.is_clone(current_loaded_models[i].model)]

    for i in reversed(to_unload):
        m = current_loaded_models.pop(i)
        m.model_unload(avoid_model_moving=True)
        del m

    if len(to_unload) > 0:
        soft_empty_cache()


def free_memory(memory_required, device, keep_loaded=[], free_all=False):
    if free_all:
        memory_required = 1e30
        print(f"[Unload] Trying to free all memory for {device} with {len(keep_loaded)} models keep loaded ... ", end="")
        offload_everything = True
    else:
        print(f"[Unload] Trying to free {memory_required / (1024 * 1024):.2f} MB for {device} with {len(keep_loaded)} models keep loaded ... ", end="")
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
                print(f"Unload model {m.model.model.__class__.__name__} ", end="")
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

    print("Done.")


def compute_memory_for_cpu_swap(current_free_mem, inference_memory, previously_loaded):
    maximum_memory_available = current_free_mem + previously_loaded - inference_memory
    suggestion = max(maximum_memory_available / 1.2, maximum_memory_available - EXTRA_RESERVED_VRAM)
    return int(max(0, suggestion - previously_loaded))


def load_models_gpu(models, memory_required=0, hard_memory_preservation=0):
    execution_start_time = time.perf_counter()
    memory_for_inference = minimum_inference_memory()
    memory_to_free = max(memory_for_inference, max(memory_required, hard_memory_preservation) + EXTRA_RESERVED_VRAM)

    models_to_load: list[LoadedModel] = []
    models_already_loaded: list[LoadedModel] = []

    for x in models:
        load_model = LoadedModel(x)

        if load_model in current_loaded_models:
            index = current_loaded_models.index(load_model)
            loaded_model = current_loaded_models.pop(index)
            current_loaded_models.insert(0, loaded_model)
            models_already_loaded.append(loaded_model)
            del load_model
        else:
            models_to_load.append(load_model)

    if len(models_to_load) == 0:
        devs = set(map(lambda a: a.device, models_already_loaded))
        for d in devs:
            if d != torch.device("cpu"):
                free_memory(memory_to_free, d, models_already_loaded)

        if (moving_time := time.perf_counter() - execution_start_time) > 0.1:
            print(f"Memory cleanup has taken {moving_time:.2f} seconds")

        return

    for loaded_model in models_to_load:
        unload_model_clones(loaded_model.model)

    total_memory_required = {}
    for loaded_model in models_to_load:
        loaded_model.compute_inclusive_exclusive_memory()
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.exclusive_memory

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory(total_memory_required[device] * 1.2 + memory_to_free, device, models_already_loaded)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_mem = get_free_memory(device)
            if free_mem < memory_for_inference:
                free_memory(memory_for_inference, device)

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            vram_set_state = VRAMState.DISABLED
        else:
            vram_set_state = vram_state

        cpu_swap_memory = -1

        if vram_set_state in (VRAMState.LOW_VRAM, VRAMState.NORMAL_VRAM):
            model_require = loaded_model.exclusive_memory
            previously_loaded = loaded_model.inclusive_memory
            current_free_mem = get_free_memory(torch_dev)
            estimated_remaining_memory = current_free_mem - model_require - memory_for_inference

            print(f"[Memory Management] Target: {loaded_model.model.model.__class__.__name__}, Free GPU: {current_free_mem / (1024 * 1024):.2f} MB, Model Require: {model_require / (1024 * 1024):.2f} MB, Previously Loaded: {previously_loaded / (1024 * 1024):.2f} MB, Inference Require: {memory_for_inference / (1024 * 1024):.2f} MB, Remaining: {estimated_remaining_memory / (1024 * 1024):.2f} MB, ", end="")

            if estimated_remaining_memory < 0:
                vram_set_state = VRAMState.LOW_VRAM
                cpu_swap_memory = compute_memory_for_cpu_swap(current_free_mem, memory_for_inference, previously_loaded)

        if vram_set_state == VRAMState.NO_VRAM:
            cpu_swap_memory = 0

        loaded_model.model_load(cpu_swap_memory)
        current_loaded_models.insert(0, loaded_model)

    moving_time = time.perf_counter() - execution_start_time
    print(f"Moving model(s) has taken {moving_time:.2f} seconds")


def load_model_gpu(model):
    return load_models_gpu([model])


def cleanup_models():
    to_delete = [i for i in range(len(current_loaded_models)) if current_loaded_models[i].real_model() is None]

    for i in reversed(to_delete):
        m = current_loaded_models.pop(i)
        del m

    if len(to_delete) > 0:
        soft_empty_cache()


def dtype_size(dtype):
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        try:
            dtype_size = dtype.itemsize
        except Exception:  # Old pytorch doesn't have .itemsize
            pass
    return dtype_size


def unet_offload_device():
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return torch.device("cpu")


def unet_initial_load_device(parameters, dtype):
    torch_dev = get_torch_device()
    if vram_state == VRAMState.HIGH_VRAM:
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


def unet_dtype(device=None, model_params=0, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    if args.unet_in_bf16:
        return torch.bfloat16

    if args.unet_in_fp16:
        return torch.float16

    if args.unet_in_fp8_e4m3fn:
        return torch.float8_e4m3fn

    if args.unet_in_fp8_e5m2:
        return torch.float8_e5m2

    for candidate in supported_dtypes:
        if candidate == torch.float16:
            if should_use_fp16(device, model_params=model_params, prioritize_performance=True, manual_cast=True):
                return candidate
        if candidate == torch.bfloat16:
            if should_use_bf16(device, model_params=model_params, prioritize_performance=True, manual_cast=True):
                return candidate

    return torch.float32


def get_computation_dtype(inference_device, parameters=0, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    for candidate in supported_dtypes:
        if candidate == torch.float16:
            if should_use_fp16(inference_device, model_params=parameters, prioritize_performance=True, manual_cast=False):
                return candidate
        if candidate == torch.bfloat16:
            if should_use_bf16(inference_device, model_params=parameters, prioritize_performance=True, manual_cast=False):
                return candidate

    return torch.float32


def text_encoder_offload_device():
    if args.always_gpu:
        return get_torch_device()
    else:
        return torch.device("cpu")


def text_encoder_device():
    if args.always_gpu:
        return get_torch_device()
    elif args.clip_in_cpu:
        return torch.device("cpu")
    elif vram_state == VRAMState.HIGH_VRAM or vram_state == VRAMState.NORMAL_VRAM:
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def text_encoder_dtype(device=None):
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


def intermediate_device():
    if args.always_gpu:
        return get_torch_device()
    else:
        return torch.device("cpu")


def vae_device():
    if args.vae_in_cpu:
        return torch.device("cpu")
    return get_torch_device()


def vae_offload_device():
    if args.always_gpu:
        return get_torch_device()
    else:
        return torch.device("cpu")


def vae_dtype(device=None, allowed_dtypes=[]):
    global VAE_DTYPES
    if args.vae_in_fp16:
        return torch.float16
    elif args.vae_in_bf16:
        return torch.bfloat16
    elif args.vae_in_fp32:
        return torch.float32

    for d in allowed_dtypes:
        if d == torch.float16 and should_use_fp16(device, prioritize_performance=False):
            return d
        if d in VAE_DTYPES:
            return d

    return VAE_DTYPES[0]


print(f"VAE dtype preferences: {VAE_DTYPES} -> {vae_dtype()}")


def get_autocast_device(dev):
    if hasattr(dev, "type"):
        return dev.type
    return "cuda"


def supports_dtype(device, dtype):  # TODO
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return False
    if dtype == torch.float16:
        return True
    if dtype == torch.bfloat16:
        return True
    return False


def supports_cast(device, dtype):  # TODO
    if dtype == torch.float32:
        return True
    if dtype == torch.float16:
        return True
    if directml_enabled:  # TODO: test this
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


def pick_weight_dtype(dtype, fallback_dtype, device=None):
    if dtype is None:
        dtype = fallback_dtype
    elif dtype_size(dtype) > dtype_size(fallback_dtype):
        dtype = fallback_dtype

    if not supports_cast(device, dtype):
        dtype = fallback_dtype

    return dtype


def device_supports_non_blocking(device):
    if is_device_mps(device):
        return False  # pytorch bug? mps doesn't support non blocking
    if is_intel_xpu():
        return False
    if args.pytorch_deterministic:  # TODO: figure out why deterministic breaks non blocking from gpu to cpu (previews)
        return False
    if directml_enabled:
        return False
    return True


def device_should_use_non_blocking(device):
    if not device_supports_non_blocking(device):
        return False
    return False
    # return True #TODO: figure out why this causes memory issues on Nvidia and possibly others


def force_channels_last():
    if args.force_channels_last:
        return True

    # TODO
    return False


def cast_to_device(tensor, device, dtype, copy=False):
    device_supports_cast = False
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        device_supports_cast = True
    elif tensor.dtype == torch.bfloat16:
        if hasattr(device, "type") and device.type.startswith("cuda"):
            device_supports_cast = True
        elif is_intel_xpu():
            device_supports_cast = True

    non_blocking = device_should_use_non_blocking(device)

    if device_supports_cast:
        if copy:
            if tensor.device == device:
                return tensor.to(dtype, copy=copy, non_blocking=non_blocking)
            return tensor.to(device, copy=copy, non_blocking=non_blocking).to(dtype, non_blocking=non_blocking)
        else:
            return tensor.to(device, non_blocking=non_blocking).to(dtype, non_blocking=non_blocking)
    else:
        return tensor.to(device, dtype, copy=copy, non_blocking=non_blocking)


def xformers_enabled():
    global directml_enabled
    global cpu_state
    if cpu_state != CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    if directml_enabled:
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
    enabled = xformers_enabled()
    if not enabled:
        return False

    return XFORMERS_ENABLED_VAE


def pytorch_attention_enabled():
    global ENABLE_PYTORCH_ATTENTION
    return ENABLE_PYTORCH_ATTENTION


def pytorch_attention_flash_attention():
    global ENABLE_PYTORCH_ATTENTION
    if ENABLE_PYTORCH_ATTENTION:
        # TODO: more reliable way of checking for flash attention?
        if is_nvidia():  # pytorch flash attention only works on Nvidia
            return True
        if is_intel_xpu():
            return True
    return False


def force_upcast_attention_dtype():
    upcast = args.force_upcast_attention
    try:
        if tuple(int(n) for n in platform.mac_ver()[0].split(".")) >= (14, 5):
            upcast = True
    except Exception:
        pass

    return {torch.float16: torch.float32} if upcast else None


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
    global cpu_state
    return cpu_state == CPUState.CPU


def mps_mode():
    global cpu_state
    return cpu_state == CPUState.MPS


def is_device_type(device, type):
    if hasattr(device, "type"):
        if device.type == type:
            return True
    return False


def is_device_cpu(device):
    return is_device_type(device, "cpu")


def is_device_mps(device):
    return is_device_type(device, "mps")


def is_device_cuda(device):
    return is_device_type(device, "cuda")


def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    global directml_enabled

    if device is not None:
        if is_device_cpu(device):
            return False

    if FORCE_FP16:
        return True

    if device is not None:
        if is_device_mps(device):
            return True

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if mps_mode():
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        return True

    if torch.version.hip:
        return True

    props = torch.cuda.get_device_properties("cuda")
    if props.major >= 8:
        return True

    if props.major < 6:
        return False

    nvidia_10_series = ["1080", "1070", "titan x", "p3000", "p3200", "p4000", "p4200", "p5000", "p5200", "p6000", "1060", "1050", "p40", "p100", "p6", "p4"]
    for x in nvidia_10_series:
        if x in props.name.lower():
            if manual_cast:
                # For storage dtype
                free_model_memory = get_free_memory() * 0.9 - minimum_inference_memory()
                if (not prioritize_performance) or model_params * 4 > free_model_memory:
                    return True
            else:
                # For computation dtype
                return False  # Flux on 1080 can store model in fp16 to reduce swap, but computation must be fp32, otherwise super slow.

    if props.major < 7:
        return False

    # FP16 is just broken on these cards
    nvidia_16_series = ["1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX", "T2000", "T1000", "T1200"]
    for x in nvidia_16_series:
        if x in props.name:
            return False

    return True


def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if device is not None:
        if is_device_cpu(device):  # TODO ? bf16 works on CPU but is extremely slow
            return False

    if device is not None:
        if is_device_mps(device):
            return True

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if mps_mode():
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        return True

    if device is None:
        device = torch.device("cuda")

    props = torch.cuda.get_device_properties(device)
    if props.major >= 8:
        return True

    if torch.cuda.is_bf16_supported():
        # This device is an old enough device but bf16 somewhat reports supported.
        # So in this case bf16 should only be used as storage dtype
        if manual_cast:
            # For storage dtype
            free_model_memory = get_free_memory() * 0.9 - minimum_inference_memory()
            if (not prioritize_performance) or model_params * 4 > free_model_memory:
                return True

    return False


signal_empty_cache = False


def soft_empty_cache(force=False):
    global cpu_state, signal_empty_cache
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        if force or is_nvidia():  # This seems to make things worse on ROCm so I only do it for cuda
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    signal_empty_cache = False


def unload_all_models():
    free_memory(float("inf"), get_torch_device(), free_all=True)
    if vram_state != VRAMState.HIGH_VRAM:
        free_memory(float("inf"), torch.device("cpu"), free_all=True)


# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.71/comfy/ops.py#L58
NVIDIA_CONV3D_WORKAROUND = False
try:
    if is_nvidia():
        cudnn_version = torch.backends.cudnn.version()
        torch_version = str(torch.version.__version__)
        if (cudnn_version >= 91002 and cudnn_version < 91500) and (int(torch_version[0]) >= 2 and int(torch_version[2]) >= 9 and int(torch_version[2]) <= 10):
            NVIDIA_CONV3D_WORKAROUND = True
except Exception:
    pass
else:
    from functools import wraps

    _forward = torch.nn.Conv3d._conv_forward

    @wraps(_forward)
    def patched_forward(self, input, weight, bias, *args, **kwargs):
        if NVIDIA_CONV3D_WORKAROUND and weight.dtype in (torch.float16, torch.bfloat16):
            out = torch.cudnn_convolution(input, weight, self.padding, self.stride, self.dilation, self.groups, benchmark=False, deterministic=False, allow_tf32=True)
            if bias is not None:
                out += bias.reshape((1, -1) + (1,) * (out.ndim - 2))
            return out
        else:
            return _forward(self, input, weight, bias, *args, **kwargs)

    torch.nn.Conv3d._conv_forward = patched_forward
