# https://github.com/Comfy-Org/ComfyUI/blob/v0.27.0/comfy/quant_ops.py

import comfy_kitchen as ck
import torch
from comfy_kitchen.tensor import (  # noqa
    QuantizedTensor,
    TensorCoreConvRotW4A4Layout,
    TensorCoreFP8Layout,
    TensorCoreMXFP8Layout,
    TensorCoreNVFP4Layout,
    TensorWiseINT8Layout,
    get_layout_class,
    register_layout_class,
)

if torch.version.cuda is None:
    ck.registry.disable("cuda")
else:
    cuda_version = tuple(map(int, str(torch.version.cuda).split(".")))
    if cuda_version < (13,):
        ck.registry.disable("cuda")

from backend.args import args

if args.enable_triton_backend:
    try:
        import triton  # noqa
    except ImportError:
        ck.registry.disable("triton")
else:
    ck.registry.disable("triton")

import importlib.metadata

ver = importlib.metadata.version("comfy-kitchen")

print(f"Comfy-Kitchen {ver}:", {k: v["available"] and not v["disabled"] for k, v in ck.list_backends().items()})


# region Registry


register_layout_class("TensorCoreFP8Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreFP8E4M3Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreFP8E5M2Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreNVFP4Layout", TensorCoreNVFP4Layout)
register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout)
register_layout_class("TensorWiseINT8Layout", TensorWiseINT8Layout)
register_layout_class("TensorCoreConvRotW4A4Layout", TensorCoreConvRotW4A4Layout)


QUANT_ALGOS = {
    "float8_e4m3fn": {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E4M3Layout",
    },
    "float8_e5m2": {
        "storage_t": torch.float8_e5m2,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E5M2Layout",
    },
    "nvfp4": {
        "storage_t": torch.uint8,
        "parameters": {"weight_scale", "weight_scale_2", "input_scale"},
        "comfy_tensor_layout": "TensorCoreNVFP4Layout",
        "group_size": 16,
    },
    "mxfp8": {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreMXFP8Layout",
        "group_size": 32,
    },
    "int8_tensorwise": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale"},
        "comfy_tensor_layout": "TensorWiseINT8Layout",
        "quantize_input": False,
    },
    "convrot_w4a4": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale"},
        "comfy_tensor_layout": "TensorCoreConvRotW4A4Layout",
        "quantize_input": False,
    },
}


# region float


def stochastic_rounding(value: torch.Tensor, dtype: torch.dtype, seed: int = 0):
    if dtype is torch.float32:
        return value.to(dtype=torch.float32)
    if dtype is torch.float16:
        return value.to(dtype=torch.float16)
    if dtype is torch.bfloat16:
        return value.to(dtype=torch.bfloat16)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        rng = torch.randint(0, 256, value.size(), dtype=torch.uint8, layout=value.layout, device=value.device, generator=generator)
        return ck.stochastic_rounding_fp8(value, rng, dtype)

    return value.to(dtype=dtype)
