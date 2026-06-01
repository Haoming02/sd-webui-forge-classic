# https://github.com/Comfy-Org/ComfyUI/blob/v0.16.4/comfy/quant_ops.py

import comfy_kitchen as ck
import torch
from comfy_kitchen.tensor import (  # noqa
    QuantizedLayout,
    QuantizedTensor,
    TensorCoreFP8Layout,
    TensorCoreMXFP8Layout,
    TensorCoreNVFP4Layout,
    get_layout_class,
    register_layout_class,
    register_layout_op,
)

from . import float

if torch.version.cuda is None:
    ck.registry.disable("cuda")
else:
    cuda_version = tuple(map(int, str(torch.version.cuda).split(".")))
    if cuda_version < (13,):
        ck.registry.disable("cuda")

from backend.args import args

if not args.enable_triton_backend:
    ck.registry.disable("triton")


# region FP8 Layouts


class _TensorCoreFP8LayoutBase(TensorCoreFP8Layout):
    FP8_DTYPE = None

    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if cls.FP8_DTYPE is None:
            raise NotImplementedError(f"{cls.__name__} must define FP8_DTYPE")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if isinstance(scale, str) and scale == "recalculate":
            scale = torch.amax(tensor.abs()).to(dtype=torch.float32) / torch.finfo(cls.FP8_DTYPE).max
            if tensor.dtype not in [torch.float32, torch.bfloat16]:
                tensor_info = torch.finfo(tensor.dtype)
                scale = 1.0 / torch.clamp((1.0 / scale), min=tensor_info.min, max=tensor_info.max)

        if scale is None:
            scale = torch.ones((), device=tensor.device, dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=tensor.device, dtype=torch.float32)

        if stochastic_rounding > 0:
            if inplace_ops:
                tensor *= (1.0 / scale).to(tensor.dtype)
            else:
                tensor = tensor * (1.0 / scale).to(tensor.dtype)
            qdata = float.stochastic_rounding(tensor, dtype=cls.FP8_DTYPE, seed=stochastic_rounding)
        else:
            qdata = ck.quantize_per_tensor_fp8(tensor, scale, cls.FP8_DTYPE)

        params = cls.Params(scale=scale.float(), orig_dtype=orig_dtype, orig_shape=orig_shape)
        return qdata, params


class TensorCoreNVFP4Layout(TensorCoreNVFP4Layout):
    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if tensor.dim() != 2:
            raise ValueError(f"NVFP4 requires 2D tensor, got {tensor.dim()}D")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if scale is None or (isinstance(scale, str) and scale == "recalculate"):
            scale = torch.amax(tensor.abs()) / (ck.float_utils.F8_E4M3_MAX * ck.float_utils.F4_E2M1_MAX)

        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        scale = scale.to(device=tensor.device, dtype=torch.float32)

        padded_shape = cls.get_padded_shape(orig_shape)
        needs_padding = padded_shape != orig_shape

        if stochastic_rounding > 0:
            qdata, block_scale = float.stochastic_round_quantize_nvfp4_by_block(tensor, scale, pad_16x=needs_padding, seed=stochastic_rounding)
        else:
            qdata, block_scale = ck.quantize_nvfp4(tensor, scale, pad_16x=needs_padding)

        params = cls.Params(
            scale=scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
            block_scale=block_scale,
        )
        return qdata, params


class TensorCoreMXFP8Layout(TensorCoreMXFP8Layout):
    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if tensor.dim() != 2:
            raise ValueError(f"MXFP8 requires 2D tensor, got {tensor.dim()}D")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        padded_shape = cls.get_padded_shape(orig_shape)
        needs_padding = padded_shape != orig_shape

        if stochastic_rounding > 0:
            qdata, block_scale = float.stochastic_round_quantize_mxfp8_by_block(tensor, pad_32x=needs_padding, seed=stochastic_rounding)
        else:
            qdata, block_scale = ck.quantize_mxfp8(tensor, pad_32x=needs_padding)

        params = cls.Params(
            scale=block_scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
        )
        return qdata, params


class TensorCoreFP8E4M3Layout(_TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e4m3fn


class TensorCoreFP8E5M2Layout(_TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e5m2


TensorCoreFP8Layout = TensorCoreFP8E4M3Layout


# region Registry


register_layout_class("TensorCoreFP8Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreFP8E4M3Layout", TensorCoreFP8E4M3Layout)
register_layout_class("TensorCoreFP8E5M2Layout", TensorCoreFP8E5M2Layout)
register_layout_class("TensorCoreNVFP4Layout", TensorCoreNVFP4Layout)
register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout)

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
}
