# Copyright (C) 2024 Forge - Establish the Structures
# Copyright (C) 2025 ComfyUI - where Optimization is Stolen
# Copyright (C) 2026 Haoming02 - Burnt the Kitchen

import contextlib
import inspect
import time
from typing import Callable, Union

import torch

from backend import memory_management, stream, utils
from backend.args import args, dynamic_args


def gqa_repeat_factor(query_heads: int, key_heads: int, value_heads: int) -> int:
    assert key_heads == value_heads
    if query_heads == key_heads:
        return 1
    assert query_heads % key_heads == 0
    return query_heads // key_heads


def repeat_kv_for_gqa(k: torch.Tensor, v: torch.Tensor, query_heads: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    n_rep = gqa_repeat_factor(query_heads, k.shape[head_dim], v.shape[head_dim])
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=head_dim)
        v = v.repeat_interleave(n_rep, dim=head_dim)
    return k, v


def scaled_dot_product_attention(q, k, v, *args, **kwargs):
    attn_mask = args[0] if len(args) > 0 else kwargs.get("attn_mask")
    if kwargs.get("enable_gqa", False) and attn_mask is not None:
        k, v = repeat_kv_for_gqa(k, v, q.shape[-3], -3)
        kwargs["enable_gqa"] = False
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)


try:
    if torch.cuda.is_available():
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if "set_priority" in inspect.signature(sdpa_kernel).parameters:
            SDPA_BACKEND_PRIORITY = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]

            def scaled_dot_product_attention(q, k, v, *args, **kwargs):
                attn_mask = args[0] if len(args) > 0 else kwargs.get("attn_mask")
                if kwargs.get("enable_gqa", False) and attn_mask is not None and not memory_management.is_nvidia():
                    k, v = repeat_kv_for_gqa(k, v, q.shape[-3], -3)
                    kwargs["enable_gqa"] = False
                with sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True):
                    if kwargs.get("enable_gqa", False) and attn_mask is not None and q.shape[-3] != k.shape[-3]:
                        dropout_p = args[1] if len(args) > 1 else kwargs.get("dropout_p", 0.0)
                        is_causal = args[2] if len(args) > 2 else kwargs.get("is_causal", False)
                        params = torch.backends.cuda.SDPAParams(q, k, v, attn_mask, dropout_p, is_causal, True)
                        supports_native_gqa = torch.backends.cuda.can_use_flash_attention(params) or torch.backends.cuda.can_use_cudnn_attention(params) or torch.backends.cuda.can_use_efficient_attention(params)
                        if not supports_native_gqa:
                            k, v = repeat_kv_for_gqa(k, v, q.shape[-3], -3)
                            kwargs["enable_gqa"] = False
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)

except Exception:
    pass


# region Cast


def get_weight_and_bias(layer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Forge-Specific Function for on-the-fly LoRA"""

    weight: torch.Tensor = getattr(layer, "weight", None)
    for f in getattr(layer, "weight_function", []):
        weight = f(weight)

    bias: torch.Tensor = getattr(layer, "bias", None)
    for f in getattr(layer, "bias_function", []):
        bias = f(bias)

    return weight, bias


def weights_manual_cast(
    layer: Union[torch.nn.Module, "ForgeWeights"],
    x: torch.Tensor = None,
    *,
    dtype: torch.dtype = None,
    device: torch.device = None,
    bias_dtype: torch.dtype = None,
    weight_fn: Callable = None,
    bias_fn: Callable = None,
    skip_weight_dtype: bool = False,
    skip_bias_dtype: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    """
    Cast layer to input dtype/device
    * Reference: https://github.com/Comfy-Org/ComfyUI/blob/v0.16.4/comfy/ops.py#L210
    """

    if x is not None:
        target_dtype, target_device = x.dtype, x.device
    else:
        target_dtype, target_device = dtype, device

    non_blocking = memory_management.device_supports_non_blocking(target_device)
    weight, bias = None, None

    weight_has_function: bool = len(layer.weight_function) > 0 or weight_fn is not None
    bias_has_function: bool = len(layer.bias_function) > 0 or bias_fn is not None

    weight_args = dict(device=target_device, dtype=dtype or target_dtype, non_blocking=non_blocking)
    if skip_weight_dtype or weight_has_function:
        weight_args.pop("dtype")

    bias_args = dict(device=target_device, dtype=bias_dtype or target_dtype, non_blocking=non_blocking)
    if skip_bias_dtype or bias_has_function:
        bias_args.pop("dtype")

    if stream.should_use_stream():
        offload_stream = memory_management.get_offload_stream(target_device)
        context = stream.stream_context()(offload_stream)
    else:
        offload_stream = None
        context = None

    if layer.weight is not None:
        weight = memory_management.cast_to(
            layer.weight,
            **weight_args,
            copy=weight_has_function,
            context=context if layer.weight.device != target_device else None,
        )

    if layer.bias is not None:
        bias = memory_management.cast_to(
            layer.bias,
            **bias_args,
            copy=bias_has_function,
            context=context if layer.bias.device != target_device else None,
        )

    memory_management.sync_stream(target_device, offload_stream)

    weight_a = weight
    bias_a = bias

    if weight_has_function:
        if isinstance(weight, QuantizedTensor):
            weight = weight.dequantize()
        if weight_fn is not None:
            weight = weight_fn(weight)
        if not skip_weight_dtype:
            weight = weight.to(dtype=target_dtype)
        for f in layer.weight_function:
            weight = f(weight)

    if bias_has_function:
        if bias_fn is not None:
            bias = bias_fn(bias)
        if not skip_bias_dtype:
            bias = bias.to(dtype=target_dtype)
        for f in layer.bias_function:
            bias = f(bias)

    return weight, bias, (offload_stream, weight_a, bias_a)


@contextlib.contextmanager
def main_stream_worker(weight, bias, offload_stream: tuple[torch.Stream, torch.Tensor, torch.Tensor]):
    yield
    if offload_stream is None:
        return
    os, weight_a, bias_a = offload_stream
    if os is None:
        return
    if weight_a is not None:
        device = weight_a.device
    elif bias_a is not None:
        device = bias_a.device
    else:
        return
    os.wait_stream(memory_management.current_stream(device))


current_device: torch.device = None
current_dtype: torch.dtype = None
current_manual_cast_enabled: bool = False


# region Forge OPs


class ForgeWeights:
    parameters_manual_cast = False
    weight_function = []
    bias_function = []


class ForgeOperations:
    class Linear(torch.nn.Linear, ForgeWeights):
        def __init__(self, *args, **kwargs):
            kwargs["device"] = current_device
            kwargs["dtype"] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.linear(x, weight, bias)
            else:
                weight, bias = get_weight_and_bias(self)
                return torch.nn.functional.linear(x, weight, bias)

    class Conv1d(torch.nn.Conv1d, ForgeWeights):

        def __init__(self, *args, **kwargs):
            kwargs["device"] = current_device
            kwargs["dtype"] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return self._conv_forward(x, weight, bias)
            else:
                weight, bias = get_weight_and_bias(self)
                return super()._conv_forward(x, weight, bias)

    class Conv2d(torch.nn.Conv2d, ForgeWeights):

        def __init__(self, *args, **kwargs):
            kwargs["device"] = current_device
            kwargs["dtype"] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return self._conv_forward(x, weight, bias)
            else:
                weight, bias = get_weight_and_bias(self)
                return super()._conv_forward(x, weight, bias)

    class Conv3d(torch.nn.Conv3d, ForgeWeights):

        def __init__(self, *args, **kwargs):
            kwargs["device"] = current_device
            kwargs["dtype"] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def _conv_forward(self, input, weight, bias, autopad=None, *args, **kwargs):
            if autopad == "causal_zero":
                weight = weight[:, :, -input.shape[2] :, :, :]
            if memory_management.NVIDIA_CONV3D_WORKAROUND and weight.dtype in (torch.float16, torch.bfloat16):
                out = torch.cudnn_convolution(input, weight, self.padding, self.stride, self.dilation, self.groups, benchmark=False, deterministic=False, allow_tf32=True)
                if bias is not None:
                    out += bias.reshape((1, -1) + (1,) * (out.ndim - 2))
                return out
            else:
                return super()._conv_forward(input, weight, bias, *args, **kwargs)

        def forward(self, x, *, autopad=None):
            if self.parameters_manual_cast or autopad is not None:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return self._conv_forward(x, weight, bias, autopad)
            else:
                weight, bias = get_weight_and_bias(self)
                return super()._conv_forward(x, weight, bias)

    class GroupNorm(torch.nn.GroupNorm, ForgeWeights):

        def __init__(self, *args, **kwargs):
            kwargs["device"] = current_device
            kwargs["dtype"] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.group_norm(x, self.num_groups, weight, bias, self.eps)
            else:
                return super().forward(x)

    class LayerNorm(torch.nn.LayerNorm, ForgeWeights):

        def __init__(self, *args, **kwargs):
            kwargs["device"] = current_device
            kwargs["dtype"] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
            else:
                return super().forward(x)

    class RMSNorm(torch.nn.RMSNorm, ForgeWeights):

        def __init__(self, *args, add=False, **kwargs):
            kwargs["device"] = current_device
            kwargs["dtype"] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled
            self.bias = None
            self.add = add  # used by llama.py

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if prefix + "scale" in state_dict:  # Flux
                state_dict[prefix + "weight"] = state_dict.pop(prefix + "scale")
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        def reset_parameters(self):
            self.bias = None
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.rms_norm(x, self.normalized_shape, (weight + 1.0) if self.add else weight, self.eps)
            elif self.add:
                return torch.nn.functional.rms_norm(x, self.normalized_shape, self.weight + 1.0, self.eps)
            else:
                return super().forward(x)

    class Embedding(torch.nn.Embedding, ForgeWeights):

        def __init__(self, *args, **kwargs):
            kwargs["device"] = current_device
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled
            self.bias = None

        def reset_parameters(self):
            self.bias = None
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x, skip_weight_dtype=True, skip_bias_dtype=True)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.embedding(x, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
            else:
                return super().forward(x)


# region GGUF


from backend.operations_gguf import dequantize_tensor


class ForgeOperationsGGUF(ForgeOperations):
    class Linear(torch.nn.Module, ForgeWeights):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.dummy = {"device": current_device, "dtype": current_dtype}
            self.weight = None
            self.bias = None

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            if hasattr(self, "dummy"):
                if (computation_dtype := self.dummy["dtype"]) not in [torch.float16, torch.bfloat16]:
                    computation_dtype = torch.float16

                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"].to(device=self.dummy["device"])
                    self.weight.computation_dtype = computation_dtype
                if prefix + "bias" in state_dict:
                    self.bias = state_dict[prefix + "bias"].to(device=self.dummy["device"])
                    self.bias.computation_dtype = computation_dtype

                del self.dummy
            else:
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"]
                if prefix + "bias" in state_dict:
                    self.bias = state_dict[prefix + "bias"]

        def _apply(self, fn, recurse=True):
            for k, p in self.named_parameters(recurse=False, remove_duplicate=True):
                setattr(self, k, utils.tensor2parameter(fn(p)))
            return self

        def forward(self, x):
            if self.bias is not None and self.bias.dtype != x.dtype:
                self.bias = utils.tensor2parameter(dequantize_tensor(self.bias).to(x.dtype))
            if self.weight is not None and self.weight.dtype != x.dtype and getattr(self.weight, "gguf_cls", None) is None:
                self.weight = utils.tensor2parameter(self.weight.to(x.dtype))

            weight, bias, signal = weights_manual_cast(self, x, weight_fn=dequantize_tensor, skip_bias_dtype=True)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.linear(x, weight, bias)

    class Conv2d(torch.nn.Conv2d, ForgeWeights):
        def __init__(self, *args, **kwargs):
            kwargs["device"] = current_device
            kwargs["dtype"] = current_dtype
            super().__init__(*args, **kwargs)
            self.dummy = {"device": current_device, "dtype": current_dtype}
            self.weight = None
            self.bias = None

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            if hasattr(self, "dummy"):
                if (computation_dtype := self.dummy["dtype"]) not in [torch.float16, torch.bfloat16]:
                    computation_dtype = torch.float16

                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"].to(device=self.dummy["device"])
                    self.weight.computation_dtype = computation_dtype
                if prefix + "bias" in state_dict:
                    self.bias = state_dict[prefix + "bias"].to(device=self.dummy["device"])
                    self.bias.computation_dtype = computation_dtype

                del self.dummy
            else:
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"]
                if prefix + "bias" in state_dict:
                    self.bias = state_dict[prefix + "bias"]

        def _apply(self, fn, recurse=True):
            for k, p in self.named_parameters(recurse=False, remove_duplicate=True):
                setattr(self, k, utils.tensor2parameter(fn(p)))
            return self

        def forward(self, x):
            if self.bias is not None and self.bias.dtype != x.dtype:
                self.bias = utils.tensor2parameter(dequantize_tensor(self.bias).to(x.dtype))
            if self.weight is not None and self.weight.dtype != x.dtype and getattr(self.weight, "gguf_cls", None) is None:
                self.weight = utils.tensor2parameter(self.weight.to(x.dtype))

            weight, bias, signal = weights_manual_cast(self, x, weight_fn=dequantize_tensor, skip_bias_dtype=True)
            with main_stream_worker(weight, bias, signal):
                return super()._conv_forward(x, weight, bias)

    class Embedding(torch.nn.Embedding, ForgeWeights):
        def __init__(self, *args, **kwargs):
            kwargs["device"] = current_device
            kwargs["dtype"] = current_dtype
            super().__init__(*args, **kwargs)
            self.dummy = {"device": current_device, "dtype": current_dtype}
            self.weight = None
            self.bias = None

            self._dtype = current_dtype

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            if hasattr(self, "dummy"):
                if (computation_dtype := self.dummy["dtype"]) not in [torch.float16, torch.bfloat16]:
                    computation_dtype = torch.float16

                if prefix + "weight" in state_dict:
                    _weight = state_dict[prefix + "weight"].to(device=self.dummy["device"])
                    if not isinstance(_weight, torch.nn.Parameter):
                        _weight = torch.nn.Parameter(_weight, requires_grad=False)
                    self.weight = _weight
                    self.weight.computation_dtype = computation_dtype

                del self.dummy
            else:
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"]

        def _apply(self, fn, recurse=True):
            for k, p in self.named_parameters(recurse=False, remove_duplicate=True):
                setattr(self, k, utils.tensor2parameter(fn(p)))
            return self

        def reset_parameters(self):
            self.bias = None
            return None

        def forward(self, x):
            weight, bias, signal = weights_manual_cast(self, x, weight_fn=dequantize_tensor, skip_weight_dtype=True, skip_bias_dtype=True)
            with main_stream_worker(weight, bias, signal):
                o = torch.nn.functional.embedding(x, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
                return o.to(dtype=self._dtype)


# region fp8


from backend.operations_mixed_precision import (
    QuantizedTensor,
    TensorCoreFP8Layout,
    mixed_precision_ops,
)


def fp8_linear(self: torch.nn.Linear, input: torch.Tensor):
    # https://github.com/Comfy-Org/ComfyUI/blob/v0.16.4/comfy/ops.py#L615
    dtype = self.weight.dtype
    if dtype is not torch.float8_e4m3fn:
        return None

    input_dtype = input.dtype
    input_shape = input.shape
    tensor_3d = input.ndim == 3

    if tensor_3d:
        input = input.reshape(-1, input_shape[2])

    if input.ndim != 2:
        return None

    scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
    scale_input = torch.ones((), device=input.device, dtype=torch.float32)

    w, bias, signal = weights_manual_cast(self, input, dtype=dtype)

    with main_stream_worker(w, bias, signal):
        input = torch.clamp(input, min=-448, max=448, out=input)
        input_fp8 = input.to(dtype).contiguous()
        layout_params_input = TensorCoreFP8Layout.Params(scale=scale_input, orig_dtype=input_dtype, orig_shape=tuple(input_fp8.shape))
        quantized_input = QuantizedTensor(input_fp8, "TensorCoreFP8Layout", layout_params_input)

        layout_params_weight = TensorCoreFP8Layout.Params(scale=scale_weight, orig_dtype=input_dtype, orig_shape=tuple(w.shape))
        quantized_weight = QuantizedTensor(w, "TensorCoreFP8Layout", layout_params_weight)
        o = torch.nn.functional.linear(quantized_input, quantized_weight, bias)

    if tensor_3d:
        o = o.reshape((input_shape[0], input_shape[1], w.shape[0]))

    return o


class ForgeOperationsFP8(ForgeOperations):
    class Linear(ForgeOperations.Linear, ForgeWeights):
        def forward(self, x):
            try:
                if (out := fp8_linear(self, x)) is not None:
                    return out
            except Exception as e:
                memory_management.logger.error(f"Error during fp8_fast: {e}")

            return super().forward(x)


# region Tiled


class TiledOperations(ForgeOperations):
    class Conv2d(ForgeOperations.Conv2d):
        tile_size: int

        def __init__(self, *arg, **kwargs):
            super().__init__(*arg, **kwargs)
            self._3x1x1: bool = self.kernel_size == (3, 3) and self.stride == (1, 1) and self.padding == (1, 1)
            self.tile_size = args.tiled_conv2d

        @torch.inference_mode()
        def forward(self, x: torch.Tensor):
            if not self._3x1x1:
                return super().forward(x)

            B, C, H, W = x.shape

            if H <= self.tile_size and W <= self.tile_size:
                return super().forward(x)

            orig_forward = super().forward
            out_channels = self.out_channels if self.out_channels is not None else C

            out = torch.empty((B, out_channels, H, W), device=x.device, dtype=x.dtype, memory_format=torch.contiguous_format)
            non_blocking = memory_management.device_supports_non_blocking(x.device)

            for i in range(0, H, self.tile_size):
                i0 = max(i - 1, 0)
                i1 = min(i + self.tile_size + 1, H)
                pi = i - i0
                ph = min(self.tile_size, H - i)

                for j in range(0, W, self.tile_size):
                    j0 = max(j - 1, 0)
                    j1 = min(j + self.tile_size + 1, W)

                    tile = x[:, :, i0:i1, j0:j1]
                    tile_conv = orig_forward(tile)

                    pj = j - j0
                    pw = min(self.tile_size, W - j)

                    out[:, :, i : i + ph, j : j + pw].copy_(tile_conv[:, :, pi : pi + ph, pj : pj + pw], non_blocking=non_blocking)
                    del tile_conv

            return out


# region Pick OPs


@contextlib.contextmanager
def using_forge_operations(operations=None, device=None, dtype=None, manual_cast_enabled=False, extra_dtype=None):
    global current_device, current_dtype, current_manual_cast_enabled

    current_device, current_dtype, current_manual_cast_enabled = device, dtype, manual_cast_enabled

    if isinstance(extra_dtype, dict):
        # https://github.com/Comfy-Org/ComfyUI/blob/v0.16.4/comfy/ops.py#L950

        _device = memory_management.get_torch_device()
        _dtype = torch.bfloat16 if memory_management.should_use_bf16(_device) else torch.float32
        fp8_compute = memory_management.supports_fp8_compute(_device)
        nvfp4_compute = memory_management.supports_nvfp4_compute(_device)
        mxfp8_compute = memory_management.supports_mxfp8_compute(_device)

        disabled = set()
        if not nvfp4_compute:
            disabled.add("nvfp4")
        if not mxfp8_compute:
            disabled.add("mxfp8")
        if not fp8_compute:
            disabled.add("float8_e4m3fn")
            disabled.add("float8_e5m2")

        _full: bool = extra_dtype.pop("TE", False)  # https://github.com/Comfy-Org/ComfyUI/blob/v0.16.4/comfy/sd1_clip.py#L114
        operations = mixed_precision_ops(quant_config=extra_dtype, compute_dtype=_dtype, full_precision_mm=_full, disabled=disabled)

    if operations is None:
        if extra_dtype in ["gguf"]:
            operations = ForgeOperationsGGUF
        elif extra_dtype in ["vae"] and args.tiled_conv2d:
            memory_management.logger.info(f"Using TiledOperations ({args.tiled_conv2d}) for VAE")
            operations = TiledOperations
        elif dtype is torch.float8_e4m3fn and args.fast_fp8 and memory_management.supports_fp8_compute(memory_management.get_torch_device()):
            operations = ForgeOperationsFP8
        else:
            operations = ForgeOperations

    if dynamic_args.ops is None:
        dynamic_args.ops = str(operations.__name__)

    op_names = ("Linear", "Conv1d", "Conv2d", "Conv3d", "GroupNorm", "LayerNorm", "RMSNorm", "Embedding")
    backups = {op_name: getattr(torch.nn, op_name) for op_name in op_names}

    try:
        for op_name in op_names:
            setattr(torch.nn, op_name, getattr(operations, op_name))

        yield

    finally:
        for op_name in op_names:
            setattr(torch.nn, op_name, backups[op_name])


from functools import wraps


@contextlib.contextmanager
def automatic_memory_management():
    memory_management.free_memory(memory_required=3 * 1024 * 1024 * 1024, device=memory_management.get_torch_device())

    module_list: list[torch.nn.Module] = []

    original_init = torch.nn.Module.__init__
    original_to = torch.nn.Module.to

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        module_list.append(self)
        return original_init(self, *args, **kwargs)

    @wraps(original_to)
    def patched_to(self, *args, **kwargs):
        module_list.append(self)
        return original_to(self, *args, **kwargs)

    try:
        torch.nn.Module.__init__ = patched_init
        torch.nn.Module.to = patched_to
        yield
    finally:
        torch.nn.Module.__init__ = original_init
        torch.nn.Module.to = original_to

    start = time.perf_counter()
    module_list = set(module_list)

    for module in module_list:
        module.cpu()

    memory_management.soft_empty_cache()
    end = time.perf_counter()

    memory_management.logger.debug(f"Automatic Memory Management: {len(module_list)} Modules in {(end - start):.2f} seconds")
