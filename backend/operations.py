# Copyright (C) 2024 Forge - Establish the Structures
# Copyright (C) 2025 ComfyUI - where Optimization is Stolen
# Copyright (C) 2026 Haoming02 - Burnt the Kitchen

import contextlib
import time
from typing import Callable, Union

import torch

from backend import memory_management, stream, utils
from backend.args import args, dynamic_args
from backend.patcher.lora import merge_lora_to_weight


def scaled_dot_product_attention(q, k, v, *args, **kwargs):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)


try:
    if torch.cuda.is_available() and memory_management.WINDOWS:
        import inspect

        from torch.nn.attention import SDPBackend, sdpa_kernel

        if "set_priority" in inspect.signature(sdpa_kernel).parameters:
            SDPA_BACKEND_PRIORITY = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]

            SDPA_BACKEND_PRIORITY.insert(0, SDPBackend.CUDNN_ATTENTION)

            def scaled_dot_product_attention(q, k, v, *args, **kwargs):
                with sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True):
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)

except Exception:
    pass


# region Cast


def get_weight_and_bias(layer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Forge-Specific Function for on-the-fly LoRA"""
    loras: dict[str, list] = getattr(layer, "forge_online_loras", dict())

    weight: torch.Tensor = getattr(layer, "weight", None)
    weight_patches: list = loras.get("weight", None)
    if weight is not None and weight_patches is not None:
        weight = merge_lora_to_weight(patches=weight_patches, weight=weight, key="online_weight_lora", computation_dtype=weight.dtype)

    bias: torch.Tensor = getattr(layer, "bias", None)
    bias_patches: list = loras.get("bias", None)
    if bias is not None and bias_patches is not None:
        bias = merge_lora_to_weight(patches=bias_patches, weight=bias, key="online_bias_lora", computation_dtype=bias.dtype)

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

    loras: dict[str, list[torch.Tensor]] = getattr(layer, "forge_online_loras", dict())

    weight_patches = loras.get("weight", None)
    bias_patches = loras.get("bias", None)

    if weight is not None and weight_patches is not None:
        weight = merge_lora_to_weight(patches=weight_patches, weight=weight, key="online_weight_lora", computation_dtype=weight.dtype)

    if bias is not None and bias_patches is not None:
        bias = merge_lora_to_weight(patches=bias_patches, weight=bias, key="online_bias_lora", computation_dtype=bias.dtype)

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
current_bnb_dtype: str = None


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


# region Int8


from backend.operations_int8 import (
    CONVROT_GROUP_SIZE,
    dequantize,
    int8_forward_dynamic,
    int8_forward_dynamic_per_row,
    quantize_int8,
    quantize_int8_axiswise,
)
from backend.patcher.lora import merge_lora_to_weight
from backend.quant_rotation import build_hadamard, rotate_activation, rotate_weight


class ForgeOperationsInt8(ForgeOperations):
    """Custom operations for INT8 tensorwise quantization"""

    excluded_names = []
    dynamic_quantize = True  # Toggle for on-the-fly quantization
    enable_convrot = True  # Toggle for ConvRot Hadamard rotation

    _is_prequantized = False  # status flag (not used for detection)

    applied_lora_patches = set()
    lora_patches = {}  # Map of model_key -> patch list (from load_lora)
    lora_strength = 1.0

    class Linear(torch.nn.Linear, ForgeWeights):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.register_buffer("weight_scale", None)
            self._is_quantized = False
            self._is_per_row = False  # Track quantization granularity
            self._use_convrot = False  # Track if ConvRot was applied
            self._weight_scale_scalar = None  # For scalar (non-tensor) scales
            self.compute_dtype = torch.bfloat16
            self.lora_patches = []  # List of (down_scaled, up, start, size) set by INT8ModelPatcher

            self.parameters_manual_cast = current_dtype != self.compute_dtype

        def reset_parameters(self):
            return None

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            weight_key = prefix + "weight"

            # Utility to normalize keys by stripping common prefixes
            def normalize_key(key):
                if not isinstance(key, str):
                    return key
                for p in ["diffusion_model.", "model.diffusion_model.", "model.", "transformer."]:
                    if key.startswith(p):
                        return key[len(p) :]
                return key

            def apply_lora_patches(tensor, key):
                if not ForgeOperationsInt8.lora_patches or tensor.dtype == torch.int8:
                    return tensor
                nk = normalize_key(key)
                patches = ForgeOperationsInt8.lora_patches.get(nk)
                if patches:
                    # calculate_weight expects: [(strength, v, strength_model, offset, function)]
                    formatted = []
                    for patch in patches:
                        if len(patch) == 4:
                            v, offset, function, strength = patch
                        else:
                            v, offset, function = patch
                            strength = getattr(ForgeOperationsInt8, "lora_strength", 1.0)
                        formatted.append((strength, v, 1.0, offset, function))

                    # Track applied patches
                    ForgeOperationsInt8.applied_lora_patches.add(nk)

                    device = torch.device("cuda") if torch.cuda.is_available() else tensor.device
                    temp_dtype = memory_management.lora_compute_dtype(device)

                    tensor_temp = tensor.to(temp_dtype)
                    result_temp = merge_lora_to_weight(formatted, tensor_temp, key)
                    return result_temp.to(tensor.dtype)
                return tensor

            input_scale_key = prefix + "input_scale"
            bias_key = prefix + "bias"

            def pop_metadata(sd, p, k):
                v = sd.pop(p + k, None)
                if v is not None:
                    return v
                v = sd.pop("model." + p + k, None)
                if v is not None:
                    return v
                if p.startswith("model."):
                    v = sd.pop(p[6:] + k, None)
                    if v is not None:
                        return v
                if p.startswith("diffusion_model."):
                    v = sd.pop("diffusion_model." + p + k, None)
                    if v is not None:
                        return v
                return None

            weight_scale = pop_metadata(state_dict, prefix, "weight_scale")
            comfy_quant_tensor = pop_metadata(state_dict, prefix, "comfy_quant")

            weight_tensor = state_dict.pop(weight_key, None)
            bias_tensor = state_dict.pop(bias_key, None)

            # Pop input_scale to clean state_dict, but ignore it
            _ = state_dict.pop(input_scale_key, None)

            if comfy_quant_tensor is not None:
                try:
                    import json

                    quant_conf = json.loads(bytes(comfy_quant_tensor.tolist()).decode("utf-8"))
                    if quant_conf.get("convrot", False):
                        self._use_convrot = True
                        ForgeOperationsInt8.enable_convrot = True  # Propagate globally for LoRA
                        if "convrot_groupsize" in quant_conf:
                            self._convrot_groupsize = quant_conf["convrot_groupsize"]
                except Exception:
                    pass

            # Apply LoRA patches to weight and bias once
            if weight_tensor is not None:
                weight_tensor = apply_lora_patches(weight_tensor, weight_key)
            if bias_tensor is not None:
                bias_tensor = apply_lora_patches(bias_tensor, bias_key)

            if weight_tensor is not None:
                if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                    # Load Quantized
                    self._is_quantized = True
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                    ForgeOperationsInt8._is_prequantized = True  # Found a quantized layer

                    if isinstance(weight_scale, torch.Tensor):
                        if weight_scale.numel() == 1:
                            # Scalar scale — store as float for speed
                            self._weight_scale_scalar = weight_scale.float().item()
                            self.weight_scale = None
                            self._is_per_row = False
                        elif weight_scale.dim() == 2 and weight_scale.shape[1] == 1:
                            self.register_buffer("weight_scale", weight_scale.float())
                            self._weight_scale_scalar = None
                            self._is_per_row = True
                        else:
                            self.register_buffer("weight_scale", weight_scale.float())
                            self._weight_scale_scalar = None
                            self._is_per_row = False
                    else:
                        self._weight_scale_scalar = float(weight_scale)
                        self.weight_scale = None
                        self._is_per_row = False

                elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn):
                    # Load High-Precision
                    is_excluded = any(ex in prefix for ex in ForgeOperationsInt8.excluded_names)
                    is_dim1 = self.in_features == 1 or self.out_features == 1 or weight_tensor.ndim == 1

                    if is_excluded or is_dim1 or not ForgeOperationsInt8.dynamic_quantize:
                        self._is_quantized = False
                        self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                    else:
                        # Quantize on the fly
                        device = torch.device("cuda") if torch.cuda.is_available() else weight_tensor.device

                        # Cast to float32 before rotation and scale computation
                        w_gpu = weight_tensor.to(device, non_blocking=True).float()

                        self._use_convrot = False
                        if getattr(ForgeOperationsInt8, "enable_convrot", False) and self.in_features % CONVROT_GROUP_SIZE == 0:
                            try:
                                H = build_hadamard(CONVROT_GROUP_SIZE, device=w_gpu.device, dtype=w_gpu.dtype)
                                w_gpu = rotate_weight(w_gpu, H, group_size=CONVROT_GROUP_SIZE)
                                self._use_convrot = True
                            except ImportError as e:
                                memory_management.logger.warning(f"[INT8 Fast] ConvRot Error: {e}")

                        q_weight, q_scale = quantize_int8_axiswise(w_gpu, dim=1)

                        self.weight = torch.nn.Parameter(q_weight.cpu(), requires_grad=False)
                        self.register_buffer("weight_scale", q_scale.cpu())
                        self._weight_scale_scalar = None
                        self._is_quantized = True
                        self._is_per_row = True
                else:
                    self._is_quantized = False
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
            else:
                missing_keys.append(weight_key)

            # Assign bias if it exists (already patched if needed)
            if bias_tensor is not None:
                self.bias = torch.nn.Parameter(bias_tensor, requires_grad=False)
            else:
                self.bias = None

        def _get_weight_scale(self):
            """Get weight scale, preferring scalar if available."""
            if self._weight_scale_scalar is not None:
                return self._weight_scale_scalar
            return self.weight_scale

        def convert_weight(self, _weight, inplace=False):
            if not self._is_quantized:
                return _weight
            return self.weight

        def set_weight(self, out_weight, inplace_update=False, seed=0, return_weight=False, **kwargs):
            if not self._is_quantized:
                new_weight = out_weight.to(self.weight.dtype)
                if return_weight:
                    return new_weight

                if inplace_update:
                    self.weight.data.copy_(new_weight)
                else:
                    self.weight = torch.nn.Parameter(new_weight, requires_grad=False)
                return

            if out_weight.dtype == torch.int8:
                if return_weight:
                    return out_weight

                if inplace_update:
                    self.weight.data.copy_(out_weight)
                else:
                    self.weight = torch.nn.Parameter(out_weight, requires_grad=False)
                return

            # Re-quantize if fallback occurred
            new_weight = quantize_int8(out_weight, self._get_weight_scale())

            if return_weight:
                return new_weight

            if inplace_update:
                self.weight.data.copy_(new_weight)
            else:
                self.weight = torch.nn.Parameter(new_weight, requires_grad=False)

        def set_bias(self, out_bias, inplace_update=False, seed=0, return_weight=False, **kwargs):
            if out_bias is None:
                return None

            new_bias = out_bias
            if return_weight:
                return new_bias

            if inplace_update:
                if self.bias is not None:
                    self.bias.data.copy_(new_bias)
            else:
                self.bias = torch.nn.Parameter(new_bias, requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Fast forward using torch._int_mm for quantized weights."""

            # Check if ComfyUI needs to manage weight transfer (VBAR, offloading, LoRA patches, etc.)
            # This mirrors the base class check in disable_weight_init.Linear.forward()
            need_cast = self.parameters_manual_cast or len(self.weight_function) > 0 or len(self.bias_function) > 0

            if not self._is_quantized:
                if need_cast:
                    weight, bias, signal = weights_manual_cast(self, x)
                    with main_stream_worker(weight, bias, signal):
                        return torch.nn.functional.linear(x, weight, bias)
                else:
                    return torch.nn.functional.linear(x, self.weight, self.bias)

            # INT8 quantized path
            if need_cast:
                # VBAR / offload / lowvram path
                weight, bias, signal = weights_manual_cast(self, x=None, dtype=torch.int8, device=x.device, bias_dtype=x.dtype)
            else:
                # Fast path: weights already on GPU, no functions to apply
                weight = self.weight
                bias = self.bias

            w_scale = self._get_weight_scale()
            if isinstance(w_scale, torch.Tensor) and w_scale.device != x.device:
                w_scale = w_scale.to(x.device, non_blocking=True)

            compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

            x_shape = x.shape
            x_2d = x.reshape(-1, x_shape[-1])

            if getattr(self, "_use_convrot", False):
                group_size = getattr(self, "_convrot_groupsize", CONVROT_GROUP_SIZE)
                H = build_hadamard(group_size, device=x.device, dtype=x.dtype)
                x_2d = rotate_activation(x_2d, H, group_size=group_size)

            if x_2d.shape[0] > 16:
                if self._is_per_row:
                    y = int8_forward_dynamic_per_row(x_2d, weight, w_scale, bias, compute_dtype)
                else:
                    y = int8_forward_dynamic(x_2d, weight, w_scale, bias, compute_dtype)
            else:
                # Small batch fallback
                w_float = dequantize(weight, w_scale).to(x.dtype)
                bias_typed = bias.to(x.dtype) if bias is not None else None
                y = torch.nn.functional.linear(x_2d, w_float, bias_typed)

            # Dynamic LoRA Path — handles split QKV via per-patch offsets
            for lora_down, lora_up, lora_start, lora_size in self.lora_patches:
                lD = lora_down.to(x.device, non_blocking=True)
                lU = lora_up.to(x.device, non_blocking=True)
                lora_x = torch.nn.functional.linear(x_2d.to(lD.dtype), lD)
                lora_y = torch.nn.functional.linear(lora_x, lU)  # [batch, slice_size or full_out]
                if lora_start is not None:
                    y[:, lora_start : lora_start + lora_size] = y[:, lora_start : lora_start + lora_size] + lora_y.to(y.dtype)
                else:
                    y = y + lora_y.to(y.dtype)

            if need_cast:
                with main_stream_worker(weight, bias, signal):
                    pass

            return y.reshape(*x_shape[:-1], y.shape[-1])


# region BnB


if memory_management.bnb_enabled():

    from backend.operations_bnb import (
        ForgeLoader4Bit,
        functional_dequantize_4bit,
        functional_linear_4bits,
    )

    class ForgeOperationsBNB4bits(ForgeOperations):
        class Linear(ForgeLoader4Bit, ForgeWeights):
            def __init__(self, *args, **kwargs):
                super().__init__(device=current_device, dtype=current_dtype, quant_type=current_bnb_dtype)
                self.parameters_manual_cast = current_manual_cast_enabled

            def forward(self, x):
                if self.bias is not None and self.bias.dtype != x.dtype:
                    self.bias = utils.tensor2parameter(self.bias.to(x.dtype))

                if hasattr(self, "forge_online_loras"):
                    weight, bias, signal = weights_manual_cast(self, x, weight_fn=functional_dequantize_4bit, skip_bias_dtype=True)
                    with main_stream_worker(weight, bias, signal):
                        return torch.nn.functional.linear(x, weight, bias)

                if not self.parameters_manual_cast:
                    return functional_linear_4bits(x, self.weight, self.bias)
                elif not self.weight.bnb_quantized:
                    assert x.device.type == "cuda", "BnB must use CUDA as Computation Device"
                    layer_original_device = self.weight.device
                    self.weight = self.weight._quantize(x.device)
                    bias = self.bias.to(x.device) if self.bias is not None else None
                    out = functional_linear_4bits(x, self.weight, bias)
                    self.weight = self.weight.to(layer_original_device)
                    return out
                else:
                    weight, bias, signal = weights_manual_cast(self, x, skip_weight_dtype=True, skip_bias_dtype=True)
                    with main_stream_worker(weight, bias, signal):
                        return functional_linear_4bits(x, weight, bias)


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

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            if hasattr(self, "dummy"):
                if (computation_dtype := self.dummy["dtype"]) not in [torch.float16, torch.bfloat16]:
                    computation_dtype = torch.float16

                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"].to(device=self.dummy["device"])
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
                return torch.nn.functional.embedding(x, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)


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
def using_forge_operations(operations=None, device=None, dtype=None, manual_cast_enabled=False, bnb_dtype=None):
    global current_device, current_dtype, current_manual_cast_enabled, current_bnb_dtype

    current_device, current_dtype, current_manual_cast_enabled, current_bnb_dtype = device, dtype, manual_cast_enabled, bnb_dtype

    if isinstance(bnb_dtype, str):
        # https://github.com/BobJohnson24/ComfyUI-Flux2-INT8/blob/main/int8_unet_loader.py
        ForgeOperationsInt8._is_prequantized = False

        match bnb_dtype:
            case "Flux2K4B" | "Flux2K9B":
                ForgeOperationsInt8.excluded_names = ["img_in", "time_in", "guidance_in", "txt_in", "double_stream_modulation_img", "double_stream_modulation_txt", "single_stream_modulation"]
                operations = ForgeOperationsInt8
            case "ZImage":
                ForgeOperationsInt8.excluded_names = ["cap_embedder", "t_embedder", "x_embedder", "cap_pad_token", "context_refiner", "final_layer", "noise_refiner", "adaLN", "x_pad_token", "layers.0."]
                operations = ForgeOperationsInt8
            case "Chroma":
                ForgeOperationsInt8.excluded_names = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in", "nerf_image_embedder", "nerf_blocks", "nerf_final_layer_conv", "__x0__", "nerf_final_layer_conv"]
                operations = ForgeOperationsInt8
            case "QwenImage":
                ForgeOperationsInt8.excluded_names = ["time_text_embed", "img_in", "norm_out", "proj_out", "txt_in"]
                operations = ForgeOperationsInt8
            case "ErnieImage":
                ForgeOperationsInt8.excluded_names = ["time", "x_embedder", "text_proj", "adaLN"]
                operations = ForgeOperationsInt8
            case "Anima":
                ForgeOperationsInt8.excluded_names = ["embed", "adaln"]
                operations = ForgeOperationsInt8
            case "WAN21_T2V" | "WAN21_I2V":
                ForgeOperationsInt8.excluded_names = ["patch_embedding", "text_embedding", "time_embedding", "time_projection", "head", "img_emb"]
                operations = ForgeOperationsInt8
    elif isinstance(bnb_dtype, dict):
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

        _full: bool = bnb_dtype.pop("TE", False)  # https://github.com/Comfy-Org/ComfyUI/blob/v0.16.4/comfy/sd1_clip.py#L114
        operations = mixed_precision_ops(quant_config=bnb_dtype, compute_dtype=_dtype, full_precision_mm=_full, disabled=disabled)

    if operations is None:
        if bnb_dtype in ["gguf"]:
            operations = ForgeOperationsGGUF
        elif bnb_dtype in ["nf4", "fp4"]:
            assert memory_management.bnb_enabled(), 'Install the "bitsandbytes" package with --bnb'
            operations = ForgeOperationsBNB4bits
        elif bnb_dtype in ["vae"] and args.tiled_conv2d:
            memory_management.logger.info(f"Using TiledOperations ({args.tiled_conv2d}) for VAE")
            operations = TiledOperations
        elif dtype is torch.float8_e4m3fn and args.fast_fp8 and memory_management.supports_fp8_compute(memory_management.get_torch_device()):
            operations = ForgeOperationsFP8
        else:
            operations = ForgeOperations

    if operations is ForgeOperationsInt8:
        memory_management.logger.info("Quantizing to int8...")

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
