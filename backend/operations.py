# Copyright (C) 2024 Forge - Establish the Structures
# Copyright (C) 2025 ComfyUI - where Optimization is Stolen
# Copyright (C) 2026 Haoming02 - Burnt the Kitchen

import contextlib
import time
from typing import Callable

import torch

from backend import memory_management, stream, utils
from backend.args import args, dynamic_args
from backend.patcher.lora import merge_lora_to_weight


def _calc_mantissa(abs_x, exponent, normal_mask, mantissa_bits, exponent_bias, generator=None):
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - exponent_bias)) - 1.0) * (2**mantissa_bits),
        (abs_x / (2.0 ** (-exponent_bias + 1 - mantissa_bits))),
    )
    mantissa_scaled += torch.rand(
        mantissa_scaled.size(),
        dtype=mantissa_scaled.dtype,
        layout=mantissa_scaled.layout,
        device=mantissa_scaled.device,
        generator=generator,
    )
    return mantissa_scaled.floor() / (2**mantissa_bits)


def _stochastic_round_to_fp8(value: torch.Tensor, dtype: torch.dtype, seed: int) -> torch.Tensor:
    if dtype == torch.float8_e4m3fn:
        exponent_bits, mantissa_bits, exponent_bias = 4, 3, 7
    elif dtype == torch.float8_e5m2:
        exponent_bits, mantissa_bits, exponent_bias = 5, 2, 15
    else:
        return value.to(dtype=dtype)

    generator = torch.Generator(device=value.device)
    generator.manual_seed(int(seed))

    def _round_chunk(chunk: torch.Tensor) -> torch.Tensor:
        x = chunk.to(dtype=torch.float16)
        sign = torch.sign(x)
        abs_x = x.abs()
        sign = torch.where(abs_x == 0, 0, sign)

        exponent = torch.clamp(
            torch.floor(torch.log2(abs_x)) + exponent_bias,
            0,
            2**exponent_bits - 1,
        )
        normal_mask = ~(exponent == 0)
        abs_x[:] = _calc_mantissa(abs_x, exponent, normal_mask, mantissa_bits, exponent_bias, generator=generator)

        sign *= torch.where(
            normal_mask,
            (2.0 ** (exponent - exponent_bias)) * (1.0 + abs_x),
            (2.0 ** (-exponent_bias + 1)) * abs_x,
        )

        finfo = torch.finfo(dtype)
        sign.clamp_(min=finfo.min, max=finfo.max)
        return sign

    if value.ndim == 0:
        return _round_chunk(value).to(dtype=dtype)

    out = torch.empty_like(value, dtype=dtype)

    # Process in slices to cap peak memory on very large layers.
    rows = max(1, value.shape[0])
    num_slices = max(1, int(value.numel() / (4096 * 4096)))
    slice_size = max(1, round(rows / num_slices))

    for i in range(0, rows, slice_size):
        rounded = _round_chunk(value[i:i + slice_size])
        out[i:i + slice_size].copy_(rounded.to(dtype=dtype))

    return out


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
    scale_weight: torch.Tensor = getattr(layer, "scale_weight", None)
    loras: dict[str, list[torch.Tensor]] = getattr(layer, "forge_online_loras", dict())

    weight_patches = loras.get("weight", None)
    bias_patches = loras.get("bias", None)

    weight: torch.Tensor = getattr(layer, "weight", None)
    if weight is not None:
        if scale_weight is not None:
            weight = weight * scale_weight.to(device=weight.device, dtype=weight.dtype)
        if weight_patches is not None:
            weight = merge_lora_to_weight(patches=weight_patches, weight=weight, key="online weight lora", computation_dtype=weight.dtype)

    bias: torch.Tensor = getattr(layer, "bias", None)
    if bias is not None:
        if bias_patches is not None:
            bias = merge_lora_to_weight(patches=bias_patches, weight=bias, key="online bias lora", computation_dtype=bias.dtype)

    return weight, bias


def weights_manual_cast(layer: torch.nn.Module, x: torch.Tensor, skip_weight_dtype: bool = False, skip_bias_dtype: bool = False, weight_fn: Callable = None, bias_fn: Callable = None, *, dtype: torch.dtype = None, _cast: bool = True, _scale: bool = True):
    weight, bias = None, None
    target_dtype, target_device = x.dtype, x.device
    weight_has_function: bool = weight_fn is not None
    bias_has_function: bool = bias_fn is not None

    non_blocking = memory_management.device_supports_non_blocking(target_device)

    weight_args = dict(device=target_device, dtype=dtype or target_dtype, non_blocking=non_blocking)
    if skip_weight_dtype or weight_has_function:
        weight_args.pop("dtype")

    bias_args = dict(device=target_device, dtype=target_dtype, non_blocking=non_blocking)
    if skip_bias_dtype or bias_has_function:
        bias_args.pop("dtype")

    offload_stream, context = None, contextlib.nullcontext()

    if not _cast:
        # cast_to breaks BnB & GGUF
        if layer.weight is not None:
            weight = layer.weight.to(**weight_args, copy=True)
        if layer.bias is not None:
            bias = layer.bias.to(**bias_args, copy=True)

    else:
        if stream.should_use_stream():
            if (layer.weight is not None and target_device != layer.weight.device) or (layer.bias is not None and target_device != layer.bias.device):
                offload_stream = memory_management.get_offload_stream(target_device)
                context = stream.stream_context()(offload_stream)

        if layer.weight is not None:
            weight = memory_management.cast_to(layer.weight, **weight_args, copy=weight_has_function, context=context)

        if layer.bias is not None:
            bias = memory_management.cast_to(layer.bias, **bias_args, copy=bias_has_function, context=context)

    memory_management.sync_stream(target_device, offload_stream)

    if not _scale:
        return weight, bias, (offload_stream, weight, bias)

    weight_a = weight
    bias_a = bias

    if weight_has_function:
        weight = weight_fn(weight)
        if not skip_weight_dtype:
            weight = weight.to(dtype=target_dtype)

    if bias_has_function:
        bias = bias_fn(bias)
        if not skip_bias_dtype:
            bias = bias.to(dtype=target_dtype)

    scale_weight: torch.Tensor = getattr(layer, "scale_weight", None)
    if weight is not None and scale_weight is not None:
        weight = weight * scale_weight.to(weight)

    loras: dict[str, list[torch.Tensor]] = getattr(layer, "forge_online_loras", dict())

    weight_patches = loras.get("weight", None)
    bias_patches = loras.get("bias", None)

    if weight is not None and weight_patches is not None:
        weight = merge_lora_to_weight(patches=weight_patches, weight=weight, key="online weight lora", computation_dtype=weight.dtype)

    if bias is not None and bias_patches is not None:
        bias = merge_lora_to_weight(patches=bias_patches, weight=bias, key="online bias lora", computation_dtype=bias.dtype)

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


class ForgeOperations:
    class Linear(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, *args, **kwargs):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.dummy = {"device": current_device, "dtype": current_dtype}
            self.weight = None
            self.bias = None
            self.scale_weight = None
            self.scale_input = None
            self.parameters_manual_cast = current_manual_cast_enabled

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if hasattr(self, "dummy"):
                if prefix + "weight" in state_dict:
                    weight = state_dict[prefix + "weight"]
                    has_weight_scale = prefix + "scale_weight" in state_dict or prefix + "weight_scale" in state_dict

                    # Mixed fp8 checkpoints can contain a small set of unscaled BF16 weights.
                    # Preserve their source dtype to keep offline LoRA behavior consistent.
                    if self.dummy["dtype"] in memory_management.FLOAT8_TYPES and not has_weight_scale:
                        self.weight = torch.nn.Parameter(weight.to(device=self.dummy["device"], dtype=weight.dtype))
                    else:
                        self.weight = torch.nn.Parameter(weight.to(**self.dummy))
                if prefix + "bias" in state_dict:
                    self.bias = torch.nn.Parameter(state_dict[prefix + "bias"].to(**self.dummy))

                del self.dummy

                if prefix + "scale_weight" in state_dict:
                    self.scale_weight = torch.nn.Parameter(state_dict[prefix + "scale_weight"])
                elif prefix + "weight_scale" in state_dict:
                    self.scale_weight = torch.nn.Parameter(state_dict[prefix + "weight_scale"])
                if prefix + "scale_input" in state_dict:
                    self.scale_input = torch.nn.Parameter(state_dict[prefix + "scale_input"])
                elif prefix + "input_scale" in state_dict:
                    self.scale_input = torch.nn.Parameter(state_dict[prefix + "input_scale"])
            else:
                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        def reset_parameters(self):
            return None

        def convert_weight(self, _weight, inplace=False):
            if self.scale_weight is None:
                return _weight
            return _weight * self.scale_weight.to(device=_weight.device, dtype=_weight.dtype)

        def set_weight(self, out_weight, inplace_update=False, seed=0):
            target_dtype = self.weight.dtype

            # For fp8_scaled weights, re-quantize weight and recalculate scale_weight.
            if self.scale_weight is not None and target_dtype in memory_management.FLOAT8_TYPES:
                out_weight_f32 = out_weight.to(dtype=torch.float32)
                abs_max = out_weight_f32.abs().amax()

                max_fp8 = torch.finfo(target_dtype).max
                if torch.isfinite(abs_max) and abs_max > 0:
                    new_scale = (abs_max / max_fp8).to(dtype=torch.float32)
                else:
                    new_scale = torch.ones((), dtype=torch.float32, device=out_weight.device)

                new_scale = torch.clamp(new_scale, min=torch.finfo(torch.float32).tiny)
                normalized = out_weight_f32 / new_scale
                if seed is not None and int(seed) != 0:
                    try:
                        quantized = _stochastic_round_to_fp8(normalized, target_dtype, int(seed))
                    except torch.OutOfMemoryError:
                        memory_management.soft_empty_cache()
                        memory_management.logger.warning(
                            "OOM during fp8 stochastic rounding in set_weight; falling back to deterministic quantization."
                        )
                        quantized = normalized.clamp(min=-max_fp8, max=max_fp8).to(dtype=target_dtype)
                else:
                    quantized = normalized.clamp(min=-max_fp8, max=max_fp8).to(dtype=target_dtype)

                if inplace_update:
                    self.weight.data.copy_(quantized)
                else:
                    self.weight = torch.nn.Parameter(quantized, requires_grad=False)

                target_scale = new_scale.to(device=self.scale_weight.device, dtype=self.scale_weight.dtype)
                if self.scale_weight.shape != target_scale.shape:
                    target_scale = target_scale.expand_as(self.scale_weight)

                if inplace_update:
                    self.scale_weight.data.copy_(target_scale)
                else:
                    self.scale_weight = torch.nn.Parameter(target_scale, requires_grad=False)
                return

            out_weight = out_weight.to(dtype=target_dtype)
            if inplace_update:
                self.weight.data.copy_(out_weight)
            else:
                self.weight = torch.nn.Parameter(out_weight, requires_grad=False)

        def forward(self, x):
            # if self.scale_input is not None:  # TODO ?
            #     x = (x * self.scale_input.to(x)).contiguous()
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.linear(x, weight, bias)
            else:
                weight, bias = get_weight_and_bias(self)
                return torch.nn.functional.linear(x, weight, bias)

    class Conv1d(torch.nn.Conv1d):

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

    class Conv2d(torch.nn.Conv2d):

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

    class Conv3d(torch.nn.Conv3d):

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

    class GroupNorm(torch.nn.GroupNorm):

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

    class LayerNorm(torch.nn.LayerNorm):

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

    class RMSNorm(torch.nn.RMSNorm):

        def __init__(self, *args, add=False, **kwargs):
            kwargs["device"] = current_device
            kwargs["dtype"] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled
            self.bias = None
            self.add = add  # used by llama.py

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if prefix + "scale" in state_dict:  # Flux
                self.weight = torch.nn.Parameter(state_dict[prefix + "scale"].to(device=self.weight.device, dtype=self.weight.dtype))
            else:
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

    class Embedding(torch.nn.Embedding):

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
    dequantize,
    int8_forward_dynamic,
    quantize_int8_tensorwise,
    stochastic_round_int8_delta,
)


class ForgeOperationsInt8(ForgeOperations):
    """Custom operations for INT8 tensorwise quantization"""

    excluded_names = []
    _is_prequantized = None

    class Linear(torch.nn.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weight_scale = None
            self._is_quantized = False
            self.compute_dtype = torch.bfloat16
            self.lora_A = None
            self.lora_B = None
            self.lora_alpha = None

        def reset_parameters(self):
            return None

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            weight_key = prefix + "weight"
            scale_key = prefix + "weight_scale"
            input_scale_key = prefix + "input_scale"
            bias_key = prefix + "bias"

            weight_scale = state_dict.pop(scale_key, None)
            state_dict.pop(prefix + "comfy_quant", None)
            weight_tensor = state_dict.pop(weight_key, None)

            # Pop input_scale to clean state_dict, but ignore it
            _ = state_dict.pop(input_scale_key, None)

            if weight_tensor is not None:
                if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                    # Load Quantized
                    self._is_quantized = True
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                    ForgeOperationsInt8._is_prequantized = True  # Found a quantized layer

                    if isinstance(weight_scale, torch.Tensor):
                        self.weight_scale = weight_scale.float().item() if weight_scale.numel() == 1 else weight_scale.float()
                    else:
                        self.weight_scale = float(weight_scale)

                elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    # Load High-Precision
                    # Detect if the model is pre-quantized if we don't know yet
                    if ForgeOperationsInt8._is_prequantized is None:
                        # Robust detection: scan keys and a sample of values
                        is_prequant = False
                        for k in state_dict.keys():
                            if "weight_scale" in k or "comfy_quant" in k:
                                is_prequant = True
                                break

                        if not is_prequant:
                            # Fallback: scan a sample of values for int8 tensors
                            for i, v in enumerate(state_dict.values()):
                                if i > 200:
                                    break  # Safety limit
                                if getattr(v, "dtype", None) == torch.int8:
                                    is_prequant = True
                                    break
                        ForgeOperationsInt8._is_prequantized = is_prequant

                    is_excluded = any(ex in prefix for ex in ForgeOperationsInt8.excluded_names)
                    is_dim1 = self.in_features == 1 or self.out_features == 1 or weight_tensor.ndim == 1

                    if is_excluded or is_dim1 or ForgeOperationsInt8._is_prequantized:
                        self._is_quantized = False
                        self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                    else:
                        # Quantize on the fly
                        device = torch.device("cuda") if torch.cuda.is_available() else weight_tensor.device
                        w_gpu = weight_tensor.to(device, non_blocking=True)
                        q_weight, q_scale = quantize_int8_tensorwise(w_gpu)

                        self.weight = torch.nn.Parameter(q_weight.cpu(), requires_grad=False)
                        self.weight_scale = q_scale.cpu() if isinstance(q_scale, torch.Tensor) else q_scale
                        self._is_quantized = True
                else:
                    self._is_quantized = False
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
            else:
                missing_keys.append(weight_key)

            bias_tensor = state_dict.pop(bias_key, None)
            if bias_tensor is not None:
                self.bias = torch.nn.Parameter(bias_tensor, requires_grad=False)
            else:
                self.bias = None

        def convert_weight(self, _weight, inplace=False):
            if not self._is_quantized:
                return _weight
            return self.weight

        def set_weight(self, out_weight, inplace_update=False, seed=0):
            if not self._is_quantized:
                if inplace_update:
                    self.weight.data.copy_(out_weight)
                else:
                    self.weight = torch.nn.Parameter(out_weight.to(self.weight.dtype), requires_grad=False)
                return

            if out_weight.dtype == torch.int8:
                if inplace_update:
                    self.weight.data.copy_(out_weight)
                else:
                    self.weight = torch.nn.Parameter(out_weight, requires_grad=False)
                return

            # Re-quantize if fallback occurred
            new_weight = stochastic_round_int8_delta(out_weight, self.weight_scale, seed)
            if inplace_update:
                self.weight.data.copy_(new_weight)
            else:
                self.weight = torch.nn.Parameter(new_weight, requires_grad=False)

        def set_bias(self, out_bias, inplace_update=False, seed=0):
            if out_bias is None:
                return
            if inplace_update:
                if self.bias is not None:
                    self.bias.data.copy_(out_bias)
            else:
                self.bias = torch.nn.Parameter(out_bias, requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Fast forward using torch._int_mm for quantized weights."""

            if not self._is_quantized:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.linear(x, weight, bias)

            # 1. Move weight/bias/scale to device (non_blocking)
            weight = self.weight.to(x.device, non_blocking=True)
            bias = self.bias.to(x.device, non_blocking=True) if self.bias is not None else None

            w_scale = self.weight_scale
            if isinstance(w_scale, torch.Tensor):
                w_scale = w_scale.to(x.device, non_blocking=True)

            compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

            x_shape = x.shape
            x_2d = x.reshape(-1, x_shape[-1])

            if x_2d.shape[0] > 16:
                y = int8_forward_dynamic(x_2d, weight, w_scale, bias, compute_dtype)
            else:
                # Small batch fallback
                w_float = dequantize(weight, w_scale).to(x.dtype)
                y = torch.nn.functional.linear(x_2d, w_float, bias)

            # Dynamic LoRA Path
            if self.lora_A is not None and self.lora_B is not None:
                # Ensure LoRA tensors are on the same device as x
                lA = self.lora_A.to(x.device, non_blocking=True)
                lB = self.lora_B.to(x.device, non_blocking=True)

                lora_x = torch.nn.functional.linear(x_2d.to(lA.dtype), lA)
                lora_y = torch.nn.functional.linear(lora_x, lB)

                if self.lora_alpha is not None:
                    lora_y = lora_y * self.lora_alpha

                y = y + lora_y.to(y.dtype)

            return y.reshape(*x_shape[:-1], y.shape[-1])


# region BnB


if memory_management.bnb_enabled():

    from backend.operations_bnb import (
        ForgeLoader4Bit,
        functional_dequantize_4bit,
        functional_linear_4bits,
    )

    class ForgeOperationsBNB4bits(ForgeOperations):
        class Linear(ForgeLoader4Bit):
            def __init__(self, *args, **kwargs):
                super().__init__(device=current_device, dtype=current_dtype, quant_type=current_bnb_dtype)
                self.parameters_manual_cast = current_manual_cast_enabled

            def forward(self, x):
                if self.bias is not None and self.bias.dtype != x.dtype:
                    self.bias = utils.tensor2parameter(self.bias.to(x.dtype))

                if hasattr(self, "forge_online_loras"):
                    weight, bias, signal = weights_manual_cast(self, x, weight_fn=functional_dequantize_4bit, skip_bias_dtype=True, _cast=False)
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
                    weight, bias, signal = weights_manual_cast(self, x, skip_weight_dtype=True, skip_bias_dtype=True, _cast=False)
                    with main_stream_worker(weight, bias, signal):
                        return functional_linear_4bits(x, weight, bias)


# region GGUF


from backend.operations_gguf import dequantize_tensor


class ForgeOperationsGGUF(ForgeOperations):
    class Linear(torch.nn.Module):
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

            weight, bias, signal = weights_manual_cast(self, x, weight_fn=dequantize_tensor, skip_bias_dtype=True, _cast=False)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.linear(x, weight, bias)

    class Embedding(torch.nn.Embedding):
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
            weight, bias, signal = weights_manual_cast(self, x, weight_fn=dequantize_tensor, skip_weight_dtype=True, skip_bias_dtype=True, _cast=False)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.embedding(x, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

    class RMSNorm(torch.nn.RMSNorm):
        def __init__(self, *args, add=False, **kwargs):
            kwargs["device"] = current_device
            kwargs["dtype"] = current_dtype
            super().__init__(*args, **kwargs)
            self.dummy = {"device": current_device, "dtype": current_dtype}
            self.weight = None
            self.bias = None
            self.add = add  # used by llama.py

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            if hasattr(self, "dummy"):
                if (computation_dtype := self.dummy["dtype"]) not in [torch.float16, torch.bfloat16]:
                    computation_dtype = torch.float16

                if prefix + "scale" in state_dict:
                    self.weight = state_dict[prefix + "scale"].to(device=self.dummy["device"])
                elif prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"].to(device=self.dummy["device"])

                self.weight.computation_dtype = computation_dtype
                del self.dummy
            else:
                if prefix + "scale" in state_dict:
                    self.weight = state_dict[prefix + "scale"]
                elif prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"]

        def _apply(self, fn, recurse=True):
            for k, p in self.named_parameters(recurse=False, remove_duplicate=True):
                setattr(self, k, utils.tensor2parameter(fn(p)))
            return self

        def reset_parameters(self):
            self.bias = None
            return None

        def forward(self, x):
            weight, bias, signal = weights_manual_cast(self, x, weight_fn=dequantize_tensor, skip_weight_dtype=True, skip_bias_dtype=True, _cast=False)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.rms_norm(x, self.normalized_shape, (weight + 1.0) if self.add else weight, self.eps)


# region Nunchaku


class ForgeOperationsNunchaku(ForgeOperations):
    class Linear(torch.nn.Linear):
        def __init__(self, *args, **kwargs):
            kwargs["device"] = current_device
            kwargs["dtype"] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.linear(x, weight, bias)
            else:
                weight, bias = get_weight_and_bias(self)
                return torch.nn.functional.linear(x, weight, bias)


# region fp8


def _fp8_prepare_online_lora(linear: torch.nn.Linear, x: torch.Tensor):
    loras: dict[str, list[torch.Tensor]] = getattr(linear, "forge_online_loras", {})
    if not loras:
        return None, None, True

    weight_patches = loras.get("weight", None)
    bias_patches = loras.get("bias", None)

    # bias patches and non-LoRA adapters are not supported by the fast low-rank path
    if bias_patches:
        return None, None, False

    if not weight_patches:
        return None, None, True

    cache_key = (id(weight_patches), x.device, x.dtype)
    if getattr(linear, "_forge_fp8_lora_cache_key", None) == cache_key:
        return linear._forge_fp8_lora_A, linear._forge_fp8_lora_B, True

    all_A = []
    all_B = []

    for patch in weight_patches:
        if not isinstance(patch, (tuple, list)) or len(patch) < 5:
            return None, None, False

        strength, adapter, strength_model, offset, function = patch[:5]

        if strength == 0:
            continue
        if strength_model != 1.0 or offset is not None or function is not None:
            return None, None, False

        # Only classic LoRA can be composed into fast low-rank residual.
        if getattr(adapter, "name", None) != "lora" or not hasattr(adapter, "weights"):
            return None, None, False

        up, down, alpha, mid, dora_scale, reshape = adapter.weights
        if dora_scale is not None or reshape is not None:
            return None, None, False

        rank = down.shape[0] if down.ndim >= 2 else 1
        scale = strength * ((alpha / rank) if alpha is not None else 1.0)

        curr_A = down
        if mid is not None:
            curr_A = torch.mm(mid.flatten(start_dim=1), down.flatten(start_dim=1)).reshape(down.shape)

        all_A.append(curr_A.flatten(start_dim=1) * scale)
        all_B.append(up.flatten(start_dim=1))

    if not all_A:
        linear._forge_fp8_lora_cache_key = cache_key
        linear._forge_fp8_lora_A = None
        linear._forge_fp8_lora_B = None
        return None, None, True

    dtype = x.dtype if x.dtype in [torch.float16, torch.bfloat16, torch.float32] else torch.float16
    lora_A = torch.cat(all_A, dim=0).to(device=x.device, dtype=dtype, non_blocking=True)
    lora_B = torch.cat(all_B, dim=1).to(device=x.device, dtype=dtype, non_blocking=True)

    linear._forge_fp8_lora_cache_key = cache_key
    linear._forge_fp8_lora_A = lora_A
    linear._forge_fp8_lora_B = lora_B
    return lora_A, lora_B, True


def fp8_linear(self: torch.nn.Linear, input: torch.Tensor):
    dtype = self.weight.dtype
    if dtype != torch.float8_e4m3fn:
        return None

    tensor_2d = False
    if len(input.shape) == 2:
        tensor_2d = True
        input = input.unsqueeze(1)

    input_shape, input_dtype = input.shape, input.dtype

    if len(input.shape) == 3:
        lora_input = input.reshape(-1, input_shape[2])
        lora_A, lora_B, lora_supported = _fp8_prepare_online_lora(self, input)
        if not lora_supported:
            return None

        w, bias, signal = weights_manual_cast(self, input, dtype=dtype, _scale=False)
        w = w.t()

        if getattr(self, "scale_weight", None) is None:
            scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
        else:
            scale_weight = self.scale_weight.to(input.device)

        scale_input = torch.ones((), device=input.device, dtype=torch.float32)  # TODO ?

        input = torch.clamp(input, min=-448, max=448, out=input)
        input = input.reshape(-1, input_shape[2]).to(dtype).contiguous()

        with main_stream_worker(w, bias, signal):
            o = torch._scaled_mm(input, w, out_dtype=input_dtype, bias=bias, scale_a=scale_input, scale_b=scale_weight)

        if isinstance(o, tuple):
            o = o[0]

        if lora_A is not None and lora_B is not None:
            lora_x = torch.nn.functional.linear(lora_input.to(lora_A.dtype), lora_A)
            lora_y = torch.nn.functional.linear(lora_x, lora_B)
            o = o + lora_y.to(o.dtype)

        if tensor_2d:
            return o.reshape(input_shape[0], -1)

        return o.reshape((-1, input_shape[1], self.weight.shape[0]))

    return None


class fp8Operations(ForgeOperations):
    class Linear(ForgeOperations.Linear):
        def forward(self, x):
            try:
                if (out := fp8_linear(self, x)) is not None:
                    return out
            except Exception as e:
                memory_management.logger.error(f"Error during fp8_fast: {e}")

            # Fallback is needed for unsupported online adapter types.
            if getattr(self, "forge_online_loras", None):
                return ForgeOperations.Linear.forward(self, x)

            weight, bias = get_weight_and_bias(self)
            return torch.nn.functional.linear(x, weight, bias)


# region Pick OPs


@contextlib.contextmanager
def using_forge_operations(operations=None, device=None, dtype=None, manual_cast_enabled=False, bnb_dtype=None):
    global current_device, current_dtype, current_manual_cast_enabled, current_bnb_dtype

    current_device, current_dtype, current_manual_cast_enabled, current_bnb_dtype = device, dtype, manual_cast_enabled, bnb_dtype

    if operations is False:
        operations = ForgeOperationsNunchaku
    elif isinstance(bnb_dtype, str):
        # https://github.com/BobJohnson24/ComfyUI-Flux2-INT8/blob/main/int8_unet_loader.py
        ForgeOperationsInt8._is_prequantized = None

        match bnb_dtype:
            case "Flux2K4B" | "Flux2K9B":
                ForgeOperationsInt8.excluded_names = ["img_in", "time_in", "guidance_in", "txt_in", "final_layer", "double_stream_modulation_img", "double_stream_modulation_txt", "single_stream_modulation"]
                operations = ForgeOperationsInt8
            case "ZImage":
                ForgeOperationsInt8.excluded_names = ["cap_embedder", "t_embedder", "x_embedder", "cap_pad_token", "context_refiner", "final_layer", "noise_refiner", "adaLN", "x_pad_token"]
                operations = ForgeOperationsInt8
            case "Chroma":
                ForgeOperationsInt8.excluded_names = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in", "nerf_image_embedder", "nerf_blocks", "nerf_final_layer_conv", "__x0__", "nerf_final_layer_conv"]
                operations = ForgeOperationsInt8
            case "QwenImage":
                ForgeOperationsInt8.excluded_names = ["time_text_embed", "img_in", "norm_out", "proj_out", "txt_in"]
                operations = ForgeOperationsInt8
            case "WAN21_T2V" | "WAN21_I2V":
                ForgeOperationsInt8.excluded_names = ["patch_embedding", "text_embedding", "time_embedding", "time_projection" "head", "img_emb"]
                operations = ForgeOperationsInt8

    if operations is None:
        if bnb_dtype in [torch.float8_e4m3fn] and args.fast_fp8 and memory_management.supports_fp8_compute(memory_management.get_torch_device()):
            operations = fp8Operations
        elif bnb_dtype in ["gguf"]:
            operations = ForgeOperationsGGUF
        elif bnb_dtype in ["nf4", "fp4"]:
            assert memory_management.bnb_enabled()
            operations = ForgeOperationsBNB4bits
        else:
            operations = ForgeOperations

    if operations is ForgeOperationsInt8:
        memory_management.logger.info("Quantizing to int8...")

    if dynamic_args["ops"] is None:
        dynamic_args["ops"] = str(operations.__name__)

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
