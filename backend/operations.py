# Copyright (C) 2024 Forge - Establish the Structures
# Copyright (C) 2025 ComfyUI - where Optimization is Stolen
# Copyright (C) 2026 Haoming02 - Burnt the Kitchen

import contextlib
import time
from typing import Callable

import torch

from backend import memory_management, stream, utils
from backend.args import args
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


def weights_manual_cast(layer: torch.nn.Module, x: torch.Tensor, skip_weight_dtype: bool = False, skip_bias_dtype: bool = False, weight_fn: Callable = None, bias_fn: Callable = None, *, dtype: torch.dtype = None, _cast: bool = True):
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
            self.dummy = torch.nn.Parameter(torch.empty([], device=current_device, dtype=current_dtype))
            self.weight = torch.empty([], device=current_device, dtype=current_dtype)  # SVDQW4A4Linear.from_linear
            self.scale_weight = None
            self.bias = None
            self.scale_weight = None
            self.scale_input = None
            self.parameters_manual_cast = current_manual_cast_enabled

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if hasattr(self, "dummy"):
                if prefix + "weight" in state_dict:
                    del self.weight
                    self.weight = torch.nn.Parameter(state_dict[prefix + "weight"].to(self.dummy))
                if prefix + "bias" in state_dict:
                    self.bias = torch.nn.Parameter(state_dict[prefix + "bias"].to(self.dummy))
                if prefix + "scale_weight" in state_dict:
                    self.scale_weight = torch.nn.Parameter(state_dict[prefix + "scale_weight"])
                elif prefix + "weight_scale" in state_dict:
                    self.scale_weight = torch.nn.Parameter(state_dict[prefix + "weight_scale"])
                if prefix + "scale_input" in state_dict:
                    self.scale_input = torch.nn.Parameter(state_dict[prefix + "scale_input"])
                elif prefix + "input_scale" in state_dict:
                    self.scale_input = torch.nn.Parameter(state_dict[prefix + "input_scale"])
                del self.dummy
            else:
                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

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

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return self._conv_forward(x, weight, bias)
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
            self.dummy = torch.nn.Parameter(torch.empty([], device=current_device, dtype=current_dtype))
            self.weight = None
            self.bias = None

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            if hasattr(self, "dummy"):
                computation_dtype = self.dummy.dtype
                if computation_dtype not in [torch.float16, torch.bfloat16]:
                    computation_dtype = torch.float16
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"].to(device=self.dummy.device)
                    self.weight.computation_dtype = computation_dtype
                if prefix + "bias" in state_dict:
                    self.bias = state_dict[prefix + "bias"].to(device=self.dummy.device)
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
            self.dummy = torch.nn.Parameter(torch.empty([], device=current_device, dtype=current_dtype))
            self.bias = None

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            if hasattr(self, "dummy"):
                computation_dtype = self.dummy.dtype
                if computation_dtype not in [torch.float16, torch.bfloat16]:
                    computation_dtype = torch.float16
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"].to(device=self.dummy.device)
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
        w, bias, s = weights_manual_cast(self, input, dtype=dtype)
        w = w.t()

        scale_weight: torch.Tensor = getattr(self, "scale_weight", None)

        if scale_weight is None:
            scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
        else:
            scale_weight = scale_weight.to(input.device)

        scale_input = torch.ones((), device=input.device, dtype=torch.float32)
        input = torch.clamp(input, min=-448, max=448, out=input)
        input = input.reshape(-1, input_shape[2]).to(dtype).contiguous()

        o = torch._scaled_mm(input, w, out_dtype=input_dtype, bias=bias, scale_a=scale_input, scale_b=scale_weight)

        if isinstance(o, tuple):
            o = o[0]

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

            weight, bias = get_weight_and_bias(self)
            return torch.nn.functional.linear(x, weight, bias)


# region Pick OPs


@contextlib.contextmanager
def using_forge_operations(operations=None, device=None, dtype=None, manual_cast_enabled=False, bnb_dtype=None):
    global current_device, current_dtype, current_manual_cast_enabled, current_bnb_dtype

    current_device, current_dtype, current_manual_cast_enabled, current_bnb_dtype = device, dtype, manual_cast_enabled, bnb_dtype

    if operations is False:
        operations = ForgeOperationsNunchaku
    elif operations is None:
        if bnb_dtype in [torch.float8_e4m3fn] and args.fast_fp8 and memory_management.supports_fp8_compute(memory_management.get_torch_device()):
            operations = fp8Operations
        elif bnb_dtype in ["gguf"]:
            operations = ForgeOperationsGGUF
        elif bnb_dtype in ["nf4", "fp4"]:
            assert memory_management.bnb_enabled()
            operations = ForgeOperationsBNB4bits
        else:
            operations = ForgeOperations

    op_names = ["Linear", "Conv1d", "Conv2d", "Conv3d", "GroupNorm", "LayerNorm", "RMSNorm", "Embedding"]
    backups = {op_name: getattr(torch.nn, op_name) for op_name in op_names}

    try:
        for op_name in op_names:
            setattr(torch.nn, op_name, getattr(operations, op_name))

        yield

    finally:
        for op_name in op_names:
            setattr(torch.nn, op_name, backups[op_name])
    return


def shift_manual_cast(model, enabled):
    for m in model.modules():
        if hasattr(m, "parameters_manual_cast"):
            m.parameters_manual_cast = enabled
    return


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
