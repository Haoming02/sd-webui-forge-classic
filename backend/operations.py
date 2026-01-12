# Copyright Forge 2024

import contextlib
import time
from typing import Callable

import torch

from backend import memory_management, stream, utils
from backend.patcher.lora import merge_lora_to_weight


def get_weight_and_bias(layer: torch.nn.Module, weight_args: dict = None, bias_args: dict = None, weight_fn: Callable = None, bias_fn: Callable = None):
    scale_weight: torch.Tensor = getattr(layer, "scale_weight", None)
    patches: dict[str, list[torch.Tensor]] = getattr(layer, "forge_online_loras", None)
    weight_patches, bias_patches = None, None

    if patches is not None:
        weight_patches = patches.get("weight", None)

    if patches is not None:
        bias_patches = patches.get("bias", None)

    weight: torch.Tensor = None
    if layer.weight is not None:
        weight = layer.weight
        if weight_fn is not None:
            if weight_args is not None:
                fn_device = weight_args.get("device", None)
                if fn_device is not None:
                    weight = weight.to(device=fn_device)
            weight = weight_fn(weight)
        if weight_args is not None:
            weight = weight.to(**weight_args)
        if scale_weight is not None:
            weight = weight * scale_weight.to(device=weight.device, dtype=weight.dtype)
        if weight_patches is not None:
            weight = merge_lora_to_weight(patches=weight_patches, weight=weight, key="online weight lora", computation_dtype=weight.dtype)

    bias: torch.Tensor = None
    if layer.bias is not None:
        bias = layer.bias
        if bias_fn is not None:
            if bias_args is not None:
                fn_device = bias_args.get("device", None)
                if fn_device is not None:
                    bias = bias.to(device=fn_device)
            bias = bias_fn(bias)
        if bias_args is not None:
            bias = bias.to(**bias_args)
        if bias_patches is not None:
            bias = merge_lora_to_weight(patches=bias_patches, weight=bias, key="online bias lora", computation_dtype=bias.dtype)

    return weight, bias


def weights_manual_cast(layer: torch.nn.Module, x: torch.Tensor, skip_weight_dtype: bool = False, skip_bias_dtype: bool = False, weight_fn: Callable = None, bias_fn: Callable = None):
    weight, bias = None, None
    target_dtype, target_device = x.dtype, x.device
    weight_has_function: bool = weight_fn is not None
    bias_has_function: bool = bias_fn is not None

    non_blocking = memory_management.device_supports_non_blocking(target_device)

    weight_args = dict(device=target_device, dtype=target_dtype, non_blocking=non_blocking)
    if skip_weight_dtype or weight_has_function:
        weight_args.pop("dtype")

    bias_args = dict(device=target_device, dtype=target_dtype, non_blocking=non_blocking)
    if skip_bias_dtype:
        bias_args.pop("dtype")

    if (layer.weight is not None and target_device != layer.weight.device) or (layer.bias is not None and target_device != layer.bias.device):
        offload_stream = memory_management.get_offload_stream(target_device)
        _stream = stream.stream_context()(offload_stream)
    else:
        offload_stream = None
        _stream = None

    if layer.weight is not None:
        weight = memory_management.cast_to(layer.weight, **weight_args, copy=weight_has_function, stream=_stream)

    if layer.bias is not None:
        bias = memory_management.cast_to(layer.bias, **bias_args, copy=bias_has_function, stream=_stream)

    memory_management.sync_stream(target_device, offload_stream)

    weight_a = weight
    bias_a = bias

    if weight_has_function:
        weight = weight_fn(weight)
        if not skip_weight_dtype:
            weight = weight.to(dtype=target_dtype)
    if bias_has_function:
        bias = bias_fn(bias)

    scale_weight: torch.Tensor = getattr(layer, "scale_weight", None)
    if scale_weight is not None:
        weight = weight * scale_weight.to(weight)

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


# region: ForgeOperations


class ForgeOperations:
    class Linear(torch.nn.Module):
        def __init__(self, in_features, out_features, *args, **kwargs):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.dummy = torch.nn.Parameter(torch.empty(1, device=current_device, dtype=current_dtype))
            self.weight = torch.empty([0], device=current_device, dtype=current_dtype)  # SVDQW4A4Linear.from_linear
            self.scale_weight = None
            self.bias = None
            self.parameters_manual_cast = current_manual_cast_enabled

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if hasattr(self, "dummy"):
                if prefix + "weight" in state_dict:
                    del self.weight
                    self.weight = torch.nn.Parameter(state_dict[prefix + "weight"].to(self.dummy))
                if prefix + "scale_weight" in state_dict:
                    self.scale_weight = torch.nn.Parameter(state_dict[prefix + "scale_weight"])
                if prefix + "bias" in state_dict:
                    self.bias = torch.nn.Parameter(state_dict[prefix + "bias"].to(self.dummy))
                del self.dummy
            else:
                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.linear(x, weight, bias)
            else:
                weight, bias = get_weight_and_bias(self)
                return torch.nn.functional.linear(x, weight, bias)

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

    class ConvTranspose2d(torch.nn.ConvTranspose2d):

        def __init__(self, *args, **kwargs):
            kwargs["device"] = current_device
            kwargs["dtype"] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x, output_size=None):
            if self.parameters_manual_cast:
                num_spatial_dims = 2
                output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims, self.dilation)

                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.conv_transpose2d(x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
            else:
                weight, bias = get_weight_and_bias(self)
                num_spatial_dims = 2
                output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims, self.dilation)
                return torch.nn.functional.conv_transpose2d(x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

    class ConvTranspose1d(torch.nn.ConvTranspose1d):

        def __init__(self, *args, **kwargs):
            kwargs["device"] = current_device
            kwargs["dtype"] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x, output_size=None):
            if self.parameters_manual_cast:
                num_spatial_dims = 1
                output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims, self.dilation)

                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.conv_transpose1d(x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
            else:
                weight, bias = get_weight_and_bias(self)
                num_spatial_dims = 1
                output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims, self.dilation)
                return torch.nn.functional.conv_transpose2d(x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

    class ConvTranspose3d(torch.nn.ConvTranspose3d):

        def __init__(self, *args, **kwargs):
            kwargs["device"] = current_device
            kwargs["dtype"] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x, output_size=None):
            if self.parameters_manual_cast:
                num_spatial_dims = 3
                output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims, self.dilation)

                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.conv_transpose3d(x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
            else:
                weight, bias = get_weight_and_bias(self)
                num_spatial_dims = 3
                output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims, self.dilation)
                return torch.nn.functional.conv_transpose2d(x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

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
            self.add = add  # add is used by Gemma2 2B

        def reset_parameters(self):
            self.bias = None
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    if self.add:
                        return torch.nn.functional.rms_norm(x, self.normalized_shape, weight + 1.0, self.eps)
                    return torch.nn.functional.rms_norm(x, self.normalized_shape, weight, self.eps)
            else:
                if self.add:
                    return torch.nn.functional.rms_norm(x, self.normalized_shape, self.weight + 1.0, self.eps)
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


# region: BnB


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
                    # Maybe this can also be set to all non-bnb ops since the cost is very low.
                    # And it only invokes one time, and most linear does not have bias
                    self.bias = utils.tensor2parameter(self.bias.to(x.dtype))

                if hasattr(self, "forge_online_loras"):
                    weight, bias, signal = weights_manual_cast(self, x, weight_fn=functional_dequantize_4bit, bias_fn=None, skip_bias_dtype=True)
                    with main_stream_worker(weight, bias, signal):
                        return torch.nn.functional.linear(x, weight, bias)

                if not self.parameters_manual_cast:
                    return functional_linear_4bits(x, self.weight, self.bias)
                elif not self.weight.bnb_quantized:
                    assert x.device.type == "cuda", "BNB Must Use CUDA as Computation Device!"
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


# region: GGUF


from backend.operations_gguf import dequantize_tensor


class ForgeOperationsGGUF(ForgeOperations):
    class Linear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.empty(1, device=current_device, dtype=current_dtype))
            self.weight = None
            self.bias = None
            self.parameters_manual_cast = current_manual_cast_enabled

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if hasattr(self, "dummy"):
                computation_dtype = self.dummy.dtype
                if computation_dtype not in [torch.float16, torch.bfloat16]:
                    # GGUF cast only supports 16bits otherwise super slow
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
            return

        def _apply(self, fn, recurse=True):
            for k, p in self.named_parameters(recurse=False, remove_duplicate=True):
                setattr(self, k, utils.tensor2parameter(fn(p)))
            return self

        def forward(self, x):
            if self.bias is not None and self.bias.dtype != x.dtype:
                self.bias = utils.tensor2parameter(dequantize_tensor(self.bias).to(x.dtype))

            if self.weight is not None and self.weight.dtype != x.dtype and getattr(self.weight, "gguf_cls", None) is None:
                self.weight = utils.tensor2parameter(self.weight.to(x.dtype))

            weight, bias, signal = weights_manual_cast(self, x, weight_fn=dequantize_tensor, bias_fn=None, skip_bias_dtype=True)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.linear(x, weight, bias)

    class Embedding(torch.nn.Embedding):
        def __init__(self, *args, **kwargs):
            kwargs["device"] = current_device
            kwargs["dtype"] = current_dtype
            super().__init__(*args, **kwargs)
            self.dummy = torch.nn.Parameter(torch.empty(1, device=current_device, dtype=current_dtype))
            self.bias = None
            self.parameters_manual_cast = current_manual_cast_enabled

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if hasattr(self, "dummy"):
                computation_dtype = self.dummy.dtype
                if computation_dtype not in [torch.float16, torch.bfloat16]:
                    # GGUF cast only supports 16bits otherwise super slow
                    computation_dtype = torch.float16
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"].to(device=self.dummy.device)
                    self.weight.computation_dtype = computation_dtype
                del self.dummy
            else:
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"]
            return

        def _apply(self, fn, recurse=True):
            for k, p in self.named_parameters(recurse=False, remove_duplicate=True):
                setattr(self, k, utils.tensor2parameter(fn(p)))
            return self

        def reset_parameters(self):
            self.bias = None
            return None

        def forward(self, x):
            weight, bias, signal = weights_manual_cast(self, x, weight_fn=dequantize_tensor, bias_fn=None, skip_weight_dtype=True, skip_bias_dtype=True)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.embedding(x, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)


@contextlib.contextmanager
def using_forge_operations(operations=None, device=None, dtype=None, manual_cast_enabled=False, bnb_dtype=None):
    global current_device, current_dtype, current_manual_cast_enabled, current_bnb_dtype

    current_device, current_dtype, current_manual_cast_enabled, current_bnb_dtype = device, dtype, manual_cast_enabled, bnb_dtype

    if operations is False:
        _dev = torch.get_default_device()
        _dtype = torch.get_default_dtype()

        torch.set_default_device(current_device)
        torch.set_default_dtype(current_dtype)

        yield

        torch.set_default_device(_dev)
        torch.set_default_dtype(_dtype)

        return

    if operations is None:
        if bnb_dtype in ["gguf"]:
            operations = ForgeOperationsGGUF
        elif bnb_dtype in ["nf4", "fp4"]:
            assert memory_management.bnb_enabled()
            operations = ForgeOperationsBNB4bits
        else:
            operations = ForgeOperations

    op_names = ["Linear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "GroupNorm", "LayerNorm", "RMSNorm", "Embedding"]
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
