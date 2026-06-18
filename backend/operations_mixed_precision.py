# https://github.com/Comfy-Org/ComfyUI/blob/v0.16.4/comfy/ops.py#L713

import json

import torch

from backend.memory_management import cast_to_device, logger

from .operations import (
    ForgeOperations,
    ForgeWeights,
    main_stream_worker,
    weights_manual_cast,
)
from .quant_ops import (  # noqa
    QUANT_ALGOS,
    QuantizedTensor,
    TensorCoreFP8Layout,
    get_layout_class,
)


def _quantized_weight_state_dict(module: torch.nn.Module, sd: dict[str, torch.Tensor], prefix: str, extra_quant_conf: dict = None, extra_quant_params: tuple[str] = ()):
    if not hasattr(module, "weight"):
        logger.warning(f"uninitialized op {prefix}")
        return sd
    bias = getattr(module, "bias", None)
    if bias is not None:
        sd[f"{prefix}bias"] = bias
    if module.weight is None:
        return sd

    if not isinstance(module.weight, QuantizedTensor):
        sd[f"{prefix}weight"] = module.weight
    else:
        sd.update(module.weight.state_dict(f"{prefix}weight"))
        quant_conf = {"format": module.quant_format}
        if getattr(module, "_full_precision_mm_config", False):
            quant_conf["full_precision_matrix_mult"] = True
        if extra_quant_conf:
            quant_conf.update(extra_quant_conf)
        sd[f"{prefix}comfy_quant"] = torch.tensor(list(json.dumps(quant_conf).encode("utf-8")), dtype=torch.uint8)
        for name in extra_quant_params:
            value = getattr(module, name, None)
            if value is not None:
                sd[f"{prefix}{name}"] = value

    return sd


def mixed_precision_ops(quant_config={}, compute_dtype=torch.bfloat16, full_precision_mm=False, disabled=[]):
    class MixedPrecisionOps(ForgeOperations):
        _quant_config = quant_config
        _compute_dtype = compute_dtype
        _full_precision_mm = full_precision_mm
        _disabled = disabled

        class Linear(torch.nn.Module, ForgeWeights):
            def __init__(
                self,
                in_features: int,
                out_features: int,
                bias: bool = True,
                device=None,
                dtype=None,
            ) -> None:
                super().__init__()

                self.factory_kwargs = {"device": device, "dtype": MixedPrecisionOps._compute_dtype}

                self.in_features = in_features
                self.out_features = out_features
                if bias:
                    self.bias = torch.nn.Parameter(torch.empty(out_features, **self.factory_kwargs))
                else:
                    self.register_parameter("bias", None)

                self.tensor_class = None
                self._full_precision_mm = MixedPrecisionOps._full_precision_mm
                self._full_precision_mm_config = False

            def reset_parameters(self):
                return None

            def _load_scale_param(self, state_dict, prefix, param_name, device, manually_loaded_keys, dtype=None):
                key = f"{prefix}{param_name}"
                value = state_dict.pop(key, None)
                if value is not None:
                    value = value.to(device=device)
                    if dtype is not None:
                        value = value.view(dtype=dtype)
                    manually_loaded_keys.append(key)
                return value

            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

                device = self.factory_kwargs["device"]
                layer_name = prefix.rstrip(".")
                weight_key = f"{prefix}weight"
                weight = state_dict.pop(weight_key, None)
                if weight is None:
                    logger.warning(f"Missing weight for layer {layer_name}")
                    return

                manually_loaded_keys = [weight_key]

                layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
                if layer_conf is not None:
                    layer_conf = json.loads(layer_conf.numpy().tobytes())

                if layer_conf is None:
                    self.weight = torch.nn.Parameter(weight.to(device=device, dtype=MixedPrecisionOps._compute_dtype), requires_grad=False)
                else:
                    self.quant_format = layer_conf.get("format", None)
                    self._full_precision_mm_config = layer_conf.get("full_precision_matrix_mult", False)
                    if not self._full_precision_mm:
                        self._full_precision_mm = self._full_precision_mm_config

                    if self.quant_format in MixedPrecisionOps._disabled:
                        self._full_precision_mm = True

                    if self.quant_format is None:
                        raise ValueError(f"Unknown quantization format for layer {layer_name}")

                    qconfig = QUANT_ALGOS[self.quant_format]
                    self.layout_type = qconfig["comfy_tensor_layout"]
                    layout_cls = get_layout_class(self.layout_type)

                    if self.quant_format in ["float8_e4m3fn", "float8_e5m2"]:
                        scale = self._load_scale_param(state_dict, prefix, "weight_scale", device, manually_loaded_keys)

                        params = layout_cls.Params(
                            scale=scale,
                            orig_dtype=MixedPrecisionOps._compute_dtype,
                            orig_shape=(self.out_features, self.in_features),
                        )
                    elif self.quant_format == "mxfp8":
                        block_scale = self._load_scale_param(state_dict, prefix, "weight_scale", device, manually_loaded_keys, dtype=torch.uint8)

                        if block_scale is None:
                            raise ValueError(f"Missing MXFP8 block scales for layer {layer_name}")

                        block_scale = block_scale.view(torch.float8_e8m0fnu)

                        params = layout_cls.Params(
                            scale=block_scale,
                            orig_dtype=MixedPrecisionOps._compute_dtype,
                            orig_shape=(self.out_features, self.in_features),
                        )
                    elif self.quant_format == "nvfp4":
                        tensor_scale = self._load_scale_param(state_dict, prefix, "weight_scale_2", device, manually_loaded_keys)
                        block_scale = self._load_scale_param(state_dict, prefix, "weight_scale", device, manually_loaded_keys, dtype=torch.float8_e4m3fn)

                        if tensor_scale is None or block_scale is None:
                            raise ValueError(f"Missing NVFP4 scales for layer {layer_name}")

                        params = layout_cls.Params(
                            scale=tensor_scale,
                            block_scale=block_scale,
                            orig_dtype=MixedPrecisionOps._compute_dtype,
                            orig_shape=(self.out_features, self.in_features),
                        )
                    else:
                        raise ValueError(f"Unsupported quantization format: {self.quant_format}")

                    self.weight = torch.nn.Parameter(QuantizedTensor(weight.to(device=device, dtype=qconfig["storage_t"]), self.layout_type, params), requires_grad=False)

                    for param_name in qconfig["parameters"]:
                        if param_name in {"weight_scale", "weight_scale_2"}:
                            continue

                        param_key = f"{prefix}{param_name}"
                        _v = state_dict.pop(param_key, None)
                        if _v is None:
                            continue
                        self.register_parameter(param_name, torch.nn.Parameter(_v.to(device=device), requires_grad=False))
                        manually_loaded_keys.append(param_key)

                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

                for key in manually_loaded_keys:
                    if key in missing_keys:
                        missing_keys.remove(key)

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                sd = destination if destination is not None else {}
                return _quantized_weight_state_dict(self, sd, prefix, extra_quant_params=("input_scale",))

            def forward(self, input, *args, **kwargs):
                input_shape = input.shape
                reshaped_3d = False
                compute_dtype = input.dtype

                if getattr(self, "layout_type", None) is not None and not isinstance(input, QuantizedTensor) and not self._full_precision_mm and not getattr(self, "forge_force_cast_weights", False) and len(self.weight_function) == 0 and len(self.bias_function) == 0:
                    input_reshaped = input.reshape(-1, input_shape[2]) if input.ndim == 3 else input

                    if input_reshaped.ndim == 2:
                        reshaped_3d = input.ndim == 3
                        scale = getattr(self, "input_scale", None)
                        if scale is not None:
                            scale = cast_to_device(scale, input.device, None)
                        input = QuantizedTensor.from_float(input_reshaped, self.layout_type, scale=scale)

                weight, bias, signal = weights_manual_cast(self, input, dtype=compute_dtype)
                with main_stream_worker(weight, bias, signal):
                    output = torch.nn.functional.linear(input, weight, bias)

                if reshaped_3d:
                    output = output.reshape((input_shape[0], input_shape[1], self.weight.shape[0]))

                return output

            @staticmethod
            def convert_weight(weight, **kwargs):
                if isinstance(weight, QuantizedTensor):
                    return weight.dequantize()
                else:
                    return weight

            def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
                if getattr(self, "layout_type", None) is not None:
                    weight = QuantizedTensor.from_float(weight, self.layout_type, scale="recalculate", stochastic_rounding=seed, inplace_ops=True).to(self.weight.dtype)
                else:
                    weight = weight.to(self.weight.dtype)
                if return_weight:
                    return weight

                assert inplace_update is False
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            def _apply(self, fn, recurse=True):
                if recurse:
                    for module in self.children():
                        module._apply(fn)

                for key, param in self._parameters.items():
                    if param is None:
                        continue
                    self.register_parameter(key, torch.nn.Parameter(fn(param), requires_grad=False))
                for key, buf in self._buffers.items():
                    if buf is not None:
                        self._buffers[key] = fn(buf)
                return self

        class Embedding(ForgeOperations.Embedding, ForgeWeights):
            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                weight_key = f"{prefix}weight"
                layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
                if layer_conf is not None:
                    layer_conf = json.loads(layer_conf.numpy().tobytes())

                quant_format = layer_conf.get("format") if layer_conf is not None else None
                manually_loaded_keys = []

                if quant_format in ("float8_e4m3fn", "float8_e5m2") and weight_key in state_dict:
                    self.quant_format = quant_format
                    qconfig = QUANT_ALGOS[quant_format]
                    self.layout_type = qconfig["comfy_tensor_layout"]
                    layout_cls = get_layout_class(self.layout_type)
                    weight = state_dict.pop(weight_key)
                    manually_loaded_keys.append(weight_key)

                    scale_key = f"{prefix}weight_scale"
                    scale = state_dict.pop(scale_key, None)
                    if scale is not None:
                        scale = scale.float()
                        manually_loaded_keys.append(scale_key)

                    params = layout_cls.Params(
                        scale=scale if scale is not None else torch.ones((), dtype=torch.float32),
                        orig_dtype=MixedPrecisionOps._compute_dtype,
                        orig_shape=(self.num_embeddings, self.embedding_dim),
                    )
                    self.weight = torch.nn.Parameter(QuantizedTensor(weight.to(dtype=qconfig["storage_t"]), qconfig["comfy_tensor_layout"], params), requires_grad=False)
                elif layer_conf is not None:
                    state_dict[f"{prefix}comfy_quant"] = torch.tensor(list(json.dumps(layer_conf).encode("utf-8")), dtype=torch.uint8)

                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

                for k in manually_loaded_keys:
                    if k in missing_keys:
                        missing_keys.remove(k)

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                sd = destination if destination is not None else {}
                return _quantized_weight_state_dict(self, sd, prefix)

            def forward(self, input):
                weight = self.weight

                if isinstance(weight, QuantizedTensor) and len(self.weight_function) == 0:
                    qdata, _, signal = weights_manual_cast(self, device=input.device, dtype=weight.dtype)
                    if isinstance(qdata, QuantizedTensor):
                        scale = qdata._params.scale
                        qdata = qdata._qdata
                    else:
                        scale = None

                    with main_stream_worker(qdata, None, signal):
                        x = torch.nn.functional.embedding(input, qdata, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

                    target_dtype = weight._params.orig_dtype
                    x = x.to(dtype=target_dtype)
                    if scale is not None and scale != 1.0:
                        x = x * scale.to(dtype=target_dtype)

                    return x

                return super().forward(input)

    return MixedPrecisionOps
