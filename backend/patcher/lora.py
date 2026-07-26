# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.77/comfy/lora.py

import logging

import torch

from backend import memory_management, utils
from backend.logging import setup_logger
from modules_forge.packages.comfy.lora import (  # noqa
    load_lora,
    model_lora_keys_clip,
    model_lora_keys_unet,
    weight_adapter,
)

logger = logging.getLogger("lora")
setup_logger(logger)


def string_to_seed(data):
    crc = 0xFFFFFFFF
    for byte in data:
        if isinstance(byte, str):
            byte = ord(byte)
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF


@torch.inference_mode()
def merge_lora_to_weight(patches: list[tuple], weight: torch.Tensor, key: str, computation_dtype: torch.dtype = torch.float32):
    # https://github.com/comfyanonymous/ComfyUI/blob/v0.3.77/comfy/lora.py#L361

    for p in patches:
        strength = p[0]
        v = p[1]
        strength_model = p[2]
        offset = p[3]
        function = p[4]
        if function is None:
            function = lambda a: a

        old_weight = None
        if offset is not None:
            old_weight = weight
            weight = weight.narrow(offset[0], offset[1], offset[2])

        if strength_model != 1.0:
            weight *= strength_model

        if isinstance(v, list):
            v = (merge_lora_to_weight(v[1:], v[0][1](memory_management.cast_to_device(v[0][0], weight.device, computation_dtype, copy=True), inplace=True), key, computation_dtype=computation_dtype),)

        if isinstance(v, weight_adapter.WeightAdapterBase):
            output = v.calculate_weight(weight, key, strength, strength_model, offset, function, computation_dtype)
            if output is None:
                logger.error("Calculate Weight Failed: {} {}".format(v.name, key))
            else:
                weight = output
                if old_weight is not None:
                    weight = old_weight
            continue

        if len(v) == 1:
            patch_type = "diff"
        elif len(v) == 2:
            patch_type = v[0]
            v = v[1]

        if patch_type == "diff":
            diff: torch.Tensor = v[0]
            do_pad_weight = len(v) > 1 and v[1]["pad_weight"]
            if do_pad_weight and diff.shape != weight.shape:
                logger.debug("Pad weight {} from {} to shape: {}".format(key, weight.shape, diff.shape))
                weight = weight_adapter.base.pad_tensor_to_shape(weight, diff.shape)

            if strength != 0.0:
                if diff.shape != weight.shape:
                    logger.warning("SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, diff.shape, weight.shape))
                else:
                    weight += function(strength * memory_management.cast_to_device(diff, weight.device, weight.dtype))
        elif patch_type == "set":
            weight.copy_(v[0])
        else:
            raise ValueError(f'"{key}" of type "{patch_type}" is not recognized...')

        if old_weight is not None:
            weight = old_weight

    return weight


def get_parameter_devices(model):
    parameter_devices = {}
    for key, p in model.named_parameters():
        parameter_devices[key] = p.device
    return parameter_devices


def set_parameter_devices(model, parameter_devices):
    for key, device in parameter_devices.items():
        p = utils.get_attr(model, key)
        if not isinstance(p, torch.nn.Parameter) or p.device != device:
            p = utils.tensor2parameter(p.to(device=device))
            utils.set_attr_raw(model, key, p)
    return model
