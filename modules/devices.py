import contextlib

import torch

from backend import memory_management

cpu: torch.device = torch.device("cpu")
fp8: bool = False
device: torch.device = memory_management.get_torch_device()
device_gfpgan = device_esrgan = device_codeformer = device
dtype_vae: torch.dtype = memory_management.vae_dtype()
dtype_unet: torch.dtype = memory_management.unet_dtype()
dtype_inference: torch.dtype = memory_management.inference_cast(dtype_unet, device)
dtype: torch.dtype = torch.float32 if dtype_unet is torch.float32 else torch.float16
unet_needs_upcast: bool = False


def has_xpu() -> bool:
    return memory_management.is_device_xpu(device)


def has_mps() -> bool:
    return memory_management.is_device_mps(device)


def get_cuda_device_id() -> int:
    return device.index


def get_cuda_device_string() -> str:
    return str(device)


def get_optimal_device_name() -> str:
    return device.type


def get_optimal_device() -> torch.device:
    return device


def get_device_for(*args, **kwargs) -> torch.device:
    return device


def torch_gc():
    memory_management.soft_empty_cache()


def autocast(*args, **kwargs):
    return contextlib.nullcontext()


def without_autocast(*args, **kwargs):
    return contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x: torch.Tensor, *args, **kwargs):
    if torch.isnan(x).any():
        memory_management.logger.warning("Encountered NaN in Latent" + ("; Try --disable-sage" if memory_management.sage_enabled() else ""))
        x.nan_to_num_(nan=0.0, posinf=1.0, neginf=0.0)
        # raise NansException
