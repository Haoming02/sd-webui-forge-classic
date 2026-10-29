"""Sticky fp16 compute / SageAttention cast for the loaded UNet.

The switches are applied once to the live model. They stay applied after
override_settings restores the UI options. The next generation that sees a
switch off reloads the checkpoint instead of converting the weights back.
"""

import contextvars
import logging

import torch

from backend.logging import setup_logger

logger = logging.getLogger("fp16_switch")
setup_logger(logger)

sage_fp16_cast_active: contextvars.ContextVar[bool] = contextvars.ContextVar("sage_fp16_cast_active", default=False)


def _dtype_name(dt) -> str:
    return str(dt).replace("torch.", "")


def _is_nunchaku(module: torch.nn.Module) -> bool:
    return type(module).__module__ == "backend.nn.svdq"


def cast_floating(obj, dtype: torch.dtype):
    """Cast floating tensors inside control residuals to the UNet compute dtype."""
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and obj.dtype != dtype:
            return obj.to(dtype=dtype)
        return obj
    if isinstance(obj, dict):
        return {key: cast_floating(value, dtype) for key, value in obj.items()}
    if isinstance(obj, list):
        return [cast_floating(value, dtype) for value in obj]
    if isinstance(obj, tuple):
        return tuple(cast_floating(value, dtype) for value in obj)
    return obj


def _enable_manual_cast(module: torch.nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, torch.nn.Embedding):
            continue
        if hasattr(layer, "parameters_manual_cast") and not layer.parameters_manual_cast:
            layer.parameters_manual_cast = True


def apply_fp16_switch(sd_model, desired: tuple[bool, bool], already: tuple[bool, bool] = (False, False)) -> bool:
    """Enable compute fp16 and/or the Sage fp16 cast. Returns True if the module changed."""
    want_compute, want_sage = desired
    had_compute, had_sage = already
    changed = False
    kmodel = sd_model.forge_objects.unet.model
    diffusion = kmodel.diffusion_model

    if want_compute and not had_compute:
        if _is_nunchaku(diffusion):
            logger.info("Nunchaku UNet keeps its compute dtype; fp16 compute switch skipped")
        elif kmodel.computation_dtype != torch.float16:
            old = kmodel.computation_dtype
            if getattr(diffusion, "storage_dtype", None) != torch.float16:
                _enable_manual_cast(diffusion)
            kmodel.computation_dtype = torch.float16
            diffusion.computation_dtype = torch.float16
            logger.info("Diffusion compute dtype %s -> float16", _dtype_name(old))
            changed = True

    if want_sage and not had_sage and not getattr(kmodel, "sage_fp16_cast", False):
        kmodel.sage_fp16_cast = True
        from backend import attention

        if attention.attention_function is attention.attention_sage:
            logger.info("SageAttention fp16 cast enabled for this UNet")
        else:
            logger.warning("SageAttention fp16 cast is on, but the active attention backend is not SageAttention; q/k/v will not be cast")
        changed = True

    return changed
