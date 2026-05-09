# https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/BobJohnson24-Beta/int8_quant.py

import torch

try:
    from backend.operations_triton import triton_int8_linear, triton_int8_linear_per_row
except ImportError:
    # Triton not found, fall back to torch._int_mm
    _TRITON_AVAILABLE = False
else:
    _TRITON_AVAILABLE = True


QUAROT_GROUP_SIZE = 256


# region Quantization Utils


def quantize_int8(x: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)


def quantize_int8_tensorwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale


def quantize_int8_axiswise(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale


def dequantize(q: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    return q.float() * scale


# region LinearW8A8 Core


@torch.no_grad()
def int8_forward_dynamic(x: torch.Tensor, weight: torch.Tensor, weight_scale: float | torch.Tensor, bias: torch.Tensor | None, compute_dtype: torch.dtype) -> torch.Tensor:
    """Forward with dynamic per-token activation quantization."""

    # --- FAST PATH: Triton Fused Kernel ---
    if _TRITON_AVAILABLE and x.is_cuda:
        return triton_int8_linear(x, weight, weight_scale, bias, compute_dtype)

    # --- SLOW PATH: Standard PyTorch ---
    # Quantize activations per row (dynamic)
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)

    # INT8 Matmul (Outputs Int32)
    res = torch._int_mm(x_8, weight.T)

    # Dequantize: (res * weight_scale * x_scale)
    # Note: Creating intermediate Float tensors here is VRAM heavy
    res_scaled = res.float().mul_(weight_scale * x_scale).to(compute_dtype)

    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)
    return res_scaled


@torch.no_grad()
def int8_forward_dynamic_per_row(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None, compute_dtype: torch.dtype) -> torch.Tensor:
    """Forward with dynamic per-token activation quantization and per-row weight quantization.

    Args:
        x: Input activations [batch, in_features]
        weight: INT8 weight matrix [out_features, in_features]
        weight_scale: Per-row weight scales [out_features, 1]
        bias: Optional bias
        compute_dtype: Output dtype
    """
    # --- FAST PATH: Triton Fused Kernel (per-row) ---
    if _TRITON_AVAILABLE and x.is_cuda:
        return triton_int8_linear_per_row(x, weight, weight_scale, bias, compute_dtype)

    # --- SLOW PATH: Standard PyTorch ---
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)

    # INT8 Matmul (Outputs Int32)
    res = torch._int_mm(x_8, weight.T)  # [batch, out_features]

    # Dequantize with per-row weight scales
    # res[i,j] = sum_k(x_8[i,k] * weight[j,k]) * x_scale[i] * weight_scale[j]
    # Broadcasting: res * x_scale * weight_scale.T
    res_scaled = res.float().mul_(x_scale).mul_(weight_scale.T).to(compute_dtype)

    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)
    return res_scaled


# region load_lora_int8


def load_lora_int8(*args, **kwargs):
    # TODO
    raise NotImplementedError("W.I.P")
