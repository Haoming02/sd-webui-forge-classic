# https://github.com/BobJohnson24/ComfyUI-Flux2-INT8/blob/main/int8_quant.py

import torch

try:
    from backend.operations_triton import triton_int8_linear
except ImportError:
    # Triton not found, fall back to torch._int_mm
    _TRITON_AVAILABLE = False
else:
    _TRITON_AVAILABLE = True


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


def stochastic_round_int8_delta(x: torch.Tensor, scale: float | torch.Tensor, seed: int = 0) -> torch.Tensor:
    """
    Quantize a delta tensor to INT8 using stochastic rounding.
    Used for LoRA deltas to minimize quantization error.
    """
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)

    # Scale to INT8 range
    x_scaled = x / scale

    # Stochastic rounding
    x_floor = torch.floor(x_scaled)
    fraction = x_scaled - x_floor

    # Speed optimization: Create random values directly on the target device
    random_vals = torch.rand(x_scaled.shape, generator=generator, device=x.device, dtype=x_scaled.dtype)
    x_rounded = torch.where(random_vals < fraction, x_floor + 1, x_floor)

    return torch.clamp(x_rounded, -128, 127).to(torch.int8)


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


# region INT8 LoRA Adapter - High Precision, Low RAM Patching


from modules_forge.packages.comfy.weight_adapter import LoRAAdapter


class INT8LoRAPatchAdapter(LoRAAdapter):
    """
    Specialized LoRA adapter that patches INT8 weights IN-PLACE in INT8 space.
    """

    def __init__(self, loaded_keys, weights, weight_scale, seed=0):
        super().__init__(loaded_keys, weights)
        self.weight_scale = weight_scale
        self.seed = seed

    def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
        v = self.weights
        up, down, alpha = v[0], v[1], v[2]

        rank = down.shape[0] if down.ndim >= 2 else 1
        scale = (alpha / rank) * strength if alpha is not None else strength

        device = weight.device

        # Compute LoRA Delta in high-precision on GPU
        comp_device = torch.device("cuda") if torch.cuda.is_available() else device

        up_f = up.to(comp_device, dtype=intermediate_dtype)
        down_f = down.to(comp_device, dtype=intermediate_dtype)

        # Handle possible mid weights (LoCon/LoHA)
        if v[3] is not None:
            mid_f = v[3].to(comp_device, dtype=intermediate_dtype)
            lora_diff = torch.mm(up_f.flatten(1), torch.mm(mid_f.flatten(1), down_f.flatten(1))).reshape(weight.shape)
        else:
            lora_diff = torch.mm(up_f.flatten(1), down_f.flatten(1)).reshape(weight.shape)

        # Apply Patch
        if weight.dtype == torch.int8:
            # --- INT8 SPACE PATCHING ---
            delta_f = lora_diff * scale
            delta_int8 = stochastic_round_int8_delta(delta_f, self.weight_scale, self.seed)

            # Perform integer addition (int32 for safety) then clamp
            res = weight.to(comp_device, torch.int32) + delta_int8.to(comp_device, torch.int32)
            return torch.clamp(res, -128, 127).to(torch.int8).to(device)
        else:
            # Fallback: Standard Float Patching
            return weight + (lora_diff * scale).to(weight.device, weight.dtype)
