# https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/main/int8_quant.py

import torch

try:
    from backend.operations_triton import triton_int8_linear, triton_int8_linear_per_row
except ImportError:
    # Triton not found, fall back to torch._int_mm
    _TRITON_AVAILABLE = False
else:
    _TRITON_AVAILABLE = True


CONVROT_GROUP_SIZE = 256


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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.operations import ForgeOperationsInt8

from backend.patcher.lora import merge_lora_to_weight
from backend.patcher.unet import UnetPatcher
from backend.quant_rotation import build_hadamard, rotate_weight
from backend.utils import get_attr, set_attr


class INT8ModelPatcher(UnetPatcher):

    def _process_online_loras(self):
        for key in self.online_patches:
            self.patch_weight_to_device(key, online=True)

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, *, online=False):
        if (not online) and (key not in self.patches):
            return
        if online and (key not in self.online_patches):
            return

        # Check if this is one of our INT8 modules
        module_path = key.rsplit(".", 1)[0]

        try:
            module: "ForgeOperationsInt8.Linear" = get_attr(self.model, module_path)
        except Exception:
            return super().patch_weight_to_device(key, device_to, inplace_update)

        if not getattr(module, "_is_quantized", False):
            return super().patch_weight_to_device(key, device_to, inplace_update)

        if online:
            # --- DYNAMIC LORA PATH ---
            # Build a list of (down_scaled, up, start, size) per patch.
            # Keeping patches separate preserves the offset info needed for
            # fused QKV layers where each of Q/K/V targets a different output slice.
            patches = self.online_patches[key]
            weight = get_attr(self.model, key)
            device = weight.device if weight is not None else self.offload_device
            lora_patches = []
            for p in patches:
                strength_patch = p[0]  # float
                adapter = p[1]  # the LoRA adapter object
                strength_model = p[2]  # float
                offset = p[3] if len(p) > 3 else None  # (dim, start, size) or None

                if not hasattr(adapter, "weights"):
                    continue

                strength = strength_patch * strength_model
                weights = adapter.weights
                # Standard LoRA: (up, down, alpha, mid, dora_scale, reshape)
                if len(weights) == 6:
                    up, down, alpha, mid, dora, reshape = weights
                    rank = down.shape[0] if down.ndim >= 2 else 1
                    scale = (alpha / rank) * strength if alpha is not None else strength

                    down_scaled = down.flatten(1) * scale
                    if mid is not None:
                        down_scaled = torch.mm(mid.flatten(1), down.flatten(1)) * scale

                    # If this layer has ConvRot applied, rotate the 'down' matrix
                    # so the LoRA delta is coherent with the rotated weight basis:
                    #   W_rot = W @ H^T  =>  ΔW_rot = ΔW @ H^T  =>  rotate down only
                    if getattr(module, "_use_convrot", False) and down_scaled.shape[1] % CONVROT_GROUP_SIZE == 0:
                        try:
                            group_size = getattr(module, "_convrot_groupsize", CONVROT_GROUP_SIZE)
                            H = build_hadamard(group_size, device=down_scaled.device, dtype=down_scaled.dtype)
                            down_scaled = rotate_weight(down_scaled, H, group_size=group_size)
                        except Exception:
                            pass

                    # Extract offset: which output rows this patch targets
                    start, size = None, None
                    if offset is not None:
                        _dim, start, size = offset  # dim is always 0 for linear weights

                    lora_patches.append((down_scaled.to(device), up.flatten(1).to(device), start, size))

            module.lora_patches = lora_patches

        else:
            # --- BAKE-IN LORA PATH (Dequant → Patch → Quant) ---
            # Works with the native ComfyUI LoRA Loader (and also INT8LoraLoader).
            # All patches are applied in float space via ComfyUI's standard mechanism,
            # then the result is re-quantized back to INT8.
            patches = self.patches[key]
            weight_int8 = get_attr(self.model, key)
            scale = module._get_weight_scale()

            if device_to is None:
                device_to = weight_int8.device

            # Save original weight so unpatch_model can restore it.
            # Must use the same namedtuple format as ComfyUI's base patcher
            # (collections.namedtuple('Dimension', ['weight', 'inplace_update']))
            # otherwise unpatch_model crashes with AttributeError on bk.inplace_update.
            if key not in self.backup:
                import collections

                BackupEntry = collections.namedtuple("Dimension", ["weight", "inplace_update"])
                self.backup[key] = BackupEntry(
                    weight=weight_int8.to(device=self.offload_device, copy=inplace_update),
                    inplace_update=inplace_update,
                )

            # 1. Dequantize to float (move scale to device_to since it lives on CPU)
            if isinstance(scale, torch.Tensor):
                scale = scale.to(device_to)
            weight_float = dequantize(weight_int8.to(device_to), scale)

            # 2. Handle ConvRot: de-rotate into weight space before patching
            use_convrot = getattr(module, "_use_convrot", False)
            if use_convrot:
                group_size = getattr(module, "_convrot_groupsize", CONVROT_GROUP_SIZE)
                try:
                    H = build_hadamard(group_size, device=device_to, dtype=weight_float.dtype)
                    weight_float = rotate_weight(weight_float, H, group_size=group_size)
                except ImportError:
                    pass

            # 3. Patch in float space using ComfyUI's standard mechanism.
            # calculate_weight handles LoRA, LoHA, LoKR, DoRA, etc.
            patches_list = self.patches.get(key, [])
            patched_weight_float = merge_lora_to_weight(patches_list, weight_float, key)

            # 4. Handle ConvRot: re-rotate
            if use_convrot:
                patched_weight_float = rotate_weight(patched_weight_float, H, group_size=group_size)

            # 5. Re-quantize back to INT8 using the original scale
            patched_weight_int8 = quantize_int8(patched_weight_float, scale)  # stochastic_round_int8_delta(patched_weight_float, scale)
            # I'm not really sure whether to stochastic round or not, results seem to depend on a per-lora basis.
            # If quality is of the utmost importance, I recommend Pre-Lora instead of worrying about this.

            # 6. Move back to original device and store
            patched_weight_int8 = patched_weight_int8.to(weight_int8.device)

            if inplace_update:
                weight_int8.data.copy_(patched_weight_int8)
            else:
                set_attr(self.model, key, patched_weight_int8)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_patches"):
                    module.lora_patches = []
        return super().unpatch_model(device_to, unpatch_weights)
