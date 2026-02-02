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


# region Dynamic LoRA Hook


class DynamicLoRAHook:
    """
    Hook registered on the diffusion_model to synchronize dynamic LoRA attributes
    with the current ModelPatcher context at the start of each forward pass.
    """

    def __init__(self):
        self.current_lora_id = None

    def pre_forward(self, module, input_args, input_kwargs):
        # 1. Try to find transformer_options
        transformer_options = input_kwargs.get("transformer_options", {})
        if not transformer_options:
            # Fallback for models that pass it in context
            context = input_args[2] if len(input_args) > 2 else None
            if isinstance(context, dict) and "transformer_options" in context:
                transformer_options = context["transformer_options"]

        dynamic_loras = transformer_options.get("dynamic_loras", [])

        # 2. Generate a unique ID for this set of LoRAs
        # We use handles/strengths to detect changes
        lora_id = hash(tuple((id(d["patches"]), d["strength"]) for d in dynamic_loras)) if dynamic_loras else None

        if lora_id == self.current_lora_id:
            return None  # Already synchronized

        # 3. Synchronize all linear layers
        self.apply_composition(module, dynamic_loras)
        self.current_lora_id = lora_id
        return None

    def apply_composition(self, diffusion_model, dynamic_loras):
        # Pre-group patches by layer
        layer_patches = {}
        if dynamic_loras:
            for entry in dynamic_loras:
                strength = entry["strength"]
                for key, adapter in entry["patches"].items():
                    if key not in layer_patches:
                        layer_patches[key] = []
                    layer_patches[key].append((adapter, strength))

        # Update all modules
        for name, module in diffusion_model.named_modules():
            if not hasattr(module, "lora_A"):
                continue

            # Find patches for this module
            # ComfyUI keys are often 'diffusion_model.path.to.weight' or 'path.to.weight'
            possible_keys = [f"diffusion_model.{name}.weight", f"{name}.weight"]
            patches = None
            for pk in possible_keys:
                if pk in layer_patches:
                    patches = layer_patches[pk]
                    break

            if not patches:
                module.lora_A = None
                module.lora_B = None
                module.lora_alpha = None
                continue

            # Compose
            all_A = []
            all_B = []
            for adapter, strength in patches:
                v = adapter.weights
                up, down, alpha, mid = v[0], v[1], v[2], v[3]
                rank = down.shape[0] if down.ndim >= 2 else 1
                scale = (alpha / rank) * strength if alpha is not None else strength

                curr_A = down
                if mid is not None:
                    curr_A = torch.mm(mid.flatten(1), down.flatten(1)).reshape(down.shape)

                all_A.append(curr_A * scale)
                all_B.append(up)

            if all_A:
                device = getattr(module, "weight", torch.tensor(0)).device
                module.lora_A = torch.cat(all_A, dim=0).to(device)
                module.lora_B = torch.cat(all_B, dim=1).to(device)
                module.lora_alpha = None
            else:
                module.lora_A = None
                module.lora_B = None

    @classmethod
    def register(cls, diffusion_model):
        if not hasattr(diffusion_model, "_dynamic_lora_hook"):
            hook = cls()
            diffusion_model._dynamic_lora_hook = hook
            diffusion_model.register_forward_pre_hook(hook.pre_forward, with_kwargs=True)
        return diffusion_model._dynamic_lora_hook


# region load_lora_int8

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.patcher.unet import UnetPatcher


from backend.patcher.lora import load_lora, model_lora_keys_unet


def load_lora_int8(model: "UnetPatcher", lora: dict[str, torch.Tensor], strength: float, lora_name: str) -> "UnetPatcher":
    if strength == 0:
        return model

    model_patcher = model.clone()

    # 1. Get Patch Map
    key_map = model_lora_keys_unet(model_patcher.model)

    patch_dict, _unmatch = load_lora(lora, key_map)

    # 2. Register Global Hook (if not exists)
    DynamicLoRAHook.register(model_patcher.model.diffusion_model)

    # 3. Add to Dynamic LoRA list in transformer_options
    # This ensures ComfyUI's cloning handles everything and it's non-sticky
    if "transformer_options" not in model_patcher.model_options:
        model_patcher.model_options["transformer_options"] = {}

    opts = model_patcher.model_options["transformer_options"]
    if "dynamic_loras" not in opts:
        opts["dynamic_loras"] = []
    else:
        # Shallow copy the list to avoid modifying the parent patcher's list
        opts["dynamic_loras"] = opts["dynamic_loras"].copy()

    opts["dynamic_loras"].append({"name": lora_name, "strength": strength, "patches": patch_dict})

    return model_patcher
