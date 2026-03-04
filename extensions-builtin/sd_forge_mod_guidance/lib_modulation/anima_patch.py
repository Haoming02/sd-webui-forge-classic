# https://github.com/Anzhc/Anima-Mod-Guidance-ComfyUI-Node/blob/main/anima_patch.py

from functools import wraps
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from backend.patcher.unet import UnetPatcher

import torch
import torch.nn.functional as F

from backend.nn.anima import Anima
from backend.utils import pad_to_patch_size

from .adapter import get_typed_adapter

WRAPPER_KEY = "anima_mod_guidance"
STATE_KEY = "anima_mod_guidance_state"

ORIG_FORWARD: Callable = None


def _normalize_layer_range(start_layer: int, end_layer: int, total_blocks: int):
    if end_layer < 0:
        end_layer = total_blocks - 1

    start_layer = max(0, int(start_layer))
    end_layer = min(total_blocks - 1, int(end_layer))

    assert start_layer < end_layer
    return start_layer, end_layer


def _prepare_pooled_for_batch(pooled: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype):
    if pooled.ndim == 1:
        pooled = pooled.unsqueeze(0)
    if pooled.shape[0] == 1:
        pooled = pooled.expand(batch_size, -1)
    return pooled.to(device=device, dtype=dtype)


def _project_clip_pooled(pooled, adapter_state):
    x = F.linear(
        pooled,
        adapter_state["text_embedder_clip.linear_1.weight"],
        adapter_state["text_embedder_clip.linear_1.bias"],
    )
    x = F.silu(x)
    x = F.linear(
        x,
        adapter_state["text_embedder_clip.linear_2.weight"],
        adapter_state["text_embedder_clip.linear_2.bias"],
    )
    return x


def register_modulation_wrapper(
    model_patcher: "UnetPatcher",
    adapter_path: str,
    clip_base_pooled: torch.Tensor,
    clip_positive_pooled: torch.Tensor,
    clip_negative_pooled: torch.Tensor,
    w: float,
    start_layer: int,
    end_layer: int,
):
    transformer_options = model_patcher.model_options.setdefault("transformer_options", {})
    transformer_options[STATE_KEY] = {
        "adapter_path": adapter_path,
        "clip_base_pooled": clip_base_pooled,
        "clip_positive_pooled": clip_positive_pooled,
        "clip_negative_pooled": clip_negative_pooled,
        "w": float(w),
        "start_layer": int(start_layer),
        "end_layer": int(end_layer),
    }

    assert model_patcher.model.config.huggingface_repo.endswith("Anima")

    global ORIG_FORWARD
    ORIG_FORWARD = Anima.forward
    Anima.forward = _forward_with_modulation


def unpatch():
    global ORIG_FORWARD
    if ORIG_FORWARD is not None:
        Anima.forward = ORIG_FORWARD
        ORIG_FORWARD = None


@wraps(ORIG_FORWARD)
def _forward_with_modulation(diffusion_model: Anima, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, fps: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None, **kwargs):
    transformer_options: dict = kwargs.get("transformer_options", {})
    if (state := transformer_options.get(STATE_KEY, None)) is None:
        return ORIG_FORWARD(diffusion_model, x, timesteps, context, fps, padding_mask, **kwargs)

    orig_shape = list(x.shape)
    x = pad_to_patch_size(x, (diffusion_model.patch_temporal, diffusion_model.patch_spatial, diffusion_model.patch_spatial))
    x_B_C_T_H_W = x
    timesteps_B_T = timesteps
    crossattn_emb = context

    x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = diffusion_model.prepare_embedded_sequence(
        x_B_C_T_H_W,
        fps=fps,
        padding_mask=padding_mask,
    )

    if timesteps_B_T.ndim == 1:
        timesteps_B_T = timesteps_B_T.unsqueeze(1)

    t_embedding_B_T_D, adaln_lora_B_T_3D = diffusion_model.t_embedder[1](diffusion_model.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype))
    t_embedding_B_T_D = diffusion_model.t_embedding_norm(t_embedding_B_T_D)

    diffusion_model.affline_emb = t_embedding_B_T_D
    diffusion_model.crossattn_emb = crossattn_emb

    if x_B_T_H_W_D.dtype == torch.float16:
        x_B_T_H_W_D = x_B_T_H_W_D.float()

    adapter_state = get_typed_adapter(
        state["adapter_path"],
        device=t_embedding_B_T_D.device,
        dtype=t_embedding_B_T_D.dtype,
    )

    batch_size = t_embedding_B_T_D.shape[0]
    pooled_base = _prepare_pooled_for_batch(state["clip_base_pooled"], batch_size, t_embedding_B_T_D.device, t_embedding_B_T_D.dtype)
    pooled_pos = _prepare_pooled_for_batch(state["clip_positive_pooled"], batch_size, t_embedding_B_T_D.device, t_embedding_B_T_D.dtype)
    pooled_neg = _prepare_pooled_for_batch(state["clip_negative_pooled"], batch_size, t_embedding_B_T_D.device, t_embedding_B_T_D.dtype)

    pooled_base_proj = _project_clip_pooled(pooled_base, adapter_state)
    pooled_pos_proj = _project_clip_pooled(pooled_pos, adapter_state)
    pooled_neg_proj = _project_clip_pooled(pooled_neg, adapter_state)
    pooled_mod = pooled_base_proj + float(state["w"]) * (pooled_pos_proj - pooled_neg_proj)

    total_blocks = len(diffusion_model.blocks)
    start_layer, end_layer = _normalize_layer_range(state["start_layer"], state["end_layer"], total_blocks)

    block_kwargs = {
        "rope_emb_L_1_1_D": rope_emb_L_1_1_D.unsqueeze(1).unsqueeze(0),
        "extra_per_block_pos_emb": extra_pos_emb,
        "transformer_options": transformer_options,
    }

    adaln_steps = adaln_lora_B_T_3D.shape[1]
    for block_index, block in enumerate(diffusion_model.blocks):
        if start_layer <= block_index <= end_layer:
            per_block_scale = adapter_state["scales"][block_index].unsqueeze(0) * pooled_mod
            per_block_scale = per_block_scale.unsqueeze(1).expand(-1, adaln_steps, -1)
            adaln_for_block = adaln_lora_B_T_3D + per_block_scale
        else:
            adaln_for_block = adaln_lora_B_T_3D

        x_B_T_H_W_D = block(
            x_B_T_H_W_D,
            t_embedding_B_T_D,
            crossattn_emb,
            adaln_lora_B_T_3D=adaln_for_block,
            **block_kwargs,
        )

    x_B_T_H_W_O = diffusion_model.final_layer(
        x_B_T_H_W_D.to(crossattn_emb.dtype),
        t_embedding_B_T_D,
        adaln_lora_B_T_3D=adaln_lora_B_T_3D,
    )
    x_B_C_Tt_Hp_Wp = diffusion_model.unpatchify(x_B_T_H_W_O)[:, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]]
    return x_B_C_Tt_Hp_Wp
