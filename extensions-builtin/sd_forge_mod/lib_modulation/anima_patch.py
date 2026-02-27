import torch
import torch.nn.functional as F

import comfy.ldm.common_dit
import comfy.patcher_extension

from .adapter import get_typed_adapter

WRAPPER_KEY = "anima_mod_guidance"
STATE_KEY = "anima_mod_guidance_state"


def _extract_transformer_options(args, kwargs):
    transformer_options = kwargs.get("transformer_options")
    if isinstance(transformer_options, dict):
        return transformer_options

    if len(args) > 0 and isinstance(args[-1], dict):
        return args[-1]
    if len(args) > 1 and isinstance(args[-2], dict):
        return args[-2]
    return {}


def _normalize_layer_range(start_layer, end_layer, total_blocks):
    if total_blocks <= 0:
        raise RuntimeError("Anima model has no blocks to modulate.")

    if end_layer < 0:
        end_layer = total_blocks - 1

    start_layer = max(0, int(start_layer))
    end_layer = min(total_blocks - 1, int(end_layer))
    if start_layer > end_layer:
        raise RuntimeError(
            f"Invalid layer range: start_layer={start_layer}, end_layer={end_layer}, total_blocks={total_blocks}."
        )
    return start_layer, end_layer


def _prepare_pooled_for_batch(pooled, batch_size, device, dtype):
    if pooled.ndim == 1:
        pooled = pooled.unsqueeze(0)
    if pooled.ndim != 2:
        raise RuntimeError(f"Expected pooled tensor with rank 2, got rank {pooled.ndim}.")
    if pooled.shape[0] == 1:
        pooled = pooled.expand(batch_size, -1)
    elif pooled.shape[0] != batch_size:
        raise RuntimeError(
            f"Pooled tensor batch mismatch: pooled batch={pooled.shape[0]}, expected {batch_size}."
        )
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
    model_patcher,
    adapter_path,
    clip_base_pooled,
    clip_positive_pooled,
    clip_negative_pooled,
    w,
    start_layer,
    end_layer,
):
    model_patcher.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)

    transformer_options = model_patcher.model_options.setdefault("transformer_options", {})
    transformer_options[STATE_KEY] = {
        "adapter_path": adapter_path,
        "clip_base_pooled": clip_base_pooled.detach().float().cpu().contiguous(),
        "clip_positive_pooled": clip_positive_pooled.detach().float().cpu().contiguous(),
        "clip_negative_pooled": clip_negative_pooled.detach().float().cpu().contiguous(),
        "w": float(w),
        "start_layer": int(start_layer),
        "end_layer": int(end_layer),
    }

    model_patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        WRAPPER_KEY,
        anima_modulation_forward_wrapper,
    )


def anima_modulation_forward_wrapper(executor, *args, **kwargs):
    transformer_options = _extract_transformer_options(args, kwargs)
    state = transformer_options.get(STATE_KEY, None)
    if state is None:
        return executor(*args, **kwargs)

    try:
        x = args[0]
        timesteps = args[1]
        context = args[2]
        fps = args[3] if len(args) > 3 else kwargs.get("fps", None)
        padding_mask = args[4] if len(args) > 4 else kwargs.get("padding_mask", None)
    except Exception as exc:
        raise RuntimeError(
            "Anima Mod Guidance failed: unexpected forward signature for MiniTrainDIT._forward."
        ) from exc

    return _forward_with_modulation(
        executor.class_obj,
        x=x,
        timesteps=timesteps,
        context=context,
        fps=fps,
        padding_mask=padding_mask,
        transformer_options=transformer_options,
        state=state,
    )


def _forward_with_modulation(
    diffusion_model,
    x,
    timesteps,
    context,
    fps,
    padding_mask,
    transformer_options,
    state,
):
    if not hasattr(diffusion_model, "blocks") or not hasattr(diffusion_model, "prepare_embedded_sequence"):
        raise RuntimeError(
            "Anima Mod Guidance failed: model is not compatible with MiniTrainDIT internals."
        )
    if not getattr(diffusion_model, "use_adaln_lora", False):
        raise RuntimeError(
            "Anima Mod Guidance requires an Anima/Cosmos model with use_adaln_lora=True."
        )

    orig_shape = list(x.shape)
    x = comfy.ldm.common_dit.pad_to_patch_size(
        x, (diffusion_model.patch_temporal, diffusion_model.patch_spatial, diffusion_model.patch_spatial)
    )
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

    t_embedding_B_T_D, adaln_lora_B_T_3D = diffusion_model.t_embedder[1](
        diffusion_model.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype)
    )
    t_embedding_B_T_D = diffusion_model.t_embedding_norm(t_embedding_B_T_D)

    if adaln_lora_B_T_3D is None:
        raise RuntimeError(
            "Anima Mod Guidance failed: model did not produce AdaLN-LoRA embeddings."
        )

    diffusion_model.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding_B_T_D.detach()}
    diffusion_model.affline_emb = t_embedding_B_T_D
    diffusion_model.crossattn_emb = crossattn_emb

    if extra_pos_emb is not None and x_B_T_H_W_D.shape != extra_pos_emb.shape:
        raise RuntimeError(
            "Anima Mod Guidance failed: extra positional embedding shape mismatch "
            f"{tuple(x_B_T_H_W_D.shape)} != {tuple(extra_pos_emb.shape)}."
        )

    if x_B_T_H_W_D.dtype == torch.float16:
        x_B_T_H_W_D = x_B_T_H_W_D.float()

    adapter_state, adapter_meta = get_typed_adapter(
        state["adapter_path"],
        diffusion_model,
        device=t_embedding_B_T_D.device,
        dtype=t_embedding_B_T_D.dtype,
    )

    batch_size = t_embedding_B_T_D.shape[0]
    pooled_base = _prepare_pooled_for_batch(
        state["clip_base_pooled"], batch_size, t_embedding_B_T_D.device, t_embedding_B_T_D.dtype
    )
    pooled_pos = _prepare_pooled_for_batch(
        state["clip_positive_pooled"], batch_size, t_embedding_B_T_D.device, t_embedding_B_T_D.dtype
    )
    pooled_neg = _prepare_pooled_for_batch(
        state["clip_negative_pooled"], batch_size, t_embedding_B_T_D.device, t_embedding_B_T_D.dtype
    )

    pooled_dim = pooled_base.shape[1]
    if pooled_dim != adapter_meta["pooled_dim"]:
        raise RuntimeError(
            "Anima Mod Guidance failed: pooled embedding dim mismatch for clip_base_conditioning "
            f"({pooled_dim} vs expected {adapter_meta['pooled_dim']})."
        )
    if pooled_pos.shape[1] != adapter_meta["pooled_dim"] or pooled_neg.shape[1] != adapter_meta["pooled_dim"]:
        raise RuntimeError(
            "Anima Mod Guidance failed: pooled embedding dim mismatch for clip_positive_conditioning/"
            "clip_negative_conditioning."
        )

    pooled_base_proj = _project_clip_pooled(pooled_base, adapter_state)
    pooled_pos_proj = _project_clip_pooled(pooled_pos, adapter_state)
    pooled_neg_proj = _project_clip_pooled(pooled_neg, adapter_state)
    pooled_mod = pooled_base_proj + float(state["w"]) * (pooled_pos_proj - pooled_neg_proj)

    total_blocks = len(diffusion_model.blocks)
    start_layer, end_layer = _normalize_layer_range(
        state["start_layer"], state["end_layer"], total_blocks
    )

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
    x_B_C_Tt_Hp_Wp = diffusion_model.unpatchify(x_B_T_H_W_O)[
        :, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]
    ]
    return x_B_C_Tt_Hp_Wp
