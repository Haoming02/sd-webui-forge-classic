from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from transformers import T5EncoderModel

import gc
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
from nunchaku.caching.diffusers_adapters.flux import apply_cache_on_transformer
from nunchaku.caching.fbcache import cache_context, create_cache_context
from nunchaku.lora.flux.compose import compose_lora
from nunchaku.models.linear import AWQW4A16Linear, SVDQW4A4Linear
from nunchaku.models.transformers.utils import patch_scale_key
from nunchaku.models.utils import CPUOffloadManager
from nunchaku.ops.fused import fused_gelu_mlp
from nunchaku.utils import load_state_dict_in_safetensors

from backend.args import dynamic_args
from backend.nn._qwen_lora import compose_loras_v2, reset_lora_v2
from backend.utils import process_img
from modules import shared


class NunchakuModelMixin(nn.Module):
    offload: bool = False

    def set_offload(self, offload: bool, use_pin_memory: bool, num_blocks_on_gpu: int):
        raise NotImplementedError

    def to(self, *args, **kwargs):
        args = (arg for arg in args if not isinstance(arg, torch.dtype))
        kwargs.pop("dtype", None)

        dev: bool = any(isinstance(arg, torch.device) for arg in args) or "device" in kwargs

        if self.offload and dev:
            return self
        else:
            return super().to(*args, **kwargs)


# region: Flux


class SVDQFluxTransformer2DModel(nn.Module):
    """https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/v1.0.0/wrappers/flux.py"""

    def __init__(self, config: dict):
        super().__init__()
        model = NunchakuFluxTransformer2dModel.from_pretrained(config.pop("filename"), offload=shared.opts.svdq_cpu_offload)
        model = apply_cache_on_transformer(transformer=model, residual_diff_threshold=shared.opts.svdq_cache_threshold)
        model.set_attention_impl(shared.opts.svdq_attention)

        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config
        self.loras = []

        # for first-block cache
        self._prev_timestep = None
        self._cache_context = None

    def forward(self, x, timestep, context, y, guidance, control=None, transformer_options={}, **kwargs):
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                timestep_float = timestep.item()
            else:
                timestep_float = timestep.flatten()[0].item()
        else:
            assert isinstance(timestep, float)
            timestep_float = timestep

        model = self.model
        assert isinstance(model, NunchakuFluxTransformer2dModel)

        bs, c, h_orig, w_orig = x.shape
        patch_size = self.config.get("patch_size", 2)
        h_len = (h_orig + (patch_size // 2)) // patch_size
        w_len = (w_orig + (patch_size // 2)) // patch_size

        img, img_ids = process_img(x)
        img_tokens = img.shape[1]

        ref_latents = dynamic_args.get("ref_latents", None)

        if ref_latents is not None:
            h = 0
            w = 0
            for ref in ref_latents:
                h_offset = 0
                w_offset = 0
                if ref.shape[-2] + h > ref.shape[-1] + w:
                    w_offset = w
                else:
                    h_offset = h

                kontext, kontext_ids = process_img(ref.to(x), index=1, h_offset=h_offset, w_offset=w_offset)
                img = torch.cat([img, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)
                h = max(h, ref.shape[-2] + h_offset)
                w = max(w, ref.shape[-1] + w_offset)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)

        # load and compose LoRA
        if self.loras != model.comfy_lora_meta_list:
            lora_to_be_composed = []
            for _ in range(max(0, len(model.comfy_lora_meta_list) - len(self.loras))):
                model.comfy_lora_meta_list.pop()
                model.comfy_lora_sd_list.pop()
            for i in range(len(self.loras)):
                meta = self.loras[i]
                if i >= len(model.comfy_lora_meta_list):
                    sd = load_state_dict_in_safetensors(meta[0])
                    model.comfy_lora_meta_list.append(meta)
                    model.comfy_lora_sd_list.append(sd)
                elif model.comfy_lora_meta_list[i] != meta:
                    if meta[0] != model.comfy_lora_meta_list[i][0]:
                        sd = load_state_dict_in_safetensors(meta[0])
                        model.comfy_lora_sd_list[i] = sd
                    model.comfy_lora_meta_list[i] = meta
                lora_to_be_composed.append(({k: v for k, v in model.comfy_lora_sd_list[i].items()}, meta[1]))

            composed_lora = compose_lora(lora_to_be_composed)

            if len(composed_lora) == 0:
                model.reset_lora()
            else:
                if "x_embedder.lora_A.weight" in composed_lora:
                    new_in_channels = composed_lora["x_embedder.lora_A.weight"].shape[1]
                    current_in_channels = model.x_embedder.in_features
                    if new_in_channels < current_in_channels:
                        model.reset_x_embedder()
                model.update_lora_params(composed_lora)

        controlnet_block_samples = None if control is None else [y.to(x.dtype) for y in control["input"]]
        controlnet_single_block_samples = None if control is None else [y.to(x.dtype) for y in control["output"]]

        if getattr(model, "_is_cached", False) or getattr(model, "residual_diff_threshold_multi", 0) != 0:
            # A more robust caching strategy
            cache_invalid = False

            # Check if timestamps have changed or are out of valid range
            if self._prev_timestep is None:
                cache_invalid = True
            elif self._prev_timestep < timestep_float + 1e-5:  # allow a small tolerance to reuse the cache
                cache_invalid = True

            if cache_invalid:
                self._cache_context = create_cache_context()

            # Update the previous timestamp
            self._prev_timestep = timestep_float
            with cache_context(self._cache_context):
                out = model(
                    hidden_states=img,
                    encoder_hidden_states=context,
                    pooled_projections=y,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance if self.config["guidance_embed"] else None,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                ).sample
        else:
            out = model(
                hidden_states=img,
                encoder_hidden_states=context,
                pooled_projections=y,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance if self.config["guidance_embed"] else None,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
            ).sample

        out = out[:, :img_tokens]
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=patch_size, pw=patch_size)
        out = out[:, :, :h_orig, :w_orig]

        self._prev_timestep = timestep_float
        return out

    def load_state_dict(self, *args, **kwargs):
        return [], []


# ========== T5 ========== #


def _forward(self: "T5EncoderModel", input_ids: torch.LongTensor, *args, **kwargs):
    outputs = self.encoder(input_ids=input_ids, *args, **kwargs)
    return outputs.last_hidden_state


class WrappedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input: torch.Tensor, *args, **kwargs):
        return self.embedding(input)

    @property
    def weight(self):
        return self.embedding.weight


class SVDQT5(torch.nn.Module):
    """https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/v1.0.0/nodes/models/text_encoder.py"""

    def __init__(self, path: str):
        super().__init__()

        transformer = NunchakuT5EncoderModel.from_pretrained(path)
        transformer.forward = types.MethodType(_forward, transformer)
        transformer.shared = WrappedEmbedding(transformer.shared)

        self.transformer = transformer
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))


# region: Qwen


from backend.memory_management import xformers_enabled

if xformers_enabled():
    from backend.attention import attention_xformers as attention_function
else:
    from backend.attention import attention_pytorch as attention_function

from backend.nn.flux import EmbedND
from backend.nn.qwen import (
    GELU,
    FeedForward,
    LastLayer,
    QwenImageTransformer2DModel,
    QwenTimestepProjEmbeddings,
    apply_rotary_emb,
)


class NunchakuGELU(GELU):

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        approximate: str = "none",
        bias: bool = True,
        **kwargs,
    ):
        super(GELU, self).__init__()
        self.proj = SVDQW4A4Linear(dim_in, dim_out, bias=bias, **kwargs)
        self.approximate = approximate


class NunchakuFeedForward(FeedForward):

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        inner_dim=None,
        bias: bool = True,
        **kwargs,
    ):
        super(FeedForward, self).__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.net = nn.ModuleList([])
        self.net.append(NunchakuGELU(dim, inner_dim, approximate="tanh", bias=bias, **kwargs))
        self.net.append(nn.Dropout(dropout))
        self.net.append(
            SVDQW4A4Linear(
                inner_dim,
                dim_out,
                bias=bias,
                act_unsigned=kwargs.get("precision", "int4") == "int4",
                **kwargs,
            )
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(self.net[0], NunchakuGELU):
            return fused_gelu_mlp(hidden_states, self.net[0].proj, self.net[2])
        else:
            for module in self.net:
                hidden_states = module(hidden_states)
            return hidden_states


class Attention(nn.Module):

    def __init__(
        self,
        query_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
        eps: float = 1e-5,
        out_bias: bool = True,
        out_dim: int = None,
        out_context_dim: int = None,
        **kwargs,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim
        self.heads = heads
        self.dim_head = dim_head
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.dropout = dropout

        # Q/K normalization for both streams
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=True)
        self.norm_added_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_added_k = nn.RMSNorm(dim_head, eps=eps)

        # Image stream projections: fused QKV for speed
        self.to_qkv = SVDQW4A4Linear(query_dim, self.inner_dim + self.inner_kv_dim * 2, bias=bias, **kwargs)

        # Text stream projections: fused QKV for speed
        self.add_qkv_proj = SVDQW4A4Linear(query_dim, self.inner_dim + self.inner_kv_dim * 2, bias=bias, **kwargs)

        # Output projections
        self.to_out = nn.ModuleList(
            [
                SVDQW4A4Linear(self.inner_dim, self.out_dim, bias=out_bias, **kwargs),
                nn.Dropout(dropout),
            ]
        )
        self.to_add_out = SVDQW4A4Linear(self.inner_dim, self.out_context_dim, bias=out_bias, **kwargs)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        seq_txt = encoder_hidden_states.shape[1]

        img_qkv = self.to_qkv(hidden_states)
        img_query, img_key, img_value = img_qkv.chunk(3, dim=-1)

        # Compute QKV for text stream (context projections)
        txt_qkv = self.add_qkv_proj(encoder_hidden_states)
        txt_query, txt_key, txt_value = txt_qkv.chunk(3, dim=-1)

        img_query = img_query.unflatten(-1, (self.heads, -1))
        img_key = img_key.unflatten(-1, (self.heads, -1))
        img_value = img_value.unflatten(-1, (self.heads, -1))

        txt_query = txt_query.unflatten(-1, (self.heads, -1))
        txt_key = txt_key.unflatten(-1, (self.heads, -1))
        txt_value = txt_value.unflatten(-1, (self.heads, -1))

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        # Concatenate image and text streams for joint attention
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Apply rotary embeddings
        joint_query = apply_rotary_emb(joint_query, image_rotary_emb)
        joint_key = apply_rotary_emb(joint_key, image_rotary_emb)

        joint_query = joint_query.flatten(start_dim=2)
        joint_key = joint_key.flatten(start_dim=2)
        joint_value = joint_value.flatten(start_dim=2)

        # Compute joint attention
        joint_hidden_states = attention_function(joint_query, joint_key, joint_value, self.heads, attention_mask)

        # Split results back to separate streams
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = self.to_out[0](img_attn_output)
        img_attn_output = self.to_out[1](img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class NunchakuQwenImageTransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        scale_shift: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.scale_shift = scale_shift
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Modulation and normalization for image stream
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            AWQW4A16Linear(dim, 6 * dim, bias=True),
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = NunchakuFeedForward(dim=dim, dim_out=dim, **kwargs)

        # Modulation and normalization for text stream
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            AWQW4A16Linear(dim, 6 * dim, bias=True),
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = NunchakuFeedForward(dim=dim, dim_out=dim, **kwargs)

        self.attn = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=eps,
            **kwargs,
        )

    def _modulate(self, x: torch.Tensor, mod_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if self.scale_shift != 0:
            scale.add_(self.scale_shift)
        return x * scale.unsqueeze(1) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # Nunchaku's mod_params is [B, 6*dim] instead of [B, dim*6]
        img_mod_params = img_mod_params.view(img_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(img_mod_params.shape[0], -1)
        txt_mod_params = txt_mod_params.view(txt_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(txt_mod_params.shape[0], -1)

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Joint attention computation (DoubleStreamLayerMegatron logic)
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream ("sample")
            encoder_hidden_states=txt_modulated,  # Text stream ("context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        return encoder_hidden_states, hidden_states


class NunchakuQwenImageTransformer2DModel(NunchakuModelMixin, QwenImageTransformer2DModel):
    """https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/v1.0.1/models/qwenimage.py"""

    def __init__(
        self,
        filename: str = None,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        scale_shift: float = 1.0,
        **kwargs,
    ):
        super(QwenImageTransformer2DModel, self).__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
        )

        self.txt_norm = nn.RMSNorm(joint_attention_dim, eps=1e-6)
        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                NunchakuQwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    scale_shift=scale_shift,
                    **kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = LastLayer(
            self.inner_dim,
            self.inner_dim,
        )
        self.proj_out = nn.Linear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=True,
        )

        self.set_offload(
            offload=shared.opts.svdq_cpu_offload,
            use_pin_memory=shared.opts.svdq_use_pin_memory,
            num_blocks_on_gpu=shared.opts.svdq_num_blocks_on_gpu,
        )

        self.loras = []
        self._applied_loras = []

    def forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        guidance: torch.Tensor = None,
        ref_latents=None,
        transformer_options={},
        control=None,
        **kwargs,
    ):

        device = x.device
        if self.offload:
            self.offload_manager.set_device(device)

        timestep = timesteps
        encoder_hidden_states = context
        encoder_hidden_states_mask = attention_mask

        hidden_states, img_ids, orig_shape = self.process_img(x)
        num_embeds = hidden_states.shape[1]

        ref_latents = dynamic_args.get("ref_latents", ref_latents)

        if ref_latents is not None:
            h = 0
            w = 0
            index = 0
            index_ref_method = kwargs.get("ref_latents_method", "index") == "index"
            for ref in ref_latents:
                if index_ref_method:
                    index += 1
                    h_offset = 0
                    w_offset = 0
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                kontext, kontext_ids, _ = self.process_img(ref.to(x), index=index, h_offset=h_offset, w_offset=w_offset)
                hidden_states = torch.cat([hidden_states, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)

        txt_start = round(
            max(
                ((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2,
                ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2,
            )
        )
        txt_ids = torch.arange(txt_start, txt_start + context.shape[1], device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids).squeeze(1).unsqueeze(2).to(x.dtype)
        del ids, txt_ids, img_ids

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance * 1000

        if self.loras != self._applied_loras:
            self._applied_loras = self.loras.copy()

            reset_lora_v2(self)
            self.set_offload(False, None, None)

            print("[Qwen] Composing LoRAs...")
            compose_loras_v2(self, self.loras)
            print("[Qwen] LoRAs Composed~")

            self.set_offload(
                offload=shared.opts.svdq_cpu_offload,
                use_pin_memory=shared.opts.svdq_use_pin_memory,
                num_blocks_on_gpu=shared.opts.svdq_num_blocks_on_gpu,
            )

        temb = self.time_text_embed(timestep, hidden_states) if guidance is None else self.time_text_embed(timestep, guidance, hidden_states)

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        # Setup compute stream for offloading
        compute_stream = torch.cuda.current_stream()
        if self.offload:
            self.offload_manager.initialize(compute_stream)

        for i, block in enumerate(self.transformer_blocks):
            with torch.cuda.stream(compute_stream):
                if self.offload:
                    block = self.offload_manager.get_block(i)
                if ("double_block", i) in blocks_replace:

                    def block_wrap(args):
                        out = {}
                        out["txt"], out["img"] = block(
                            hidden_states=args["img"],
                            encoder_hidden_states=args["txt"],
                            encoder_hidden_states_mask=encoder_hidden_states_mask,
                            temb=args["vec"],
                            image_rotary_emb=args["pe"],
                        )
                        return out

                    out = blocks_replace[("double_block", i)](
                        {"img": hidden_states, "txt": encoder_hidden_states, "vec": temb, "pe": image_rotary_emb},
                        {"original_block": block_wrap},
                    )
                    hidden_states = out["img"]
                    encoder_hidden_states = out["txt"]
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                    )
                # ControlNet helpers(device/dtype-safe residual adds)
                _control = control if control is not None else (transformer_options.get("control", None) if isinstance(transformer_options, dict) else None)
                if isinstance(_control, dict):
                    control_i = _control.get("input")
                    try:
                        _scale = float(_control.get("weight", _control.get("scale", 1.0)))
                    except Exception:
                        _scale = 1.0
                else:
                    control_i = None
                    _scale = 1.0
                if control_i is not None and i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        if getattr(add, "device", None) != hidden_states.device or getattr(add, "dtype", None) != hidden_states.dtype:
                            add = add.to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
                        t = min(hidden_states.shape[1], add.shape[1])
                        if t > 0:
                            hidden_states[:, :t].add_(add[:, :t], alpha=_scale)

            if self.offload:
                self.offload_manager.step(compute_stream)

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states[:, :num_embeds].view(orig_shape[0], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2)
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
        return hidden_states.reshape(orig_shape)[:, :, :, : x.shape[-2], : x.shape[-1]]

    def set_offload(self, offload: bool, use_pin_memory: bool, num_blocks_on_gpu: int):
        if offload == self.offload:
            return

        self.offload = offload
        if offload:
            self.offload_manager = CPUOffloadManager(
                self.transformer_blocks,
                use_pin_memory=use_pin_memory,
                on_gpu_modules=[
                    self.img_in,
                    self.txt_in,
                    self.txt_norm,
                    self.time_text_embed,
                    self.norm_out,
                    self.proj_out,
                ],
                num_blocks_on_gpu=num_blocks_on_gpu,
            )
        else:
            self.offload_manager = None
            gc.collect()
            torch.cuda.empty_cache()

    def load_state_dict(self, sd, *args, **kwargs):
        state_dict = self.state_dict()
        for k in state_dict.keys():
            if k not in sd:
                if "dummy" in k:
                    continue
                if ".wcscales" not in k:
                    raise ValueError(f"Key {k} not found in state_dict")
                sd[k] = torch.ones_like(state_dict[k])
        for n, m in self.named_modules():
            if isinstance(m, SVDQW4A4Linear):
                if m.wtscale is not None:
                    m.wtscale = sd.pop(f"{n}.wtscale", 1.0)

        return super().load_state_dict(sd, *args, **kwargs)


# region: Z-Image

from functools import wraps

from backend.nn.lumina import JointAttention, NextDiT, clamp_fp16


def fuse_to_svdquant_linear(linear1: nn.Linear, linear2: nn.Linear, **kwargs) -> SVDQW4A4Linear:
    assert linear1.in_features == linear2.in_features
    assert linear1.bias is None and linear2.bias is None
    return SVDQW4A4Linear(
        linear1.in_features,
        linear1.out_features + linear2.out_features,
        bias=False,
        torch_dtype=linear1.weight.dtype,
        device=linear1.weight.device,
        **kwargs,
    )


class NunchakuZImageAttention(JointAttention):

    def __init__(self, orig_attn: JointAttention, **kwargs):
        nn.Module.__init__(self)
        self.n_kv_heads = orig_attn.n_kv_heads
        self.n_local_heads = orig_attn.n_local_heads
        self.n_local_kv_heads = orig_attn.n_local_kv_heads
        self.n_rep = orig_attn.n_rep
        self.head_dim = orig_attn.head_dim

        self.qkv = SVDQW4A4Linear.from_linear(orig_attn.qkv, **kwargs)
        self.out = SVDQW4A4Linear.from_linear(orig_attn.out, **kwargs)

        self.q_norm = orig_attn.q_norm
        self.k_norm = orig_attn.k_norm

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        transformer_options={},
    ) -> torch.Tensor:
        return super().forward(x, x_mask, freqs_cis, transformer_options)


class NunchakuZImageFeedForward(nn.Module):

    def __init__(self, orig_ff: FeedForward, **kwargs):
        super().__init__()
        self.w13 = fuse_to_svdquant_linear(orig_ff.w1, orig_ff.w3, **kwargs)
        self.w2 = SVDQW4A4Linear.from_linear(orig_ff.w2, **kwargs)

    def _forward_silu_gating(self, x1, x3):
        return clamp_fp16(F.silu(x1) * x3)

    def forward(self, x: torch.Tensor):
        x = self.w13(x)
        x3, x1 = x.chunk(2, dim=-1)
        return self.w2(self._forward_silu_gating(x1, x3))


def patch_z_image_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    patched_state_dict = {}
    quant_sub_keys = ["wscales", "wcscales", "wtscale", "smooth_factor_orig", "smooth_factor", "proj_down", "proj_up"]

    for key, value in state_dict.items():
        if "attention.to_qkv" in key:
            patched_state_dict[key.replace("to_qkv", "qkv")] = value
        elif "attention.to_q" in key:
            q_weight = state_dict[key]
            k_weight = state_dict[key.replace("to_q", "to_k")]
            v_weight = state_dict[key.replace("to_q", "to_v")]
            patched_state_dict[key.replace("to_q", "qkv")] = torch.cat([q_weight, k_weight, v_weight], dim=0)
        elif "attention.to_k" in key or "attention.to_v" in key:
            continue
        elif "attention.to_out" in key:
            patched_state_dict[key.replace("to_out.0", "out")] = value
        elif "feed_forward.net.0.proj.qweight" in key:
            patched_state_dict[key.replace("net.0.proj", "w13")] = value
            for subkey in quant_sub_keys:
                quant_param_key = key.replace("qweight", subkey)
                if quant_param_key in state_dict:
                    patched_state_dict[quant_param_key.replace("net.0.proj", "w13")] = state_dict[quant_param_key]
        elif any("feed_forward.net.0.proj." + subkey in key for subkey in quant_sub_keys):
            continue
        elif "feed_forward.net.2.qweight" in key:
            patched_state_dict[key.replace("net.2", "w2")] = value
            for subkey in quant_sub_keys:
                quant_param_key = key.replace("qweight", subkey)
                if quant_param_key in state_dict:
                    patched_state_dict[quant_param_key.replace("net.2", "w2")] = state_dict[quant_param_key]
        elif any("feed_forward.net.2." + subkey in key for subkey in quant_sub_keys):
            continue
        elif "feed_forward.net.0.proj.weight" in key:
            w3, w1 = torch.chunk(value, chunks=2, dim=0)
            w2 = state_dict[key.replace("0.proj", "2")]
            patched_state_dict[key.replace("net.0.proj", "w1")] = w1
            patched_state_dict[key.replace("net.0.proj", "w2")] = w2
            patched_state_dict[key.replace("net.0.proj", "w3")] = w3
        elif "feed_forward.net.2.weight" in key:
            continue
        elif "attention.norm_q" in key:
            patched_state_dict[key.replace("norm_q", "q_norm")] = value
        elif "attention.norm_k" in key:
            patched_state_dict[key.replace("norm_k", "k_norm")] = value
        elif "all_final_layer.2-1" in key:
            patched_state_dict[key.replace("all_final_layer.2-1", "final_layer")] = value
        elif "all_x_embedder.2-1" in key:
            patched_state_dict[key.replace("all_x_embedder.2-1", "x_embedder")] = value
        else:
            patched_state_dict[key] = value

    return patched_state_dict


def patch_nunchaku_zimage(model: NextDiT, precision: str, rank: int):
    kwargs = {"precision": precision, "rank": rank}

    def patch_transformer_block(block_list: list[torch.nn.Module]):
        for _, block in enumerate(block_list):
            block.attention = NunchakuZImageAttention(block.attention, **kwargs)
            block.feed_forward = NunchakuZImageFeedForward(block.feed_forward, **kwargs)

    patch_transformer_block(model.layers)
    patch_transformer_block(model.noise_refiner)
    patch_transformer_block(model.context_refiner)

    _load_state_dict: Callable = model.load_state_dict

    @wraps(_load_state_dict)
    def load_state_dict(sd, *args, **kwargs):
        sd = patch_z_image_state_dict(sd)
        patch_scale_key(model, sd)
        return _load_state_dict(sd, *args, **kwargs)

    model.load_state_dict = load_state_dict

    return model
