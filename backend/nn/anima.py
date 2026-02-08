"""
Anima diffusion model implementation for Forge.

Copyright and license notice:
- This file contains code derived from ComfyUI.
- Upstream project: https://github.com/comfyanonymous/ComfyUI
- Upstream files:
  - https://github.com/comfyanonymous/ComfyUI/blob/f350a842611f4d75da7104c2d2965f45989089b9/comfy/ldm/anima/model.py
  - https://github.com/comfyanonymous/ComfyUI/blob/f350a842611f4d75da7104c2d2965f45989089b9/comfy/ldm/cosmos/predict2.py
- ComfyUI license: GNU General Public License v3.0.
- The upstream license text is available at:
  https://github.com/comfyanonymous/ComfyUI/blob/f350a842611f4d75da7104c2d2965f45989089b9/LICENSE

Porting policy in this file:
- Keep class/function names close to upstream where possible.
- Mark intentional local behavior differences with `PORT_NOTE`.
- Avoid silent behavior changes; if needed, document the reason inline.

Local modifications from upstream:
- Integrated Anima-specific text conditioning in-model by accepting
  `t5xxl_ids` / `t5xxl_weights` via kwargs and applying adapter/weights in
  `IntegratedAnimaTransformer.forward`.
- Added optional in-model padding of cross-attention sequence length to 512 to
  match Comfy runtime behavior for Anima conditioning.
- Added checkpoint key conversion helper (`convert_anima_state_dict`) for
  single-file weights stored under `net.*` prefix.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from backend.attention import attention_function
from backend.utils import pad_to_patch_size


def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
    t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    t_out = t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)
    return t_out


def normalize(x: torch.Tensor, eps: float = 0) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=math.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class VideoRopePosition3DEmb(nn.Module):
    def __init__(
        self,
        *,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
    ):
        super().__init__()
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.enable_fps_modulation = enable_fps_modulation

        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"

        self.register_buffer("dim_spatial_range", torch.arange(0, dim_h, 2).float()[: (dim_h // 2)] / dim_h, persistent=False)
        self.register_buffer("dim_temporal_range", torch.arange(0, dim_t, 2).float()[: (dim_t // 2)] / dim_t, persistent=False)

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    def forward(self, x_b_t_h_w_c: torch.Tensor, fps: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_ntk_factor = self.h_ntk_factor
        w_ntk_factor = self.w_ntk_factor
        t_ntk_factor = self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range.to(device=x_b_t_h_w_c.device))
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range.to(device=x_b_t_h_w_c.device))
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range.to(device=x_b_t_h_w_c.device))

        b, t, h, w, _ = x_b_t_h_w_c.shape
        seq = torch.arange(max(h, w, t), dtype=torch.float, device=x_b_t_h_w_c.device)
        uniform_fps = (fps is None) or isinstance(fps, (int, float)) or (fps.min() == fps.max())
        assert uniform_fps or b == 1 or t == 1, "For non-uniform fps, batch size must be 1 (unless T=1)"

        half_emb_h = torch.outer(seq[:h], h_spatial_freqs)
        half_emb_w = torch.outer(seq[:w], w_spatial_freqs)

        if fps is None or self.enable_fps_modulation is False:
            half_emb_t = torch.outer(seq[:t], temporal_freqs)
        else:
            half_emb_t = torch.outer(seq[:t] / fps * self.base_fps, temporal_freqs)

        half_emb_h = torch.stack([torch.cos(half_emb_h), -torch.sin(half_emb_h), torch.sin(half_emb_h), torch.cos(half_emb_h)], dim=-1)
        half_emb_w = torch.stack([torch.cos(half_emb_w), -torch.sin(half_emb_w), torch.sin(half_emb_w), torch.cos(half_emb_w)], dim=-1)
        half_emb_t = torch.stack([torch.cos(half_emb_t), -torch.sin(half_emb_t), torch.sin(half_emb_t), torch.cos(half_emb_t)], dim=-1)

        em_t_h_w_d = torch.cat(
            [
                repeat(half_emb_t, "t d x -> t h w d x", h=h, w=w),
                repeat(half_emb_h, "h d x -> t h w d x", t=t, w=w),
                repeat(half_emb_w, "w d x -> t h w d x", t=t, h=h),
            ],
            dim=-2,
        )

        return rearrange(em_t_h_w_d, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()


class LearnablePosEmbAxis(nn.Module):
    def __init__(self, *, interpolation: str, model_channels: int, len_h: int, len_w: int, len_t: int):
        super().__init__()
        self.interpolation = interpolation
        assert self.interpolation in ["crop"], f"Unknown interpolation method {self.interpolation}"

        self.pos_emb_h = nn.Parameter(torch.empty(len_h, model_channels))
        self.pos_emb_w = nn.Parameter(torch.empty(len_w, model_channels))
        self.pos_emb_t = nn.Parameter(torch.empty(len_t, model_channels))

    def forward(self, x_b_t_h_w_c: torch.Tensor) -> torch.Tensor:
        b, t, h, w, _ = x_b_t_h_w_c.shape
        emb_h_h = self.pos_emb_h[:h].to(device=x_b_t_h_w_c.device, dtype=x_b_t_h_w_c.dtype)
        emb_w_w = self.pos_emb_w[:w].to(device=x_b_t_h_w_c.device, dtype=x_b_t_h_w_c.dtype)
        emb_t_t = self.pos_emb_t[:t].to(device=x_b_t_h_w_c.device, dtype=x_b_t_h_w_c.dtype)
        emb = repeat(emb_t_t, "t d-> b t h w d", b=b, h=h, w=w) + repeat(emb_h_h, "h d-> b t h w d", b=b, t=t, w=w) + repeat(emb_w_w, "w d-> b t h w d", b=b, t=t, h=h)
        return normalize(emb, eps=1e-6)


class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(x)))


class Attention(nn.Module):
    def __init__(self, query_dim: int, context_dim: Optional[int] = None, n_heads: int = 8, head_dim: int = 64) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Identity()

    def compute_qkv(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, rope_emb: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim), (q, k, v))

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        if self.is_selfattn and rope_emb is not None:
            q = apply_rotary_pos_emb(q, rope_emb)
            k = apply_rotary_pos_emb(k, rope_emb)
        return q, k, v

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, rope_emb: Optional[torch.Tensor] = None, transformer_options: Optional[dict] = None) -> torch.Tensor:
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        out = attention_function(
            q.movedim(1, 2),
            k.movedim(1, 2),
            v.movedim(1, 2),
            self.n_heads,
            skip_reshape=True,
            transformer_options=transformer_options or {},
        )
        return self.output_dropout(self.output_proj(out))


class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_b_t: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps_b_t.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / half_dim
        emb = torch.exp(exponent)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return rearrange(emb, "(b t) d -> b t d", b=timesteps_b_t.shape[0], t=timesteps_b_t.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

    def forward(self, sample: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.linear_2(self.activation(self.linear_1(sample)))
        if self.use_adaln_lora:
            return sample, emb
        return emb, None


class PatchEmbed(nn.Module):
    def __init__(self, spatial_patch_size: int, temporal_patch_size: int, in_channels: int = 3, out_channels: int = 768):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Sequential(
            Rearrange("b c (t r) (h m) (w n) -> b t h w (c r m n)", r=temporal_patch_size, m=spatial_patch_size, n=spatial_patch_size),
            nn.Linear(in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, spatial_patch_size: int, temporal_patch_size: int, out_channels: int, use_adaln_lora: bool = False, adaln_lora_dim: int = 256):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False)
        self.hidden_size = hidden_size
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, adaln_lora_dim, bias=False), nn.Linear(adaln_lora_dim, 2 * hidden_size, bias=False))
        else:
            self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x_b_t_h_w_d: torch.Tensor, emb_b_t_d: torch.Tensor, adaln_lora_b_t_3d: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_adaln_lora:
            assert adaln_lora_b_t_3d is not None
            shift_b_t_d, scale_b_t_d = (self.adaln_modulation(emb_b_t_d) + adaln_lora_b_t_3d[:, :, : 2 * self.hidden_size]).chunk(2, dim=-1)
        else:
            shift_b_t_d, scale_b_t_d = self.adaln_modulation(emb_b_t_d).chunk(2, dim=-1)
        shift_b_t_1_1_d = rearrange(shift_b_t_d, "b t d -> b t 1 1 d")
        scale_b_t_1_1_d = rearrange(scale_b_t_d, "b t d -> b t 1 1 d")
        x_b_t_h_w_d = self.layer_norm(x_b_t_h_w_d) * (1 + scale_b_t_1_1_d) + shift_b_t_1_1_d
        return self.linear(x_b_t_h_w_d)


class Block(nn.Module):
    def __init__(self, x_dim: int, context_dim: int, num_heads: int, mlp_ratio: float = 4.0, use_adaln_lora: bool = False, adaln_lora_dim: int = 256):
        super().__init__()
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(x_dim, None, num_heads, x_dim // num_heads)
        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(x_dim, context_dim, num_heads, x_dim // num_heads)
        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))
        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, adaln_lora_dim, bias=False), nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, adaln_lora_dim, bias=False), nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, adaln_lora_dim, bias=False), nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False))
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

    def forward(
        self,
        x_b_t_h_w_d: torch.Tensor,
        emb_b_t_d: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_l_1_1_d: Optional[torch.Tensor] = None,
        adaln_lora_b_t_3d: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        transformer_options: Optional[dict] = None,
    ) -> torch.Tensor:
        residual_dtype = x_b_t_h_w_d.dtype
        compute_dtype = emb_b_t_d.dtype
        if extra_per_block_pos_emb is not None:
            x_b_t_h_w_d = x_b_t_h_w_d + extra_per_block_pos_emb

        if self.use_adaln_lora:
            assert adaln_lora_b_t_3d is not None
            shift_self, scale_self, gate_self = (self.adaln_modulation_self_attn(emb_b_t_d) + adaln_lora_b_t_3d).chunk(3, dim=-1)
            shift_cross, scale_cross, gate_cross = (self.adaln_modulation_cross_attn(emb_b_t_d) + adaln_lora_b_t_3d).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = (self.adaln_modulation_mlp(emb_b_t_d) + adaln_lora_b_t_3d).chunk(3, dim=-1)
        else:
            shift_self, scale_self, gate_self = self.adaln_modulation_self_attn(emb_b_t_d).chunk(3, dim=-1)
            shift_cross, scale_cross, gate_cross = self.adaln_modulation_cross_attn(emb_b_t_d).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_b_t_d).chunk(3, dim=-1)

        shift_self = rearrange(shift_self, "b t d -> b t 1 1 d")
        scale_self = rearrange(scale_self, "b t d -> b t 1 1 d")
        gate_self = rearrange(gate_self, "b t d -> b t 1 1 d")
        shift_cross = rearrange(shift_cross, "b t d -> b t 1 1 d")
        scale_cross = rearrange(scale_cross, "b t d -> b t 1 1 d")
        gate_cross = rearrange(gate_cross, "b t d -> b t 1 1 d")
        shift_mlp = rearrange(shift_mlp, "b t d -> b t 1 1 d")
        scale_mlp = rearrange(scale_mlp, "b t d -> b t 1 1 d")
        gate_mlp = rearrange(gate_mlp, "b t d -> b t 1 1 d")

        b, t, h, w, _ = x_b_t_h_w_d.shape
        normalized = self.layer_norm_self_attn(x_b_t_h_w_d) * (1 + scale_self) + shift_self
        attn_out = self.self_attn(rearrange(normalized.to(compute_dtype), "b t h w d -> b (t h w) d"), None, rope_emb=rope_emb_l_1_1_d, transformer_options=transformer_options or {})
        attn_out = rearrange(attn_out, "b (t h w) d -> b t h w d", t=t, h=h, w=w)
        x_b_t_h_w_d = x_b_t_h_w_d + gate_self.to(residual_dtype) * attn_out.to(residual_dtype)

        normalized = self.layer_norm_cross_attn(x_b_t_h_w_d) * (1 + scale_cross) + shift_cross
        attn_out = self.cross_attn(rearrange(normalized.to(compute_dtype), "b t h w d -> b (t h w) d"), crossattn_emb, rope_emb=rope_emb_l_1_1_d, transformer_options=transformer_options or {})
        attn_out = rearrange(attn_out, "b (t h w) d -> b t h w d", t=t, h=h, w=w)
        x_b_t_h_w_d = x_b_t_h_w_d + gate_cross.to(residual_dtype) * attn_out.to(residual_dtype)

        normalized = self.layer_norm_mlp(x_b_t_h_w_d) * (1 + scale_mlp) + shift_mlp
        x_b_t_h_w_d = x_b_t_h_w_d + gate_mlp.to(residual_dtype) * self.mlp(normalized.to(compute_dtype)).to(residual_dtype)
        return x_b_t_h_w_d


class MiniTrainDIT(nn.Module):
    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        pos_emb_cls: str = "rope3d",
        pos_emb_learnable: bool = True,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        **_,
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation

        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim

        self._build_pos_embed()
        self.t_embedder = nn.Sequential(Timesteps(model_channels), TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora))
        self.x_embedder = PatchEmbed(spatial_patch_size=patch_spatial, temporal_patch_size=patch_temporal, in_channels=in_channels + int(concat_padding_mask), out_channels=model_channels)
        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
        )
        self.t_embedding_norm = nn.RMSNorm(model_channels, eps=1e-6)

    def _build_pos_embed(self) -> None:
        if self.pos_emb_cls != "rope3d":
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")
        self.pos_embedder = VideoRopePosition3DEmb(
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
        )
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder = LearnablePosEmbAxis(
                interpolation=self.pos_emb_interpolation,
                model_channels=self.model_channels,
                len_h=self.max_img_h // self.patch_spatial,
                len_w=self.max_img_w // self.patch_spatial,
                len_t=self.max_frames // self.patch_temporal,
            )

    def preprocess_text_embeds(self, text_embeds: torch.Tensor, text_ids: Optional[torch.Tensor]) -> torch.Tensor:
        return text_embeds

    def prepare_embedded_sequence(
        self,
        x_b_c_t_h_w: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.concat_padding_mask:
            if padding_mask is None:
                padding_mask = torch.zeros(
                    x_b_c_t_h_w.shape[0],
                    1,
                    x_b_c_t_h_w.shape[3],
                    x_b_c_t_h_w.shape[4],
                    dtype=x_b_c_t_h_w.dtype,
                    device=x_b_c_t_h_w.device,
                )
            else:
                padding_mask = torch.nn.functional.interpolate(padding_mask.float(), size=list(x_b_c_t_h_w.shape[-2:]), mode="nearest").to(x_b_c_t_h_w.dtype)
            x_b_c_t_h_w = torch.cat([x_b_c_t_h_w, padding_mask.unsqueeze(1).repeat(1, 1, x_b_c_t_h_w.shape[2], 1, 1)], dim=1)

        x_b_t_h_w_d = self.x_embedder(x_b_c_t_h_w)
        extra_pos_emb = self.extra_pos_embedder(x_b_t_h_w_d) if self.extra_per_block_abs_pos_emb else None
        return x_b_t_h_w_d, self.pos_embedder(x_b_t_h_w_d, fps=fps), extra_pos_emb

    def unpatchify(self, x_b_t_h_w_m: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x_b_t_h_w_m,
            "b t h w (p1 p2 tt c) -> b c (t tt) (h p1) (w p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            tt=self.patch_temporal,
        )

    def _forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        orig_shape = list(x.shape)
        x = pad_to_patch_size(x, (self.patch_temporal, self.patch_spatial, self.patch_spatial))
        x_b_c_t_h_w = x
        x_b_t_h_w_d, rope_emb_l_d, extra_pos_emb = self.prepare_embedded_sequence(x_b_c_t_h_w, fps=fps, padding_mask=padding_mask)

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        t_embedding_b_t_d, adaln_lora_b_t_3d = self.t_embedder[1](self.t_embedder[0](timesteps).to(x_b_t_h_w_d.dtype))
        t_embedding_b_t_d = self.t_embedding_norm(t_embedding_b_t_d)

        text_ids = kwargs.get("t5xxl_ids", kwargs.get("text_ids", None))
        crossattn_emb = self.preprocess_text_embeds(context, text_ids)

        block_kwargs = {
            "rope_emb_l_1_1_d": rope_emb_l_d.unsqueeze(1).unsqueeze(0),
            "adaln_lora_b_t_3d": adaln_lora_b_t_3d,
            "extra_per_block_pos_emb": extra_pos_emb,
            "transformer_options": kwargs.get("transformer_options", {}),
        }

        if x_b_t_h_w_d.dtype == torch.float16:
            x_b_t_h_w_d = x_b_t_h_w_d.float()

        for block in self.blocks:
            x_b_t_h_w_d = block(x_b_t_h_w_d, t_embedding_b_t_d, crossattn_emb, **block_kwargs)

        x_b_t_h_w_o = self.final_layer(x_b_t_h_w_d.to(crossattn_emb.dtype), t_embedding_b_t_d, adaln_lora_b_t_3d=adaln_lora_b_t_3d)
        x_b_c_t_h_w = self.unpatchify(x_b_t_h_w_o)[:, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]]
        return x_b_c_t_h_w

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._forward(x, timesteps, context, fps=fps, padding_mask=padding_mask, **kwargs)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_adapter(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class AdapterAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)
        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states = apply_rotary_pos_emb_adapter(query_states, cos, sin)
            cos, sin = position_embeddings_context
            key_states = apply_rotary_pos_emb_adapter(key_states, cos, sin)
        attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, use_self_attn=True):
        super().__init__()
        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.norm_self_attn = nn.RMSNorm(model_dim, eps=1e-6)
            self.self_attn = AdapterAttention(model_dim, model_dim, num_heads, model_dim // num_heads)
        self.norm_cross_attn = nn.RMSNorm(model_dim, eps=1e-6)
        self.cross_attn = AdapterAttention(model_dim, source_dim, num_heads, model_dim // num_heads)
        self.norm_mlp = nn.RMSNorm(model_dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(model_dim, int(model_dim * mlp_ratio)), nn.GELU(), nn.Linear(int(model_dim * mlp_ratio), model_dim))

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None, position_embeddings=None, position_embeddings_context=None):
        if self.use_self_attn:
            x = x + self.self_attn(self.norm_self_attn(x), mask=target_attention_mask, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings)
        x = x + self.cross_attn(self.norm_cross_attn(x), mask=source_attention_mask, context=context, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        return x + self.mlp(self.norm_mlp(x))


class LLMAdapter(nn.Module):
    def __init__(self, source_dim=1024, target_dim=1024, model_dim=1024, num_layers=6, num_heads=16, use_self_attn=True):
        super().__init__()
        self.embed = nn.Embedding(32128, target_dim)
        self.in_proj = nn.Identity() if model_dim == target_dim else nn.Linear(target_dim, model_dim)
        self.rotary_emb = RotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList([TransformerBlock(source_dim, model_dim, num_heads=num_heads, use_self_attn=use_self_attn) for _ in range(num_layers)])
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = nn.RMSNorm(target_dim, eps=1e-6)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)
        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        x = self.in_proj(self.embed(target_input_ids))
        context = source_hidden_states
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(
                x,
                context,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings_context,
            )
        return self.norm(self.out_proj(x))


class IntegratedAnimaTransformer(MiniTrainDIT):
    def __init__(self, **config):
        super().__init__(**config)
        self.llm_adapter = LLMAdapter()

    def preprocess_text_embeds(self, text_embeds: torch.Tensor, text_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if text_ids is not None:
            text_embeds = self.llm_adapter(text_embeds, text_ids)
        # PORT_NOTE:
        # Comfy model_base pads c_crossattn to length 512 for Anima before UNet call.
        # Forge keeps this in-model to preserve equivalent runtime behavior.
        if text_embeds.shape[1] < 512:
            text_embeds = torch.nn.functional.pad(text_embeds, (0, 0, 0, 512 - text_embeds.shape[1]))
        return text_embeds

    def forward(self, x, timestep, context, control=None, transformer_options=None, **kwargs):
        del control
        squeeze_time = False
        if x.ndim == 4:
            x = x.unsqueeze(2)
            squeeze_time = True

        padding_mask = kwargs.get("padding_mask", None)
        if padding_mask is None:
            padding_mask = torch.zeros((x.shape[0], 1, x.shape[-2], x.shape[-1]), device=x.device, dtype=x.dtype)

        t5_ids = kwargs.get("t5xxl_ids", kwargs.get("text_ids", None))
        t5_weights = kwargs.get("t5xxl_weights", None)
        if t5_ids is not None:
            context = self.preprocess_text_embeds(context, t5_ids)
            if t5_weights is not None:
                if t5_weights.ndim == 1:
                    t5_weights = t5_weights.unsqueeze(0)
                w = t5_weights.unsqueeze(-1).to(context)
                n = min(w.shape[1], context.shape[1])
                context[:, :n] = context[:, :n] * w[:, :n]
        else:
            context = self.preprocess_text_embeds(context, None)

        passthrough_kwargs = dict(kwargs)
        passthrough_kwargs.pop("t5xxl_ids", None)
        passthrough_kwargs.pop("t5xxl_weights", None)
        passthrough_kwargs.pop("text_ids", None)

        out = super().forward(
            x=x,
            timesteps=timestep,
            context=context,
            padding_mask=padding_mask,
            transformer_options=transformer_options or {},
            **passthrough_kwargs,
        )
        if squeeze_time:
            out = out[:, :, 0]
        return out


def convert_anima_state_dict(state_dict: dict[str, torch.Tensor], unet_config: dict | None = None) -> dict[str, torch.Tensor]:
    del unet_config
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        # PORT_NOTE:
        # Single-file Anima checkpoints store UNet weights under `net.*`.
        # Forge internal UNet module expects keys without this prefix.
        if key.startswith("net."):
            out[key[4:]] = value
    return out
