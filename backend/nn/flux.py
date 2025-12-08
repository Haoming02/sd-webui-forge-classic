# implementation of Flux for Forge
# Copyright Forge 2024
# Reference: https://github.com/black-forest-labs/flux


import math

import torch
from einops import rearrange
from torch import nn

from backend import memory_management
from backend.args import dynamic_args
from backend.attention import attention_function
from backend.utils import fp16_fix, process_img, tensor2parameter


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pe: torch.Tensor, mask=None, transformer_options={}) -> torch.Tensor:
    q_shape = q.shape
    k_shape = k.shape

    if pe is not None:
        q = q.to(dtype=pe.dtype).reshape(*q.shape[:-1], -1, 1, 2)
        k = k.to(dtype=pe.dtype).reshape(*k.shape[:-1], -1, 1, 2)
        q = (pe[..., 0] * q[..., 0] + pe[..., 1] * q[..., 1]).reshape(*q_shape).type_as(v)
        k = (pe[..., 0] * k[..., 0] + pe[..., 1] * k[..., 1]).reshape(*k_shape).type_as(v)

    heads = q.shape[1]
    return attention_function(q, k, v, heads, skip_reshape=True, mask=mask, transformer_options=transformer_options)


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    if memory_management.is_device_mps(pos.device) or memory_management.is_intel_xpu() or memory_management.directml_enabled:
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
    return x_out.reshape(*x.shape).type_as(x)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> torch.Tensor:
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class EmbedND(nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        del ids, n_axes
        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        x = self.silu(self.in_layer(x))
        return self.out_layer(x)


if hasattr(torch, "rms_norm"):
    functional_rms_norm = torch.rms_norm
else:

    def functional_rms_norm(x, normalized_shape, weight, eps):
        if x.dtype in [torch.bfloat16, torch.float32]:
            n = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps) * weight
        else:
            n = torch.rsqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps).to(x.dtype) * weight
        return x * n


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = None  # to trigger module_profile
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6
        self.normalized_shape = [dim]

    def forward(self, x):
        if self.scale.dtype != x.dtype:
            self.scale = tensor2parameter(self.scale.to(dtype=x.dtype))
        return functional_rms_norm(x, self.normalized_shape, self.scale, self.eps)


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q, k, v):
        del v
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(k), k.to(q)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, pe):
        qkv = self.qkv(x)

        # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        B, L, _ = qkv.shape
        qkv = qkv.view(B, L, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        del qkv

        q, k = self.norm(q, k, v)

        x = attention(q, k, v, pe=pe)
        del q, k, v

        x = self.proj(x)
        return x


class Modulation(nn.Module):
    def __init__(self, dim, double):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec):
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return out


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, qkv_bias=False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img, txt, vec, pe):
        img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = self.img_mod(vec)

        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift
        del img_mod1_shift, img_mod1_scale
        img_qkv = self.img_attn.qkv(img_modulated)
        del img_modulated

        # img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        B, L, _ = img_qkv.shape
        H = self.num_heads
        D = img_qkv.shape[-1] // (3 * H)
        img_q, img_k, img_v = img_qkv.view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        del img_qkv

        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = self.txt_mod(vec)
        del vec

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift
        del txt_mod1_shift, txt_mod1_scale
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        del txt_modulated

        # txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        B, L, _ = txt_qkv.shape
        txt_q, txt_k, txt_v = txt_qkv.view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        del txt_qkv

        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        del txt_q, img_q
        k = torch.cat((txt_k, img_k), dim=2)
        del txt_k, img_k
        v = torch.cat((txt_v, img_v), dim=2)
        del txt_v, img_v

        attn = attention(q, k, v, pe=pe)
        del pe, q, k, v
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]
        del attn

        img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        del img_attn, img_mod1_gate
        img = img + img_mod2_gate * self.img_mlp((1 + img_mod2_scale) * self.img_norm2(img) + img_mod2_shift)
        del img_mod2_gate, img_mod2_scale, img_mod2_shift

        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        del txt_attn, txt_mod1_gate
        txt = txt + txt_mod2_gate * self.txt_mlp((1 + txt_mod2_scale) * self.txt_norm2(txt) + txt_mod2_shift)
        del txt_mod2_gate, txt_mod2_scale, txt_mod2_shift

        txt = fp16_fix(txt)

        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_scale=None):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x, vec, pe):
        mod_shift, mod_scale, mod_gate = self.modulation(vec)
        del vec
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
        del mod_shift, mod_scale
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        del x_mod

        # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        qkv = qkv.view(qkv.size(0), qkv.size(1), 3, self.num_heads, self.hidden_size // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        del qkv

        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe=pe)
        del q, k, v, pe
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=2))
        del attn, mlp

        x = x + mod_gate * output
        del mod_gate, output

        x = fp16_fix(x)

        return x


class LastLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        del vec
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        del scale, shift
        x = self.linear(x)
        return x


class IntegratedFluxTransformer2DModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, vec_in_dim: int, context_in_dim: int, hidden_size: int, mlp_ratio: float, num_heads: int, depth: int, depth_single_blocks: int, axes_dim: list[int], theta: int, patch_size: int, qkv_bias: bool, guidance_embed: bool):
        super().__init__()

        self.guidance_embed = guidance_embed
        self.patch_size = patch_size
        self.in_channels = in_channels * patch_size * patch_size
        self.out_channels = out_channels * patch_size * patch_size

        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")

        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)
        self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if guidance_embed else nn.Identity()
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def inner_forward(self, img, img_ids, txt, txt_ids, timesteps, y, guidance=None):
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)
        del y, guidance
        ids = torch.cat((txt_ids, img_ids), dim=1)
        del txt_ids, img_ids
        pe = self.pe_embedder(ids)
        del ids
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        del pe
        img = img[:, txt.shape[1] :, ...]
        del txt
        img = self.final_layer(img, vec)
        del vec
        return img

    def forward(self, x, timestep, context, y, guidance=None, control=None, transformer_options={}, **kwargs):
        bs, c, h_orig, w_orig = x.shape
        h_len = (h_orig + (self.patch_size // 2)) // self.patch_size
        w_len = (w_orig + (self.patch_size // 2)) // self.patch_size

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

        out = self.inner_forward(img, img_ids, context, txt_ids, timestep, y, guidance)
        del img, img_ids, txt_ids, timestep, context
        out = out[:, :img_tokens]
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=self.patch_size, pw=self.patch_size)
        out = out[:, :, :h_orig, :w_orig]
        del h_len, w_len, bs
        return out
