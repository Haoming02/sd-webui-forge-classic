# https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/qwen_image21/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend import memory_management
from backend.args import dynamic_args
from backend.attention import attention_function
from backend.nn.flux import EmbedND, timestep_embedding
from backend.nn.qwen import TimestepEmbedding
from backend.operations import main_stream_worker, weights_manual_cast
from backend.operations_mixed_precision import linear_input_act
from backend.quant_ops import ck
from backend.utils import fp16_fix


class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps

    def forward(self, x):
        w = memory_management.cast_to(self.weight, dtype=x.dtype, device=x.device) + 1.0
        return F.rms_norm(x.float(), w.shape, w, self.eps).to(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, in_dim, hidden_size, eps=1e-6):
        super().__init__()
        self.text_norm = ZeroCenteredRMSNorm(in_dim, eps=eps)
        self.in_layer = nn.Linear(in_dim, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim, sample_proj_bias=False)

    def forward(self, timestep, dtype):
        return self.timestep_embedder(timestep_embedding(timestep.float(), 256).to(dtype))


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, fused=True):
        super().__init__()
        self.fused = fused
        if fused:
            self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        else:
            self.proj = nn.Linear(dim, hidden_dim, bias=False)
            self.gate_layer = nn.Linear(dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        if self.fused:
            return linear_input_act(self.out, self.gate_up(x), "swiglu")
        return self.out(F.silu(self.gate_layer(x)) * self.proj(x))


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, eps=1e-6):
        super().__init__()
        self.heads = heads
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=False)])
        self.norm_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps)

    def forward(self, x, pe, attn_fn, prefix_len, transformer_options={}):
        B, N, _ = x.shape
        q = self.to_q(x).view(B, N, self.heads, -1)
        k = self.to_k(x).view(B, N, self.heads, -1)
        v = self.to_v(x).view(B, N, self.heads, -1)

        q_scale, _, q_stream = weights_manual_cast(self.norm_q, q)
        k_scale, _, k_stream = weights_manual_cast(self.norm_k, k)
        with main_stream_worker(q_scale, None, q_stream), main_stream_worker(k_scale, None, k_stream):
            q, k = ck.rms_rope(q, k, pe, q_scale, k_scale, self.norm_q.eps)

        return self.to_out[0](attn_fn(q, k, v, self.heads))


def _split_rows(p):
    return p[-1:].unsqueeze(1), p[:-1].unsqueeze(1)


def _modulated_norm(norm, x, scale, prefix_len, zero):
    s_prefix, s_target = scale
    out = ck.adaln(x, s_target, zero, norm.eps)
    if prefix_len:
        out[:, :prefix_len] = ck.adaln(x[:, :prefix_len], s_prefix, zero, norm.eps)
    return out


def _gated_residual(x, y, gate, prefix_len):
    g_prefix, g_target = gate
    x[:, prefix_len:].addcmul_(y[:, prefix_len:], g_target)
    if prefix_len:
        x[:, :prefix_len].addcmul_(y[:, :prefix_len], g_prefix)
    return x


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=3, eps=1e-6, fused_mlp=True):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(dim, num_attention_heads, attention_head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = SwiGLUFeedForward(dim, dim * mlp_ratio, fused=fused_mlp)

    def forward(self, x, mod, pe, attn_fn, prefix_len, transformer_options={}):
        scale1, gate1, scale2, gate2, zero = mod
        x = _gated_residual(x, self.attn(_modulated_norm(self.img_norm1, x, scale1, prefix_len, zero), pe, attn_fn, prefix_len, transformer_options), gate1, prefix_len)
        x = _gated_residual(x, self.img_mlp(_modulated_norm(self.img_norm2, x, scale2, prefix_len, zero)), gate2, prefix_len)

        return fp16_fix(x)


class LastLayer(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False)

    def forward(self, x, temb):
        scale = self.linear(F.silu(temb)).unsqueeze(1)
        return ck.adaln(x, scale, torch.zeros_like(scale[:1]), self.norm.eps)


def block_causal_attention(segments, transformer_options={}):

    def attn(q, k, v, heads, preferred_attention=None):
        outs = [attention_function(q[:, start:end].flatten(2), k[:, :end].flatten(2), v[:, :end].flatten(2), heads, mask=mask, transformer_options=transformer_options, preferred_attention=preferred_attention) for start, end, mask in segments]
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]

    return attn


class QwenImage21Transformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        num_layers=32,
        attention_head_dim=128,
        num_attention_heads=32,
        context_in_dim=4096,
        mlp_ratio=3,
        axes_dims_rope=(16, 56, 56),
        eps=1e-6,
        fused_mlp=True,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = TimestepProjEmbeddings(self.inner_dim)
        self.txt_in = TextProjection(context_in_dim, self.inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels, self.inner_dim, bias=False)

        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))

        self.transformer_blocks = nn.ModuleList([QwenImage21TransformerBlock(self.inner_dim, num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, eps=eps, fused_mlp=fused_mlp) for _ in range(num_layers)])

        self.norm_out = LastLayer(self.inner_dim, eps=eps)
        self.proj_out = nn.Linear(self.inner_dim, out_channels, bias=False)

    def build_sequence(self, x: torch.Tensor, context: torch.Tensor, ref_latents: list[torch.Tensor], image_slots: list[int]):
        txt = self.txt_in(context)
        slots = (image_slots + [txt.shape[1]] * len(ref_latents))[: len(ref_latents)]
        bounds = [0] + slots + [txt.shape[1]]

        parts, ids, segments = [], [], []
        pos, length = 0, 0
        for (start, end), img in zip(zip(bounds[:-1], bounds[1:]), ref_latents + [x]):
            n = end - start
            if n > 0:
                parts.append(txt[:, start:end])
                ids.append(torch.arange(pos, pos + n, device=x.device, dtype=torch.float32).unsqueeze(1).expand(n, 3))
                segments.append((length, length + n, torch.ones((n, length + n), dtype=torch.bool, device=x.device).tril(length)))
                pos += n
                length += n
            h, w = img.shape[-2:]

            if x.size(0) > 1 and img is not x:
                img = img.expand(x.shape[0], *img.shape[1:])

            parts.append(self.img_in(img.flatten(2).transpose(1, 2).to(x)))
            hh = torch.arange(h, device=x.device, dtype=torch.float32) - (h - h // 2) + 0.5 * (h % 2 - x.shape[-2] % 2)
            ww = torch.arange(w, device=x.device, dtype=torch.float32) - (w - w // 2) + 0.5 * (w % 2 - x.shape[-1] % 2)
            ids.append(torch.stack([torch.full((h, w), pos, device=x.device, dtype=torch.float32), hh[:, None].expand(h, w), ww[None, :].expand(h, w)], dim=-1).flatten(0, 1))
            segments.append((length, length + h * w, None))
            pos += max(h, w)
            length += h * w

        pe = self.pe_embedder(torch.cat(ids, dim=0).unsqueeze(0)).transpose(1, 2).contiguous()
        return torch.cat(parts, dim=1), pe, segments

    def forward(self, x, timesteps, context, ref_latents=None, image_slots=None, transformer_options={}, **kwargs):
        B, C, H, W = x.shape
        dtype = x.dtype
        ref_latents = list(dynamic_args.ref_latents or [])
        image_slots = list(dynamic_args.image_slots or [])

        hidden_states, pe, segments = self.build_sequence(x, context, ref_latents, image_slots)
        prefix_len = hidden_states.shape[1] - H * W
        patches = transformer_options.get("patches", {})
        for p in patches.get("post_input", []):
            out = p({"img": hidden_states, "pe": pe, "transformer_options": transformer_options})
            hidden_states, pe = out["img"], out.get("pe", pe)

        t = ((timesteps * 1000).to(dtype) / 1000).to(dtype)
        temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)
        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)
        mod = (_split_rows(scale1), _split_rows(gate1.tanh()), _split_rows(scale2), _split_rows(gate2.tanh()), torch.zeros_like(scale1[:1, None]))

        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})

        transformer_options["total_blocks"] = len(self.transformer_blocks)
        transformer_options["block_type"] = "single"

        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            attn_fn = block_causal_attention(segments, transformer_options)
            if ("single_block", i) in blocks_replace:

                def block_wrap(args):
                    return {"img": block(args["img"], mod, args["pe"], attn_fn, prefix_len, args["transformer_options"])}

                args = {"img": hidden_states, "vec": temb, "pe": pe, "mod": mod, "attn_fn": attn_fn, "prefix_len": prefix_len, "transformer_options": transformer_options}
                hidden_states = blocks_replace[("single_block", i)](args, {"original_block": block_wrap})["img"]
            else:
                hidden_states = block(hidden_states, mod, pe, attn_fn, prefix_len, transformer_options)
            for p in patches.get("single_block", []):
                hidden_states = p({"img": hidden_states, "x": x, "block_index": i, "transformer_options": transformer_options})["img"]

        hidden_states = self.norm_out(hidden_states[:, prefix_len:], temb[:-1])
        hidden_states = self.proj_out(hidden_states)
        return hidden_states.transpose(1, 2).reshape(B, self.out_channels, H, W)
