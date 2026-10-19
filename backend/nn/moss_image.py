# https://github.com/Comfy-Org/ComfyUI/blob/v0.42.0/comfy/ldm/moss_image/model.py

import torch
import torch.nn as nn

from backend.attention import attention_function
from backend.nn.flux import EmbedND, timestep_embedding
from backend.operations import main_stream_worker, weights_manual_cast
from backend.operations_mixed_precision import linear_input_act
from backend.quant_ops import ck


def situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def modulated_rms_norm(x, norm, scale, shift):
    weight, _, stream = weights_manual_cast(norm, x)
    with main_stream_worker(weight, None, stream):
        return ck.rms_adaln(x, weight * (1 + scale) - 1, shift, norm.eps)


def gated_residual(x, y, gate):
    return x.addcmul_(y, gate)


def attention(q, k, v, q_norm, k_norm, pe, heads, transformer_options={}):
    B, N, _ = q.shape
    q = q.view(B, N, heads, -1)
    k = k.view(B, N, heads, -1)
    q_scale, _, q_stream = weights_manual_cast(q_norm, q)
    k_scale, _, k_stream = weights_manual_cast(k_norm, k)
    with main_stream_worker(q_scale, None, q_stream), main_stream_worker(k_scale, None, k_stream):
        q, k = ck.rms_rope_split_half(q, k, pe, q_scale, k_scale, q_norm.eps)
    return attention_function(q.flatten(2), k.flatten(2), v.flatten(2), heads, transformer_options=transformer_options)


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.q_norm = nn.RMSNorm(dim // num_heads, eps=1e-6)
        self.k_norm = nn.RMSNorm(dim // num_heads, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate = nn.Linear(dim, ffn_dim)
        self.ffn_up = nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, pe, transformer_options={}):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        x = x + self.proj(attention(q, k, v, self.q_norm, self.k_norm, pe, self.num_heads, transformer_options))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextEmbedder(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, num_heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, num_heads) for _ in range(2)])

    def forward(self, x, pe, transformer_options={}):
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, pe, transformer_options)
        return x


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, fused_qkv=False):
        super().__init__()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.q_norm = nn.RMSNorm(dim // num_heads, eps=1e-6)
        self.k_norm = nn.RMSNorm(dim // num_heads, eps=1e-6)
        names = ("qkv", "o") if fused_qkv else ("q", "k", "v", "o")
        self.proj = nn.ModuleDict({name: nn.Linear(dim, 3 * dim if name == "qkv" else dim, bias=False) for name in names})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, mod, pe, transformer_options={}):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
        y = modulated_rms_norm(x, self.norm1, scale_a, shift_a)
        if self.fused_qkv:
            q, k, v = self.proj["qkv"](y).chunk(3, dim=-1)
        else:
            q, k, v = self.proj["q"](y), self.proj["k"](y), self.proj["v"](y)
        out = attention(q, k, v, self.q_norm, self.k_norm, pe, self.num_heads, transformer_options)
        x = gated_residual(x, self.proj["o"](out), gate_a)
        y = modulated_rms_norm(x, self.norm2, scale_f, shift_f)
        return gated_residual(x, linear_input_act(self.proj["ffn_down"], self.proj["ffn_up_gate"](y), "situ"), gate_f)


class MOSSImage(nn.Module):
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, latent_channels=128, fused_qkv=False):
        super().__init__()
        self.latent_channels = latent_channels
        head_dim = dim // num_heads
        self.pe_embedder = EmbedND(dim=head_dim, theta=10000, axes_dim=[head_dim // 4] * 4)
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextEmbedder(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, ffn_dim, num_heads, fused_qkv=fused_qkv) for _ in range(num_layers)])
        self.final_norm_video = nn.RMSNorm(dim, eps=1e-6)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def rope(self, h, w, text_len, device):
        ids = torch.zeros(h * w + text_len, 4, device=device, dtype=torch.float32)
        ids[: h * w, 1] = text_len
        ids[: h * w, 2] = (torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2).repeat_interleave(w)
        ids[: h * w, 3] = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2).repeat(h)
        ids[h * w :, 1] = torch.arange(text_len, device=device, dtype=torch.float32)
        return self.pe_embedder(ids.unsqueeze(0)).transpose(1, 2).contiguous()

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        B, _, H, W = x.shape
        dtype, device = x.dtype, x.device
        transformer_options = transformer_options.copy()
        pe = self.rope(H, W, context.shape[1], device)

        vec = self.time_embedding(timestep_embedding(timestep, 256).to(dtype))
        mod = self.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
        head_shift, head_scale = self.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)

        txt = self.text_embedder(context, pe[:, H * W :], transformer_options)
        x = torch.cat((self.video_embedder(x.flatten(2).transpose(1, 2)), txt), dim=1)
        del txt

        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        transformer_options["total_blocks"] = len(self.layers)
        transformer_options["block_type"] = "single"
        for i, block in enumerate(self.layers):
            transformer_options["block_index"] = i
            if ("single_block", i) in blocks_replace:

                def block_wrap(args):
                    return {"img": block(args["img"], args["mod"], args["pe"], args["transformer_options"])}

                x = blocks_replace[("single_block", i)]({"img": x, "vec": vec, "mod": mod, "pe": pe, "transformer_options": transformer_options}, {"original_block": block_wrap})["img"]
            else:
                x = block(x, mod, pe, transformer_options)

        out = self.final_linear_video(modulated_rms_norm(x[:, : H * W], self.final_norm_video, head_scale, head_shift))
        return out.transpose(1, 2).reshape(B, self.latent_channels, H, W)
