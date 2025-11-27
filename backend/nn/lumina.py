# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.75/comfy/ldm/lumina/model.py
# Reference: https://github.com/Alpha-VLLM/Lumina-Image-2.0

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from backend.memory_management import xformers_enabled

if xformers_enabled():
    from backend.attention import attention_xformers as attention_function
else:
    from backend.attention import attention_pytorch as attention_function

from backend.args import dynamic_args
from backend.nn.flux import EmbedND
from backend.utils import pad_to_patch_size


def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256, output_size=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size or hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t, dtype, **kwargs):
        t_freq = timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class JointAttention(nn.Module):

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, qk_norm: bool, out_bias: bool = False):
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, (n_heads + self.n_kv_heads + self.n_kv_heads) * self.head_dim, bias=False)
        self.out = nn.Linear(n_heads * self.head_dim, dim, bias=out_bias)

        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True)
            self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    @staticmethod
    def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        t_ = x_in.reshape(*x_in.shape[:-1], -1, 1, 2)
        t_out = freqs_cis[..., 0] * t_[..., 0] + freqs_cis[..., 1] * t_[..., 1]
        return t_out.reshape(*x_in.shape)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, freqs_cis: torch.Tensor, transformer_options={}) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = torch.split(
            self.qkv(x),
            [
                self.n_local_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = JointAttention.apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = JointAttention.apply_rotary_emb(xk, freqs_cis=freqs_cis)

        n_rep = self.n_local_heads // self.n_local_kv_heads
        if n_rep >= 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        output = attention_function(xq.movedim(1, 2), xk.movedim(1, 2), xv.movedim(1, 2), self.n_local_heads, x_mask, skip_reshape=True, transformer_options=transformer_options)

        return self.out(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: float):
        super().__init__()
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class JointTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, n_kv_heads: int, multiple_of: int, ffn_dim_multiplier: float, norm_eps: float, qk_norm: bool, modulation=True, z_image_modulation=False, attn_out_bias=False):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = JointAttention(dim, n_heads, n_kv_heads, qk_norm, out_bias=attn_out_bias)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)

        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)

        self.modulation = modulation
        if modulation:
            if z_image_modulation:
                self.adaLN_modulation = nn.Sequential(
                    nn.Linear(min(dim, 256), 4 * dim, bias=True),
                )
            else:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(min(dim, 1024), 4 * dim, bias=True),
                )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, freqs_cis: torch.Tensor, adaln_input: torch.Tensor = None, transformer_options={}):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                self.attention(
                    modulate(self.attention_norm1(x), scale_msa),
                    x_mask,
                    freqs_cis,
                    transformer_options=transformer_options,
                )
            )
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                self.feed_forward(
                    modulate(self.ffn_norm1(x), scale_mlp),
                )
            )
        else:
            assert adaln_input is None
            x = x + self.attention_norm2(
                self.attention(
                    self.attention_norm1(x),
                    x_mask,
                    freqs_cis,
                    transformer_options=transformer_options,
                )
            )
            x = x + self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x),
                )
            )
        return x


class FinalLayer(nn.Module):

    def __init__(self, hidden_size, patch_size, out_channels, z_image_modulation=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

        if z_image_modulation:
            min_mod = 256
        else:
            min_mod = 1024

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, min_mod), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), scale)
        x = self.linear(x)
        return x


class NextDiT(nn.Module):

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 4096,
        n_layers: int = 32,
        n_refiner_layers: int = 2,
        n_heads: int = 32,
        n_kv_heads: int = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: float = 4.0,
        norm_eps: float = 1e-5,
        qk_norm: bool = False,
        cap_feat_dim: int = 5120,
        axes_dims: list[int] = (16, 56, 56),
        axes_lens: list[int] = (1, 512, 512),
        rope_theta: float = 10000.0,
        z_image_modulation: bool = False,
        time_scale: float = 1.0,
        pad_tokens_multiple: int = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.time_scale = time_scale
        self.pad_tokens_multiple = pad_tokens_multiple

        self.x_embedder = nn.Linear(in_features=patch_size * patch_size * in_channels, out_features=dim, bias=True)

        self.noise_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                    z_image_modulation=z_image_modulation,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.t_embedder = TimestepEmbedder(min(dim, 1024), output_size=256 if z_image_modulation else None)
        self.cap_embedder = nn.Sequential(
            nn.RMSNorm(cap_feat_dim, eps=norm_eps, elementwise_affine=True),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.layers = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    z_image_modulation=z_image_modulation,
                    attn_out_bias=False,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.norm_final = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels, z_image_modulation=z_image_modulation)

        if self.pad_tokens_multiple is not None:
            self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
            self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        assert (dim // n_heads) == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = EmbedND(dim=dim // n_heads, theta=rope_theta, axes_dim=axes_dims)
        self.dim = dim
        self.n_heads = n_heads

    def unpatchify(self, x: torch.Tensor, img_size: list[tuple[int, int]], cap_size: list[int], return_tensor=False) -> list[torch.Tensor]:
        pH = pW = self.patch_size
        imgs = []
        for i in range(x.size(0)):
            H, W = img_size[i]
            begin = cap_size[i]
            end = begin + (H // pH) * (W // pW)
            imgs.append(x[i][begin:end].view(H // pH, W // pW, pH, pW, self.out_channels).permute(4, 0, 2, 1, 3).flatten(3, 4).flatten(1, 2))

        if return_tensor:
            imgs = torch.stack(imgs, dim=0)
        return imgs

    def patchify_and_embed(self, x: list[torch.Tensor] | torch.Tensor, cap_feats: torch.Tensor, cap_mask: torch.Tensor, t: torch.Tensor, num_tokens, transformer_options={}) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], list[int], torch.Tensor]:
        bsz = len(x)
        pH = pW = self.patch_size
        device = x[0].device
        dtype = x[0].dtype

        if self.pad_tokens_multiple is not None:
            pad_extra = (-cap_feats.shape[1]) % self.pad_tokens_multiple
            cap_feats = torch.cat((cap_feats, self.cap_pad_token.to(device=cap_feats.device, dtype=cap_feats.dtype, copy=True).unsqueeze(0).repeat(cap_feats.shape[0], pad_extra, 1)), dim=1)

        cap_pos_ids = torch.zeros(bsz, cap_feats.shape[1], 3, dtype=torch.float32, device=device)
        cap_pos_ids[:, :, 0] = torch.arange(cap_feats.shape[1], dtype=torch.float32, device=device) + 1.0

        B, C, H, W = x.shape
        x = self.x_embedder(x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2))

        H_tokens, W_tokens = H // pH, W // pW
        x_pos_ids = torch.zeros((bsz, x.shape[1], 3), dtype=torch.float32, device=device)
        x_pos_ids[:, :, 0] = cap_feats.shape[1] + 1
        x_pos_ids[:, :, 1] = torch.arange(H_tokens, dtype=torch.float32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
        x_pos_ids[:, :, 2] = torch.arange(W_tokens, dtype=torch.float32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()

        if self.pad_tokens_multiple is not None:
            pad_extra = (-x.shape[1]) % self.pad_tokens_multiple
            x = torch.cat((x, self.x_pad_token.to(device=x.device, dtype=dtype, copy=True).unsqueeze(0).repeat(x.shape[0], pad_extra, 1)), dim=1)
            x_pos_ids = torch.nn.functional.pad(x_pos_ids, (0, 0, 0, pad_extra))

        freqs_cis = self.rope_embedder(torch.cat((cap_pos_ids, x_pos_ids), dim=1)).movedim(1, 2).to(dtype)

        # refine context
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, freqs_cis[:, : cap_pos_ids.shape[1]], transformer_options=transformer_options)

        padded_img_mask = None
        for layer in self.noise_refiner:
            x = layer(x, padded_img_mask, freqs_cis[:, cap_pos_ids.shape[1] :], t, transformer_options=transformer_options)

        padded_full_embed = torch.cat((cap_feats, x), dim=1)
        mask = None
        img_sizes = [(H, W)] * bsz
        l_effective_cap_len = [cap_feats.shape[1]] * bsz
        return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis

    def forward(self, x, timesteps, context, num_tokens=None, attention_mask=None, **kwargs):
        t = 1.0 - timesteps
        cap_feats = context
        cap_mask = attention_mask
        bs, c, h, w = x.shape
        x = pad_to_patch_size(x, (self.patch_size, self.patch_size))

        t = self.t_embedder(t * self.time_scale, dtype=x.dtype)  # (N, D)
        adaln_input = t

        cap_feats = self.cap_embedder(cap_feats)  # (N, L, D)  # todo check if able to batchify w.o. redundant compute

        transformer_options = kwargs.get("transformer_options", {})

        x_is_tensor = isinstance(x, torch.Tensor)
        x, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(x, cap_feats, cap_mask, t, num_tokens, transformer_options=transformer_options)
        freqs_cis = freqs_cis.to(x.device)

        for layer in self.layers:
            x = layer(x, mask, freqs_cis, adaln_input, transformer_options=transformer_options)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x, img_size, cap_size, return_tensor=x_is_tensor)[:, :, :h, :w]

        return -x
