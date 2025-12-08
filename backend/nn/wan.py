# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/ldm/wan/model.py
# Copyright Wan 2024-2025
# Reference: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py

import math

import torch
import torch.nn as nn
from einops import repeat

from backend import args
from backend.attention import attention_function as optimized_attention
from backend.memory_management import cast_to_device
from backend.nn.flux import EmbedND, apply_rope1
from backend.utils import pad_to_patch_size


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


class WanSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn_q(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            return apply_rope1(q, freqs)

        def qkv_fn_k(x):
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            return apply_rope1(k, freqs)

        q = qkv_fn_q(x)
        k = qkv_fn_k(x)

        x = optimized_attention(
            q.view(b, s, n * d),
            k.view(b, s, n * d),
            self.v(x).view(b, s, n * d),
            heads=self.num_heads,
        )

        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)

        # compute attention
        x = optimized_attention(q, k, v, heads=self.num_heads)

        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_img_len):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        context_img = context[:, :context_img_len]
        context = context[:, context_img_len:]

        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        k_img = self.norm_k_img(self.k_img(context_img))
        v_img = self.v_img(context_img)
        img_x = optimized_attention(q, k_img, v_img, heads=self.num_heads)
        # compute attention
        x = optimized_attention(q, k, v, heads=self.num_heads)

        # output
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


def repeat_e(e, x):
    repeats = 1
    if e.shape[1] > 1:
        repeats = x.shape[1] // e.shape[1]
    if repeats == 1:
        return e
    return torch.repeat_interleave(e, repeats, dim=1)


class WanAttentionBlock(nn.Module):

    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, window_size=(-1, -1), qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = nn.LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.empty(1, 6, dim))

    def forward(
        self,
        x,
        e,
        freqs,
        context,
        context_img_len=257,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32

        if e.ndim < 4:
            e = (cast_to_device(self.modulation, dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
        else:
            e = (cast_to_device(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0) + e).unbind(2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(self.norm1(x) * (1 + repeat_e(e[1], x)) + repeat_e(e[0], x), freqs)

        x = x + y * repeat_e(e[2], x)

        # cross-attention & ffn
        x = x + self.cross_attn(self.norm3(x), context, context_img_len=context_img_len)
        y = self.ffn(self.norm2(x) * (1 + repeat_e(e[4], x)) + repeat_e(e[3], x))
        x = x + y * repeat_e(e[5], x)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.empty(1, 2, dim))

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # assert e.dtype == torch.float32
        if e.ndim < 3:
            e = (cast_to_device(self.modulation, dtype=x.dtype, device=x.device) + e.unsqueeze(1)).chunk(2, dim=1)
        else:
            e = (cast_to_device(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0) + e.unsqueeze(2)).unbind(2)

        x = self.head(self.norm(x) * (1 + repeat_e(e[1], x)) + repeat_e(e[0], x))
        return x


class MLPProj(nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_embed_token_number=None):
        super().__init__()

        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))

        if flf_pos_embed_token_number is not None:
            self.emb_pos = nn.Parameter(torch.empty((1, flf_pos_embed_token_number, in_dim)))
        else:
            self.emb_pos = None

    def forward(self, image_embeds):
        if self.emb_pos is not None:
            image_embeds = image_embeds[:, : self.emb_pos.shape[1]] + cast_to_device(self.emb_pos[:, : image_embeds.shape[1]], dtype=image_embeds.dtype, device=image_embeds.device)

        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(nn.Module):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        flf_pos_embed_token_number=None,
    ):

        super().__init__()

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList([WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps) for _ in range(num_layers)])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        d = dim // num_heads
        self.rope_embedder = EmbedND(dim=d, theta=10000.0, axes_dim=[d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)])

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim, flf_pos_embed_token_number=flf_pos_embed_token_number)
        else:
            self.img_emb = None

    def forward_orig(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):

        # embeddings
        x = self.patch_embedding(x).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:

                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                    return out

                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

    def forward(self, x, timestep, context, clip_fea=None, time_dim_concat=None, transformer_options={}, **kwargs):
        bs, c, t, h, w = x.shape

        if c < self.in_dim:
            assert "concat_latent" in args.dynamic_args
            r = args.dynamic_args["concat_latent"].to(x)
            if x.shape[0] == 2:  # batch_cond_uncond
                r = torch.cat((r, r), dim=0)
            x = torch.cat((x, r), dim=1)
            bs, c, t, h, w = x.shape

        x = pad_to_patch_size(x, self.patch_size)

        patch_size = self.patch_size
        t_len = (t + (patch_size[0] // 2)) // patch_size[0]
        h_len = (h + (patch_size[1] // 2)) // patch_size[1]
        w_len = (w + (patch_size[2] // 2)) // patch_size[2]

        if time_dim_concat is not None:
            time_dim_concat = pad_to_patch_size(time_dim_concat, self.patch_size)
            x = torch.cat([x, time_dim_concat], dim=2)
            t_len = (x.shape[2] + (patch_size[0] // 2)) // patch_size[0]

        img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
        img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

        freqs = self.rope_embedder(img_ids).movedim(1, 2)
        return self.forward_orig(x, timestep, context, clip_fea=clip_fea, freqs=freqs, transformer_options=transformer_options, **kwargs)[:, :, :t, :h, :w]

    def unpatchify(self, x, grid_sizes):
        """Reconstruct video tensors from patch embeddings"""

        c = self.out_dim
        u = x
        b = u.shape[0]
        u = u[:, : math.prod(grid_sizes)].view(b, *grid_sizes, *self.patch_size, c)
        u = torch.einsum("bfhwpqrc->bcfphqwr", u)
        u = u.reshape(b, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return u
