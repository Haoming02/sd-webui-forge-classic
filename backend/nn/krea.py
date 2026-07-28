# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/ldm/krea2/model.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from backend.attention import attention_function
from backend.memory_management import cast_to
from backend.nn.flux import EmbedND, timestep_embedding
from backend.quant_ops import ck
from backend.utils import pad_to_patch_size


class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.empty(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        weight = cast_to(self.scale, dtype=torch.float32, device=x.device) + 1.0
        return F.rms_norm(x.float(), (x.shape[-1],), weight=weight, eps=self.eps).to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q, k):
        return self.qnorm(q), self.knorm(k)


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()

        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)).mul_(self.up(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: Optional[int] = None, bias: bool = False):
        super().__init__()

        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(self, x, freqs=None, mask=None, transformer_options={}):
        q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = ck.apply_rope(q, k, freqs)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = attention_function(q, k, v, self.heads, mask=mask, skip_reshape=True, transformer_options=transformer_options)
        return self.wo(out * F.sigmoid(gate))


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.lin = nn.Parameter(torch.empty(2, dim))

    def forward(self, vec):
        out = vec + cast_to(self.lin, dtype=vec.dtype, device=vec.device).unsqueeze(0)
        scale, shift = out.chunk(2, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.lin = nn.Parameter(torch.empty(6 * dim))

    def forward(self, vec):
        out = vec + cast_to(self.lin, dtype=vec.dtype, device=vec.device)
        return out.chunk(6, dim=-1)


class TextFusionBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()

        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, mask=None, transformer_options={}):
        x.add_(self.attn(self.prenorm(x), mask=mask, transformer_options=transformer_options))
        x.add_(self.mlp(self.postnorm(x)))
        return x


class TextFusionTransformer(nn.Module):
    def __init__(self, num_txt_layers, txt_dim, heads, multiplier, bias=False, kvheads=None):
        super().__init__()

        self.layerwise_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])

    def forward(self, x, mask=None, transformer_options={}):
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None, transformer_options=transformer_options)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask, transformer_options=transformer_options)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()

        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, vec, freqs, mask=None, transformer_options={}):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x.addcmul_(pregate, self.attn(torch.addcmul(preshift, 1 + prescale, self.prenorm(x)), freqs, mask, transformer_options=transformer_options))
        x.addcmul_(postgate, self.mlp(torch.addcmul(postshift, 1 + postscale, self.postnorm(x))))
        return x


class LastLayer(nn.Module):
    def __init__(self, features, patch, channels):
        super().__init__()

        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x, tvec):
        scale, shift = self.modulation(tvec)
        x = torch.addcmul(shift, 1 + scale, self.norm(x))
        return self.linear(x)


class SingleStreamDiT(nn.Module):
    def __init__(
        self,
        features=6144,
        tdim=256,
        txtdim=2560,
        heads=48,
        kvheads=12,
        multiplier=4,
        layers=28,
        patch=2,
        channels=16,
        bias=False,
        theta=1e3,
        txtlayers=12,
        txtheads=20,
        txtkvheads=20,
        **kwargs,
    ):
        super().__init__()

        self.patch = patch
        self.channels = channels
        self.tdim = tdim
        self.heads = heads
        self.txtdim = txtdim
        self.txtlayers = txtlayers

        headdim = features // heads
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        assert sum(axes) == headdim, f"axes {axes} sum != headdim {headdim}"
        self.pe_embedder = EmbedND(dim=headdim, theta=int(theta), axes_dim=axes)

        self.first = nn.Linear(channels * patch**2, features, bias=True)
        self.blocks = nn.ModuleList([SingleStreamBlock(features, heads, multiplier, bias, kvheads) for _ in range(layers)])
        self.tmlp = nn.Sequential(
            nn.Linear(tdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.txtfusion = TextFusionTransformer(txtlayers, txtdim, txtheads, multiplier, bias, txtkvheads)
        self.txtmlp = nn.Sequential(
            RMSNorm(txtdim),
            nn.Linear(txtdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.last = LastLayer(features, patch, channels)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features * 6),
        )

    def forward(self, x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs):
        temporal = x.ndim == 5
        if temporal:
            b5, c5, t5, h5, w5 = x.shape
            x = x.reshape(b5 * t5, c5, h5, w5)
        bs, c, H_orig, W_orig = x.shape
        patch = self.patch

        x = pad_to_patch_size(x, (patch, patch))
        H, W = x.shape[-2], x.shape[-1]
        h_, w_ = H // patch, W // patch

        context = self._unpack_context(context.squeeze(1))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
        img = self.first(img)

        t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
        tvec = self.tproj(t)

        context = self.txtfusion(context, mask=None, transformer_options=transformer_options)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)

        device = combined.device
        txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
        imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
        imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
        imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
        imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
        pos = torch.cat((txtpos, imgpos), dim=1)

        freqs = self.pe_embedder(pos)

        for block in self.blocks:
            combined = block(combined, tvec, freqs, None, transformer_options=transformer_options)

        final = self.last(combined, t)
        out = final[:, txtlen : txtlen + imglen, :]
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=patch, pw=patch, c=self.channels)
        out = out[:, :, :H_orig, :W_orig]
        if temporal:
            out = out.reshape(b5, t5, self.channels, H_orig, W_orig).movedim(1, 2)
        return out

    def _unpack_context(self, context):
        b, seq, fused = context.shape
        if fused != self.txtlayers * self.txtdim:
            raise ValueError(f"Krea2 expects conditioning with {self.txtlayers}x{self.txtdim}={self.txtlayers * self.txtdim} " f"features (a {self.txtlayers}-layer Qwen3-VL stack) but got {fused}. " f"Load the text encoder with CLIPLoader type 'krea2'.")
        return context.reshape(b, seq, self.txtlayers, self.txtdim)
