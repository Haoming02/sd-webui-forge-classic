import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from einops import rearrange

from backend import memory_management
from backend.attention import attention_function_vae
from backend.nn._vae import ProcessLatent


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def mode(self):
        return self.mean


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    @torch.inference_mode()
    def forward(self, x):
        try:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        except Exception as e:
            b, c, h, w = x.shape
            out = torch.empty((b, c, h * 2, w * 2), dtype=x.dtype, layout=x.layout, device=x.device)
            split = 8
            l = out.shape[1] // split
            for i in range(0, out.shape[1], l):
                out[:, i : i + l] = F.interpolate(x[:, i : i + l].to(torch.float32), scale_factor=2.0, mode="nearest").to(x.dtype)
            del x
            x = out

        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    @torch.inference_mode()
    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.swish = nn.SiLU(inplace=True)
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    @torch.inference_mode()
    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.swish(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(self.swish(temb))[:, :, None, None]
        h = self.norm2(h)
        h = self.swish(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    @torch.inference_mode()
    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        h_ = attention_function_vae(q, k, v)
        h_ = self.proj_out(h_)
        return x + h_


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla", **kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    @torch.inference_mode()
    def forward(self, x):
        temb = None
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False, **kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    @torch.inference_mode()
    def forward(self, z, **kwargs):
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                _h = self.up[i_level].block[i_block](h, temb, **kwargs)
                del h
                h = self.up[i_level].attn[i_block](_h, **kwargs) if len(self.up[i_level].attn) > 0 else _h
            if i_level != 0:
                _h = self.up[i_level].upsample(h)
                del h
                h = _h

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class IntegratedAutoencoderKL(nn.Module, ProcessLatent, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(self, in_channels=3, out_channels=3, block_out_channels=(64,), layers_per_block=1, latent_channels=4, use_quant_conv=True, use_post_quant_conv=True, **kwargs):
        del kwargs
        super().__init__()

        ch = block_out_channels[0]
        ch_mult = [x // ch for x in block_out_channels]
        self.encoder = Encoder(double_z=True, z_channels=latent_channels, resolution=256, in_channels=in_channels, out_ch=out_channels, ch=ch, ch_mult=ch_mult, num_res_blocks=layers_per_block, attn_resolutions=[], dropout=0.0)
        self.decoder = Decoder(double_z=True, z_channels=latent_channels, resolution=256, in_channels=in_channels, out_ch=out_channels, ch=ch, ch_mult=ch_mult, num_res_blocks=layers_per_block, attn_resolutions=[], dropout=0.0)
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None
        self.embed_dim = latent_channels

    def encode(self, x):
        z = self.encoder(x)

        if self.quant_conv is not None:
            z = self.quant_conv(z)

        posterior = DiagonalGaussianDistribution(z)
        return posterior.sample()

    def decode(self, z):
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        x = self.decoder(z)
        return x


class AutoencoderKLFlux2(IntegratedAutoencoderKL):
    config_name = "config.json"

    @register_to_config
    def __init__(self, in_channels=3, out_channels=3, block_out_channels=(64,), layers_per_block=1, latent_channels=4, use_quant_conv=True, use_post_quant_conv=True, *, mugen: bool = False, ech: int = None, dch: int = None, **kwargs):
        del kwargs
        super().__init__()

        ch = block_out_channels[0]
        ch_mult = [x // ch for x in block_out_channels]
        self.encoder = Encoder(double_z=True, z_channels=latent_channels, resolution=256, in_channels=in_channels, out_ch=out_channels, ch=ech or ch, ch_mult=ch_mult, num_res_blocks=layers_per_block, attn_resolutions=[], dropout=0.0)
        self.decoder = Decoder(double_z=True, z_channels=latent_channels, resolution=256, in_channels=in_channels, out_ch=out_channels, ch=dch or ch, ch_mult=ch_mult, num_res_blocks=layers_per_block, attn_resolutions=[], dropout=0.0)
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None
        self.embed_dim = latent_channels

        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = nn.BatchNorm2d(
            math.prod(self.ps) * latent_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )
        self.bn.eval()

        self.mugen = mugen  # 32 <-> 128

    def encode(self, x):
        z = super().encode(x)

        z = rearrange(
            z,
            "... c (i pi) (j pj)  -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )

        z = F.batch_norm(
            z,
            memory_management.cast_to(self.bn.running_mean, dtype=z.dtype, device=z.device),
            memory_management.cast_to(self.bn.running_var, dtype=z.dtype, device=z.device),
            momentum=self.bn_momentum,
            eps=self.bn_eps,
        )

        z = self.postprocess_encode(z)
        return z

    def decode(self, z):
        z = self.preprocess_decode(z)
        s = torch.sqrt(memory_management.cast_to(self.bn.running_var.view(1, -1, 1, 1), dtype=z.dtype, device=z.device) + self.bn_eps)
        m = memory_management.cast_to(self.bn.running_mean.view(1, -1, 1, 1), dtype=z.dtype, device=z.device)
        z = z * s + m
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )

        return super().decode(z)

    def preprocess_decode(self, latent: torch.Tensor):
        packed_channels: int = latent.size(1)
        latent_channels: int = 128
        scale_factor: int = 2

        if self.mugen:
            h = latent.shape[-2]
            w = latent.shape[-1]
            if h % scale_factor != 0 or w % scale_factor != 0:
                pad_h = (scale_factor - (h % scale_factor)) % scale_factor
                pad_w = (scale_factor - (w % scale_factor)) % scale_factor
                latent = F.pad(latent, (0, pad_w, 0, pad_h))
                h = latent.shape[-2]
                w = latent.shape[-1]
            latent = latent.reshape(latent.shape[0], packed_channels, h // scale_factor, scale_factor, w // scale_factor, scale_factor)
            latent = latent.permute(0, 1, 3, 5, 2, 4).reshape(latent.shape[0], latent_channels, h // scale_factor, w // scale_factor)

        return latent

    def postprocess_encode(self, latent: torch.Tensor):
        packed_channels: int = 32
        scale_factor: int = 2

        if self.mugen:
            h = latent.shape[-2]
            w = latent.shape[-1]
            latent = latent.reshape(latent.shape[0], packed_channels, scale_factor, scale_factor, h, w)
            latent = latent.permute(0, 1, 4, 2, 5, 3).reshape(latent.shape[0], packed_channels, h * scale_factor, w * scale_factor)

        return latent
