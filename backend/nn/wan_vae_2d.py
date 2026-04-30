# https://github.com/Anzhc/anzhc-qwen2d-comfyui/blob/main/qwen2d_arch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config

from backend.attention import attention_function_vae
from backend.nn._vae import ProcessLatent


class QwenImageRMSNorm2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones((dim, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma.to(x)


class QwenImageResample2D(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
        else:
            self.resample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resample(x)


class QwenImageResidualBlock2D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = QwenImageRMSNorm2D(in_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMSNorm2D(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_shortcut(x)
        x = F.silu(self.norm1(x))
        x = self.conv1(x)
        x = F.silu(self.norm2(x))
        x = self.dropout(x)
        x = self.conv2(x)
        return x + h


class QwenImageAttentionBlock2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = QwenImageRMSNorm2D(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        x = attention_function_vae(q, k, v)
        x = self.proj(x)
        return x + identity


class QwenImageMidBlock2D(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, num_layers: int = 1):
        super().__init__()
        resnets = [QwenImageResidualBlock2D(dim, dim, dropout)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImageAttentionBlock2D(dim))
            resnets.append(QwenImageResidualBlock2D(dim, dim, dropout))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x)
        return x


class QwenImageUpBlock2D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, dropout: float = 0.0, upsample_mode: str = None):
        super().__init__()
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImageResidualBlock2D(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([QwenImageResample2D(out_dim, mode=upsample_mode)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class QwenImageEncoder2D(nn.Module):
    def __init__(self, dim=96, z_dim=32, input_channels=3, dim_mult=(1, 2, 4, 4), num_res_blocks=2, attn_scales=[], dropout=0.0):
        super().__init__()
        self.dim_mult = list(dim_mult)
        self.attn_scales = list(attn_scales)

        dims = [dim * multiplier for multiplier in [1] + self.dim_mult]
        scale = 1.0

        self.conv_in = nn.Conv2d(input_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([])
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.down_blocks.append(QwenImageResidualBlock2D(in_dim, out_dim, dropout))
                if scale in self.attn_scales:
                    self.down_blocks.append(QwenImageAttentionBlock2D(out_dim))
                in_dim = out_dim
            if index != len(self.dim_mult) - 1:
                self.down_blocks.append(QwenImageResample2D(out_dim, mode="downsample2d"))
                scale /= 2.0

        self.mid_block = QwenImageMidBlock2D(out_dim, dropout, num_layers=1)
        self.norm_out = QwenImageRMSNorm2D(out_dim)
        self.conv_out = nn.Conv2d(out_dim, z_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.mid_block(x)
        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


class QwenImageDecoder2D(nn.Module):
    def __init__(self, dim=96, z_dim=16, output_channels=3, dim_mult=(1, 2, 4, 4), num_res_blocks=2, dropout=0.0):
        super().__init__()
        self.dim_mult = list(dim_mult)

        dims = [dim * multiplier for multiplier in [self.dim_mult[-1]] + self.dim_mult[::-1]]

        self.conv_in = nn.Conv2d(z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImageMidBlock2D(dims[0], dropout, num_layers=1)

        self.up_blocks = nn.ModuleList([])
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if index > 0:
                in_dim = in_dim // 2
            upsample_mode = "upsample2d" if index != len(self.dim_mult) - 1 else None
            self.up_blocks.append(
                QwenImageUpBlock2D(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                )
            )

        self.norm_out = QwenImageRMSNorm2D(out_dim)
        self.conv_out = nn.Conv2d(out_dim, output_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


class Qwen2DVAE(nn.Module, ProcessLatent, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        base_dim=96,
        z_dim=16,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=[],
        image_channels=3,
        dropout=0.0,
    ):
        super().__init__()
        self.base_dim = base_dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)

        self.encoder = QwenImageEncoder2D(
            dim=base_dim,
            z_dim=z_dim * 2,
            input_channels=image_channels,
            dim_mult=self.dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=self.attn_scales,
            dropout=dropout,
        )
        self.quant_conv = nn.Conv2d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = nn.Conv2d(z_dim, z_dim, 1)
        self.decoder = QwenImageDecoder2D(
            dim=base_dim,
            z_dim=z_dim,
            output_channels=image_channels,
            dim_mult=self.dim_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

    def _flatten_frames(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if x.ndim == 4:
            return x, None
        if x.ndim != 5:
            raise ValueError(f"Unsupported Qwen2D VAE input shape: {tuple(x.shape)}")

        batch, channels, frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        return x, (batch, frames)

    def _restore_frames(self, x: torch.Tensor, frame_info: tuple[int, int]) -> torch.Tensor:
        if frame_info is None:
            return x

        batch, frames = frame_info
        channels, height, width = x.shape[1:]
        return x.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x, frame_info = self._flatten_frames(x)
        moments = self.quant_conv(self.encoder(x))
        mu, _ = moments.chunk(2, dim=1)
        return self._restore_frames(mu, frame_info)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z, frame_info = self._flatten_frames(z)
        out = self.decoder(self.post_quant_conv(z))
        out = torch.clamp(out, min=-1.0, max=1.0)
        return self._restore_frames(out, frame_info)
