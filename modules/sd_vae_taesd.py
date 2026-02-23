# Tiny AutoEncoder for Stable Diffusion
# https://github.com/madebyollin/taesd/blob/main/taesd.py

# Tiny AutoEncoder for Hunyuan Video
# https://github.com/madebyollin/taehv/blob/main/taehv.py

# reference:
# - https://github.com/Comfy-Org/ComfyUI/pull/12043
# - https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/taesd/taehv.py

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules_forge.packages.huggingface_guess.latent import LatentFormat

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.state_dict import load_state_dict
from backend.utils import load_torch_file
from modules import devices, paths_internal, shared

URL: str = "https://github.com/madebyollin/taesd/raw/main/"
URL_V: str = "https://github.com/madebyollin/taehv/raw/main/"
sd_vae_taesd_models: dict[str, torch.nn.Module] = {}


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    @staticmethod
    def forward(x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out, use_midblock_gn=False):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
        self.pool = None

        if use_midblock_gn:
            conv1x1 = lambda n_in, n_out: nn.Conv2d(n_in, n_out, 1, bias=False)
            n_gn = n_in * 4
            self.pool = nn.Sequential(conv1x1(n_in, n_gn), nn.GroupNorm(4, n_gn), nn.ReLU(inplace=True), conv1x1(n_gn, n_in))

    def forward(self, x):
        if self.pool is not None:
            x = x + self.pool(x)

        return self.fuse(self.conv(x) + self.skip(x))


def decoder(latent_channels=4, use_midblock_gn=False):
    mb_kw = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        *(Clamp(), conv(latent_channels, 64), nn.ReLU()),
        *(Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), nn.Upsample(scale_factor=2), conv(64, 64, bias=False)),
        *(Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False)),
        *(Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False)),
        *(Block(64, 64), conv(64, 3)),
    )


def encoder(latent_channels=4, use_midblock_gn=False):
    mb_kw = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        *(conv(3, 64), Block(64, 64)),
        *(conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64)),
        *(conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64)),
        *(conv(64, 64, stride=2, bias=False), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw)),
        conv(64, latent_channels),
    )


class TAESDDecoder(nn.Module):

    def __init__(self, decoder_path: os.PathLike, latent_channels: int):
        super().__init__()
        self.latent_channels = 32 if latent_channels == 128 else latent_channels
        self.decoder = decoder(self.latent_channels, use_midblock_gn=(self.latent_channels == 32))
        load_state_dict(self.decoder, load_torch_file(decoder_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.latent_channels == 32:
            x = x.reshape(x.shape[0], self.latent_channels, 2, 2, x.shape[-2], x.shape[-1]).permute(0, 1, 4, 2, 5, 3).reshape(x.shape[0], self.latent_channels, x.shape[-2] * 2, x.shape[-1] * 2)
        return self.decoder(x)


class TAESDEncoder(nn.Module):

    def __init__(self, encoder_path: os.PathLike, latent_channels: int):
        super().__init__()
        self.latent_channels = 32 if latent_channels == 128 else latent_channels
        self.encoder = encoder(self.latent_channels, use_midblock_gn=(self.latent_channels == 32))
        load_state_dict(self.encoder, load_torch_file(encoder_path))

    def forward(self, x_sample: torch.Tensor) -> torch.Tensor:
        if self.latent_channels == 32:
            x_sample = x_sample.reshape(x_sample.shape[0], self.latent_channels, x_sample.shape[-2] // 2, 2, x_sample.shape[-1] // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(x_sample.shape[0], self.latent_channels * 4, x_sample.shape[-2] // 2, x_sample.shape[-1] // 2)
        return self.encoder(x_sample)


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out, act_func):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), act_func, conv(n_out, n_out), act_func, conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = act_func

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


class TAEHV(nn.Module):
    def __init__(self, latent_channels, decoder_time_upscale=(False, True, True), decoder_space_upscale=(True, True, True)):
        super().__init__()
        self.image_channels = 3
        self.latent_channels = latent_channels

        if self.latent_channels == 16:  # Wan 2.1
            self.patch_size = 1
            act_func = nn.ReLU(inplace=True)
        else:  # HunyuanVideo 1.5
            self.patch_size = 2
            act_func = nn.LeakyReLU(0.2, inplace=True)

        n_f = [256, 128, 64, 64]

        self.decoder = nn.Sequential(
            *(Clamp(), conv(self.latent_channels, n_f[0]), act_func),
            *(MemBlock(n_f[0], n_f[0], act_func), MemBlock(n_f[0], n_f[0], act_func), MemBlock(n_f[0], n_f[0], act_func), nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1), TGrow(n_f[0], 2 if decoder_time_upscale[0] else 1), conv(n_f[0], n_f[1], bias=False)),
            *(MemBlock(n_f[1], n_f[1], act_func), MemBlock(n_f[1], n_f[1], act_func), MemBlock(n_f[1], n_f[1], act_func), nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[1] else 1), conv(n_f[1], n_f[2], bias=False)),
            *(MemBlock(n_f[2], n_f[2], act_func), MemBlock(n_f[2], n_f[2], act_func), MemBlock(n_f[2], n_f[2], act_func), nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[2] else 1), conv(n_f[2], n_f[3], bias=False)),
            *(act_func, conv(n_f[3], self.image_channels * self.patch_size**2)),
        )

        self.t_upscale = 2 ** sum(t.stride == 2 for t in self.decoder if isinstance(t, TGrow))
        self.frames_to_trim = self.t_upscale - 1

    @staticmethod
    def apply_model_with_memblocks(model: nn.Sequential, x: torch.Tensor):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        for b in model:
            if isinstance(b, MemBlock):
                BT, C, H, W = x.shape
                T = BT // B
                _x = x.reshape(B, T, C, H, W)
                mem = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(x.shape)
                x = b(x, mem)
            else:
                x = b(x)

        BT, C, H, W = x.shape
        T = BT // B
        return x.view(B, T, C, H, W)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.movedim(2, 1)
        x = self.apply_model_with_memblocks(self.decoder, x)
        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)
        return x[:, self.frames_to_trim :]


class TAEHVDecoder(nn.Module):

    def __init__(self, decoder_path: os.PathLike, latent_channels: int):
        super().__init__()
        self.latent_channels = latent_channels
        self.decoder = TAEHV(self.latent_channels)
        load_state_dict(self.decoder, load_torch_file(decoder_path), ignore_start="encoder")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.decoder.decode(x)
        return z.squeeze(1)


def download_model(model_path: os.PathLike, model_url: str):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f'Downloading TAESD Model to: "{model_path}"...')
        torch.hub.download_url_to_file(model_url, model_path)


def decoder_model():
    latent_format: "LatentFormat" = shared.sd_model.model_config.latent_format
    model_name: str = latent_format.taesd_decoder_name
    if model_name is None:
        return None
    else:
        _video = model_name in ["taew2_1"]
        model_name = model_name + ".pth"

    loaded_model = sd_vae_taesd_models.get(model_name)

    if loaded_model is None:
        model_path = os.path.join(paths_internal.models_path, "VAE-taesd", model_name)
        download_model(model_path, (URL_V if _video else URL) + model_name)

        if not os.path.exists(model_path):
            return None

        loaded_model = (TAEHVDecoder if _video else TAESDDecoder)(model_path, latent_format.latent_channels)
        loaded_model.eval()
        loaded_model.to(devices.device, devices.dtype)
        sd_vae_taesd_models[model_name] = loaded_model

    return loaded_model


def encoder_model():
    latent_format: "LatentFormat" = shared.sd_model.model_config.latent_format
    model_name: str = latent_format.taesd_decoder_name
    if model_name is None:
        return None
    elif model_name in ["taew2_1"]:
        return None
    else:
        model_name = model_name.replace("decoder", "encoder") + ".pth"

    loaded_model = sd_vae_taesd_models.get(model_name)

    if loaded_model is None:
        model_path = os.path.join(paths_internal.models_path, "VAE-taesd", model_name)
        download_model(model_path, URL + model_name)

        if not os.path.exists(model_path):
            return None

        loaded_model = TAESDEncoder(model_path, latent_format.latent_channels)
        loaded_model.eval()
        loaded_model.to(devices.device, devices.dtype)
        sd_vae_taesd_models[model_name] = loaded_model

    return loaded_model
