# Tiny AutoEncoder for Stable Diffusion
# https://github.com/madebyollin/taesd/blob/main/taesd.py
# https://github.com/Comfy-Org/ComfyUI/pull/12043

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules_forge.packages.huggingface_guess.latent import LatentFormat

import torch
import torch.nn as nn

from backend.state_dict import load_state_dict
from backend.utils import load_torch_file
from modules import devices, paths_internal, shared

URL: str = "https://github.com/madebyollin/taesd/raw/main/"
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

    def __init__(self, decoder_path: os.PathLike):
        super().__init__()

        if "f2" in decoder_path:
            self.latent_channels = 32
        elif "f1" in decoder_path:
            self.latent_channels = 16
        else:
            self.latent_channels = 4

        self.decoder = decoder(self.latent_channels, use_midblock_gn=(self.latent_channels == 32))
        load_state_dict(self.decoder, load_torch_file(decoder_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.latent_channels == 32:
            x = x.reshape(x.shape[0], self.latent_channels, 2, 2, x.shape[-2], x.shape[-1]).permute(0, 1, 4, 2, 5, 3).reshape(x.shape[0], self.latent_channels, x.shape[-2] * 2, x.shape[-1] * 2)
        return self.decoder(x)


class TAESDEncoder(nn.Module):

    def __init__(self, encoder_path: os.PathLike):
        super().__init__()

        if "f2" in encoder_path:
            self.latent_channels = 32
        elif "f1" in encoder_path:
            self.latent_channels = 16
        else:
            self.latent_channels = 4

        self.encoder = encoder(self.latent_channels, use_midblock_gn=(self.latent_channels == 32))
        load_state_dict(self.encoder, load_torch_file(encoder_path))

    def forward(self, x_sample: torch.Tensor) -> torch.Tensor:
        if self.latent_channels == 32:
            x_sample = x_sample.reshape(x_sample.shape[0], self.latent_channels, x_sample.shape[-2] // 2, 2, x_sample.shape[-1] // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(x_sample.shape[0], self.latent_channels * 4, x_sample.shape[-2] // 2, x_sample.shape[-1] // 2)
        return self.encoder(x_sample)


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
        model_name = model_name + ".pth"

    loaded_model = sd_vae_taesd_models.get(model_name)

    if loaded_model is None:
        model_path = os.path.join(paths_internal.models_path, "VAE-taesd", model_name)
        download_model(model_path, URL + model_name)

        if not os.path.exists(model_path):
            return None

        loaded_model = TAESDDecoder(model_path)
        loaded_model.eval()
        loaded_model.to(devices.device, devices.dtype)
        sd_vae_taesd_models[model_name] = loaded_model

    return loaded_model


def encoder_model():
    latent_format: "LatentFormat" = shared.sd_model.model_config.latent_format
    model_name: str = latent_format.taesd_decoder_name
    if model_name is None:
        return None
    else:
        model_name = model_name.replace("decoder", "encoder") + ".pth"

    loaded_model = sd_vae_taesd_models.get(model_name)

    if loaded_model is None:
        model_path = os.path.join(paths_internal.models_path, "VAE-taesd", model_name)
        download_model(model_path, URL + model_name)

        if not os.path.exists(model_path):
            return None

        loaded_model = TAESDEncoder(model_path)
        loaded_model.eval()
        loaded_model.to(devices.device, devices.dtype)
        sd_vae_taesd_models[model_name] = loaded_model

    return loaded_model
