"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)

https://github.com/madebyollin/taesd
"""

import os

import torch
import torch.nn as nn

from modules import devices, paths_internal, shared

sd_vae_taesd_models = {}


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
            conv1x1 = lambda c_in, c_out: nn.Conv2d(c_in, c_out, 1, bias=False)
            n_gn = n_in * 4
            self.pool = nn.Sequential(conv1x1(n_in, n_gn), nn.GroupNorm(4, n_gn), nn.ReLU(inplace=True), conv1x1(n_gn, n_in))

    def forward(self, x):
        if self.pool is not None:
            x = x + self.pool(x)

        return self.fuse(self.conv(x) + self.skip(x))


def decoder(latent_channels=4, use_midblock_gn=False):
    mb_kw = {"use_midblock_gn": use_midblock_gn}

    return nn.Sequential(
        Clamp(),
        conv(latent_channels, 64),
        nn.ReLU(),
        Block(64, 64, **mb_kw),
        Block(64, 64, **mb_kw),
        Block(64, 64, **mb_kw),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(64, 64),
        conv(64, 3),
    )


def encoder(latent_channels=4, use_midblock_gn=False):
    mb_kw = {"use_midblock_gn": use_midblock_gn}

    return nn.Sequential(
        conv(3, 64),
        Block(64, 64),
        conv(64, 64, stride=2, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        conv(64, 64, stride=2, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        conv(64, 64, stride=2, bias=False),
        Block(64, 64, **mb_kw),
        Block(64, 64, **mb_kw),
        Block(64, 64, **mb_kw),
        conv(64, latent_channels),
    )


def guess_latent_channels_and_arch(path):
    lower = str(path).lower()

    if "taef2" in lower:
        return 32, "flux_2"
    if "taef1" in lower or "taesd3" in lower:
        return 16, None

    return 4, None


def _pack_flux2_latents(x):
    b, c, h, w = x.shape
    if h % 2 != 0 or w % 2 != 0:
        return x

    return x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def _unpack_flux2_latents(x):
    b, c, h, w = x.shape
    if c % 4 != 0:
        return x

    return x.view(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)


class Flux2DecoderAdapter(nn.Module):
    def __init__(self, decoder_impl):
        super().__init__()
        self.decoder_impl = decoder_impl

    def forward(self, x):
        if x.shape[1] == 128:
            x = _unpack_flux2_latents(x)

        return self.decoder_impl(x)


class Flux2EncoderAdapter(nn.Module):
    def __init__(self, encoder_impl):
        super().__init__()
        self.encoder_impl = encoder_impl

    def forward(self, x):
        x = self.encoder_impl(x)

        if x.shape[1] == 32:
            x = _pack_flux2_latents(x)

        return x


class TAESDDecoder(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, decoder_path="taesd_decoder.pth", latent_channels=None, arch_variant=None):
        super().__init__()

        if latent_channels is None:
            latent_channels, guessed_arch = guess_latent_channels_and_arch(decoder_path)
            if arch_variant is None:
                arch_variant = guessed_arch

        self.decoder = decoder(latent_channels, use_midblock_gn=(arch_variant == "flux_2"))
        self.decoder_api = Flux2DecoderAdapter(self.decoder) if arch_variant == "flux_2" else self.decoder
        self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu" if devices.device.type != "cuda" else None))


class TAESDEncoder(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path="taesd_encoder.pth", latent_channels=None, arch_variant=None):
        super().__init__()

        if latent_channels is None:
            latent_channels, guessed_arch = guess_latent_channels_and_arch(encoder_path)
            if arch_variant is None:
                arch_variant = guessed_arch

        self.encoder = encoder(latent_channels, use_midblock_gn=(arch_variant == "flux_2"))
        self.encoder_api = Flux2EncoderAdapter(self.encoder) if arch_variant == "flux_2" else self.encoder
        self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu" if devices.device.type != "cuda" else None))


def download_model(model_path, model_url):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        print(f"Downloading TAESD model to: {model_path}")
        torch.hub.download_url_to_file(model_url, model_path)


def decoder_model():
    if shared.sd_model.is_flux2:
        model_name = "taef2_decoder.pth"
    elif shared.sd_model.is_flux:
        model_name = "taef1_decoder.pth"
    elif shared.sd_model.is_sdxl:
        model_name = "taesdxl_decoder.pth"
    elif shared.sd_model.is_sd1:
        model_name = "taesd_decoder.pth"
    else:
        return None

    loaded_model = sd_vae_taesd_models.get(model_name)

    if loaded_model is None:
        model_path = os.path.join(paths_internal.models_path, "VAE-taesd", model_name)
        download_model(model_path, "https://github.com/madebyollin/taesd/raw/main/" + model_name)

        if os.path.exists(model_path):
            loaded_model = TAESDDecoder(model_path)
            loaded_model.eval()
            loaded_model.to(devices.device, devices.dtype)
            sd_vae_taesd_models[model_name] = loaded_model
        else:
            raise FileNotFoundError("TAESD model not found...")

    return getattr(loaded_model, "decoder_api", loaded_model.decoder)


def encoder_model():
    if shared.sd_model.is_flux2:
        model_name = "taef2_encoder.pth"
    elif shared.sd_model.is_flux:
        model_name = "taef1_encoder.pth"
    elif shared.sd_model.is_sdxl:
        model_name = "taesdxl_encoder.pth"
    elif shared.sd_model.is_sd1:
        model_name = "taesd_encoder.pth"
    else:
        return None

    loaded_model = sd_vae_taesd_models.get(model_name)

    if loaded_model is None:
        model_path = os.path.join(paths_internal.models_path, "VAE-taesd", model_name)
        download_model(model_path, "https://github.com/madebyollin/taesd/raw/main/" + model_name)

        if os.path.exists(model_path):
            loaded_model = TAESDEncoder(model_path)
            loaded_model.eval()
            loaded_model.to(devices.device, devices.dtype)
            sd_vae_taesd_models[model_name] = loaded_model
        else:
            raise FileNotFoundError("TAESD model not found...")

    return getattr(loaded_model, "encoder_api", loaded_model.encoder)
