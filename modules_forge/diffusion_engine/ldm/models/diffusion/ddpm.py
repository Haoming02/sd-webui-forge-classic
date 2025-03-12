# Wild Mixture of
# - https://github.com/lucidrains/denoising-diffusion-pytorch/blob/0.1.0a/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# - https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# - https://github.com/CompVis/taming-transformers
#
# Thank you!

from functools import partial

import numpy as np
import torch
from ldm_patched.ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm_patched.ldm.util import default, instantiate_from_config
from ldm_patched.modules.model_sampling import make_beta_schedule
from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin

__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}


class DDPM(_DeviceDtypeModuleMixin, torch.nn.Module):
    """Classic DDPM with Gaussian Diffusion, in image space"""

    def __init__(
        self,
        unet_config,
        conditioning_key,
        parameterization="eps",
        timesteps=1000,
        beta_schedule="linear",
        first_stage_key="image",
        channels=3,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.parameterization = parameterization if parameterization in ("eps", "x0", "v") else "eps"

        self.cond_stage_model: torch.nn.Module = None
        self.first_stage_model: torch.nn.Module = None

        self.first_stage_key = first_stage_key
        self.channels = channels

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if not isinstance(cond, dict):
            if not isinstance(cond, list):
                cond = [cond]
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon


class LatentDiffusion(DDPM):
    """Main Class"""

    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        concat_mode=True,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        *args,
        **kwargs,
    ):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"

        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        self.concat_mode = concat_mode
        self.cond_stage_key = cond_stage_key
        self.num_downs = 0

        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()

    def instantiate_cond_stage(self, config):
        model = instantiate_from_config(config)
        self.cond_stage_model = model.eval()

    def get_first_stage_encoding(self, *args, **kwargs):
        raise NotImplementedError

    def get_learned_conditioning(self, c):
        if callable(getattr(self.cond_stage_model, "encode", None)):
            c = self.cond_stage_model.encode(c)
            if isinstance(c, DiagonalGaussianDistribution):
                c = c.mode()
            return c

        return self.cond_stage_model(c)

    def decode_first_stage(self, *args, **kwargs):
        raise NotImplementedError

    def encode_first_stage(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError


class DiffusionWrapper(_DeviceDtypeModuleMixin, torch.nn.Module):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key

    def forward(self, *args, **kwargs):
        return None


class LatentFinetuneDiffusion(LatentDiffusion):
    """Basis for different finetunes, such as inpainting or depth2image"""

    def __init__(self, concat_keys: tuple, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_keys = concat_keys


class LatentInpaintDiffusion(LatentFinetuneDiffusion):
    """
    Can either run as pure inpainting model (only concat mode) or with mixed conditionings,
    e.g. mask as concat and text via cross-attn
    """

    def __init__(
        self,
        concat_keys=("mask", "masked_image"),
        masked_image_key="masked_image",
        *args,
        **kwargs,
    ):
        super().__init__(concat_keys, *args, **kwargs)
        self.masked_image_key = masked_image_key
        assert self.masked_image_key in concat_keys
