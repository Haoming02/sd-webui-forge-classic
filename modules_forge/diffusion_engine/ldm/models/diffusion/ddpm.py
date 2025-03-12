# Wild Mixture of
# - https://github.com/lucidrains/denoising-diffusion-pytorch/blob/0.1.0a/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# - https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# - https://github.com/CompVis/taming-transformers
#
# Thank you!

import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager, nullcontext
from functools import partial
import itertools
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
from omegaconf import ListConfig

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(torch.nn.Module):
    """Classic DDPM with Gaussian Diffusion, in image space"""

    def __init__(self,
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
                 *args, **kwargs
                 ):
        super().__init__()

        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.parameterization = parameterization if parameterization in ("eps", "x0", "v") else "eps"

        self.cond_stage_model: torch.nn.Module = None
        self.first_stage_model: torch.nn.Module = None

        self.first_stage_key = first_stage_key
        self.channels = channels

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def dtype(self):  # this property was originally implemented in pytorch_lightning
        from modules.devices import dtype_unet
        return dtype_unet


class LatentDiffusion(DDPM):
    """Main Class"""

    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 concat_mode=True,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'

        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        self.concat_mode = concat_mode
        self.cond_stage_key = cond_stage_key
        self.num_downs = 0

        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
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
        if callable(getattr(self.cond_stage_model, 'encode', None)):
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


class DiffusionWrapper(torch.nn.Module):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key

    def forward(self, *args, **kwargs):
        return None


class LatentFinetuneDiffusion(LatentDiffusion):
    """Basis for different finetunes, such as inpainting or depth2image"""

    def __init__(self,
                 concat_keys: tuple,
                 *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.concat_keys = concat_keys


class LatentInpaintDiffusion(LatentFinetuneDiffusion):
    """
    Can either run as pure inpainting model (only concat mode) or with mixed conditionings,
    e.g. mask as concat and text via cross-attn
    """

    def __init__(self,
                 concat_keys=("mask", "masked_image"),
                 masked_image_key="masked_image",
                 *args, **kwargs
                 ):
        super().__init__(concat_keys, *args, **kwargs)
        self.masked_image_key = masked_image_key
        assert self.masked_image_key in concat_keys
