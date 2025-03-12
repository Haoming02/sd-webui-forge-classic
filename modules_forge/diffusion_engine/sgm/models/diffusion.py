import torch.nn
from ldm_patched.ldm.util import instantiate_from_config
from ldm_patched.modules.model_management import unet_dtype, text_encoder_dtype, get_torch_device
from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from omegaconf import OmegaConf
from modules.shared import opts
from functools import lru_cache


@lru_cache(maxsize=1, typed=False)
def _alpha():
    from ldm_patched.ldm.modules.diffusionmodules.util import make_beta_schedule
    import numpy as np

    betas = make_beta_schedule("linear", 1000, linear_start=0.00085, linear_end=0.0120)
    return np.cumprod(1.0 - betas, axis=0)


class DiffusionEngine(_DeviceDtypeModuleMixin, torch.nn.Module):
    def __init__(
        self,
        network_config: OmegaConf,
        denoiser_config: OmegaConf,
        first_stage_config: OmegaConf,
        conditioner_config: OmegaConf,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.input_key = input_key
        self.model = DiffusionWrapper(network_config)
        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = None
        self.conditioner = instantiate_from_config(conditioner_config)
        self.instantiate_first_stage(first_stage_config)
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.scale_factor = scale_factor

        # === extend_sdxl ===

        self.model.diffusion_model.dtype = unet_dtype()
        self.model.conditioning_key = "crossattn"
        self.cond_stage_key = "txt"

        self.parameterization = self.denoiser.scaling.parameterization
        self.alphas_cumprod = torch.asarray(_alpha(), device=get_torch_device(), dtype=torch.float32)
        self.conditioner.wrapped = torch.nn.Module()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()

    def decode_first_stage(self, *args, **kwargs):
        raise NotImplementedError

    def encode_first_stage(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    # ========== sd_models_xl.py ========== #

    def get_learned_conditioning(self, batch: list[str]):
        width = getattr(batch, "width", 1024)
        height = getattr(batch, "height", 1024)
        is_negative_prompt = getattr(batch, "is_negative_prompt", False)

        force_zero_negative_prompt = is_negative_prompt and all(x == "" for x in batch)
        aesthetic_score = opts.sdxl_refiner_low_aesthetic_score if is_negative_prompt else opts.sdxl_refiner_high_aesthetic_score

        devices_args = dict(device=self.forge_objects.clip.patcher.current_device, dtype=text_encoder_dtype())
        sdxl_conds = {
            "txt": batch,
            "original_size_as_tuple": torch.tensor([height, width], **devices_args).repeat(len(batch), 1),
            "crop_coords_top_left": torch.tensor([opts.sdxl_crop_top, opts.sdxl_crop_left], **devices_args).repeat(len(batch), 1),
            "target_size_as_tuple": torch.tensor([height, width], **devices_args).repeat(len(batch), 1),
            "aesthetic_score": torch.tensor([aesthetic_score], **devices_args).repeat(len(batch), 1),
        }

        return self.conditioner(sdxl_conds, force_zero_embeddings=["txt"] if force_zero_negative_prompt else [])

    def apply_model(self, x, t, cond, *args, **kwargs):
        if self.model.diffusion_model.in_channels == 9:
            x = torch.cat([x] + cond["c_concat"], dim=1)

        return self.model(x, t, cond, *args, **kwargs)

    def get_first_stage_encoding(self, x):  # for compatibility
        return x


class DiffusionWrapper(_DeviceDtypeModuleMixin, torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(model_config)

    def forward(self, *args, **kwargs):
        return None
