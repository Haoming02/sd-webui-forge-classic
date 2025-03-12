import torch.nn
from ldm_patched.ldm.util import instantiate_from_config
from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from omegaconf import OmegaConf


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


class DiffusionWrapper(_DeviceDtypeModuleMixin, torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(model_config)

    def forward(self, *args, **kwargs):
        return None
