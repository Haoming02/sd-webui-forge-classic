from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.patcher.clip import CLIP
    from backend.patcher.unet import UnetPatcher
    from backend.patcher.vae import VAE
    from modules_forge.packages.huggingface_guess.model_list import BASE

import torch
from safetensors.torch import save_file

from backend import memory_management, utils
from backend.modules.k_prediction import PredictionDiscreteFlow, PredictionFlux


class ForgeObjects:
    def __init__(self, unet, clip, vae, clipvision):
        self.unet: "UnetPatcher" = unet
        self.clip: "CLIP" = clip
        self.vae: "VAE" = vae
        self.clipvision = clipvision

    def shallow_copy(self):
        return ForgeObjects(self.unet, self.clip, self.vae, self.clipvision)


class ForgeDiffusionEngine:
    matched_guesses = []

    def __init__(self, estimated_config: "BASE", huggingface_components: dict[str, "torch.nn.Module"]):
        huggingface_components["vae"].latent_format = estimated_config.latent_format

        self.model_config: "BASE" = estimated_config
        self.is_inpaint: bool = estimated_config.inpaint_model()

        self.forge_objects: "ForgeObjects" = None
        self.forge_objects_original: "ForgeObjects" = None
        self.forge_objects_after_applying_lora: "ForgeObjects" = None

        self.current_lora_hash = str([])

        self.tiling_enabled = False
        self.use_distilled_cfg_scale = False
        self.use_shift = False
        self.is_sd1 = False
        self.is_sdxl = False
        self.is_wan = False  # affects the usage of WanVAE (B, C, T, H, W)

        self.ini_latent: "torch.Tensor" = None  # image from img2img input
        self.ref_latents: list["torch.Tensor"] = []  # images from ImageStitch

    def _get_predictor(self) -> PredictionDiscreteFlow | PredictionFlux:
        if self.model_config.model_type.name == "FLOW":
            return PredictionDiscreteFlow(self.model_config)
        else:
            return PredictionFlux(self.model_config)

    def set_clip_skip(self, clip_skip):
        pass

    def get_first_stage_encoding(self, x):
        return x

    @abstractmethod
    def get_learned_conditioning(self, prompt: list[str]):
        raise NotImplementedError

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        if self.is_wan:  # otherwise image batch turns into video...
            samples: list[torch.Tensor] = []
            for i in range(x.size(0)):
                start_image = x[i : i + 1].movedim(1, -1).mul(0.5).add(0.5)
                sample = self.forge_objects.vae.encode(start_image)
                sample = self.forge_objects.vae.first_stage_model.process_in(sample)
                samples.append(sample)
            return torch.cat(samples, dim=0).to(x)
        else:
            start_image = x.movedim(1, -1).mul(0.5).add(0.5)
            sample = self.forge_objects.vae.encode(start_image)
            sample = self.forge_objects.vae.first_stage_model.process_in(sample)
            return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, (2 if self.is_wan else 1)).mul_(2.0).sub_(1.0)
        return sample.to(x)

    def get_prompt_lengths_on_ui(self, prompt):
        return 0, 75

    def is_webui_legacy_model(self):
        return self.is_sd1 or self.is_sdxl

    @property
    def first_stage_model(self):
        try:
            return self.forge_objects.vae.first_stage_model
        except Exception:
            return None

    @property
    def cond_stage_model(self):
        try:
            return self.forge_objects.clip.cond_stage_model
        except Exception:
            return None

    def clear_references(self):
        # called by ImageStitch
        self.ref_latents.clear()
        memory_management.soft_empty_cache()

    def set_shift(self, shift: float):
        if not self.use_shift:
            return
        self.forge_objects.unet.model.predictor.set_parameters(shift=shift)
        memory_management.logger.debug(f"Shift: {shift}")

    def save_unet(self, filename: str) -> str:
        sd = utils.get_state_dict_after_quant(self.forge_objects.unet.model.diffusion_model)
        save_file(sd, filename)
        return filename

    def save_checkpoint(self, filename: str) -> str:
        sd = {}

        sd.update(self.model_config.process_unet_state_dict_for_saving(utils.get_state_dict_after_quant(self.forge_objects.unet.model.diffusion_model)))
        sd.update(self.model_config.process_clip_state_dict_for_saving(utils.get_state_dict_after_quant(self.forge_objects.clip.cond_stage_model)))
        sd.update(self.model_config.process_vae_state_dict_for_saving(utils.get_state_dict_after_quant(self.forge_objects.vae.first_stage_model)))

        save_file(sd, filename)
        return filename
