import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.moss_image_engine import Qwen359BEngine
from modules.shared import opts

REFERENCE_METHODS = {
    "ai-toolkit": ("index_timestep_zero_isolated", True),
    "Moss-Image-Edit": ("index_timestep_zero_text_start", False),
}


class MOSSImage(ForgeDiffusionEngine):
    matched_guesses = [model_list.MOSSImage]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen35_9b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen35_9b": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_flux2=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_qwen = Qwen359BEngine(
            text_encoder=clip.cond_stage_model.qwen35_9b,
            tokenizer=clip.tokenizer.qwen35_9b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.use_shift = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        if not prompt.is_negative_prompt:
            _references = [*self.ref_latents]
            if self.ini_latent is not None:
                _references.insert(0, self.ini_latent)
                self.ini_latent = None

            if (mode := opts.moss_do_reference) in REFERENCE_METHODS and bool(_references):
                return self.get_learned_conditioning_with_image(prompt, _references, *REFERENCE_METHODS[mode])
            else:
                dynamic_args.ref_latents.clear()
                dynamic_args.moss_reference_method = None
                dynamic_args.moss_ref_kv = None

        return self.text_processing_engine_qwen(prompt)

    @torch.inference_mode()
    def get_learned_conditioning_with_image(self, prompt: list[str], images: list[torch.Tensor], method: str, vision: bool):
        images_vl, ref_latents = [], []
        for image in images:
            v, r = self.encode_vision(image)
            if vision:
                images_vl.append(v)
            ref_latents.append(r)

        dynamic_args.moss_reference_method = method
        dynamic_args.ref_latents = ref_latents.copy()
        dynamic_args.moss_ref_kv = None
        return self.text_processing_engine_qwen(prompt, images=images_vl)

    @torch.inference_mode()
    def encode_vision(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        samples = image.movedim(-1, 1)  # b, c, h, w

        total = int(384 * 384)
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)

        s = torch.nn.functional.interpolate(samples, size=(height, width), mode="area")
        _vision = s.movedim(1, -1)[:, :, :, :3]

        width = round(samples.shape[3] / 16.0) * 16
        height = round(samples.shape[2] / 16.0) * 16
        s = torch.nn.functional.interpolate(samples, size=(height, width), mode="area")
        sample = self.forge_objects.vae.encode(s.movedim(1, -1)[:, :, :, :3])
        _latent = self.forge_objects.vae.first_stage_model.process_in(sample)

        channels = self.model_config.latent_format.latent_channels
        if _latent.shape[1] < channels:
            _latent = torch.nn.functional.pad(_latent, (0, 0, 0, 0, 0, channels - _latent.shape[1]))

        return (_vision, _latent)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_qwen.tokenize(prompt))
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        if opts.moss_do_reference in REFERENCE_METHODS:
            start_image = x[0].movedim(0, -1).mul(0.5).add(0.5).unsqueeze(0)
            if dynamic_args.is_referencing:
                self.ref_latents.append(start_image.cpu())
            else:
                self.ini_latent = start_image.cpu()

        return super().encode_first_stage(x)
