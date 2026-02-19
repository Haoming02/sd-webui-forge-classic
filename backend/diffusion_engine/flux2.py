from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionFlux2
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.klein_engine import KleinTextProcessingEngine


class Flux2(ForgeDiffusionEngine):
    matched_guesses = [model_list.Flux2K4B, model_list.Flux2K9B]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        clip = CLIP(model_dict={"qwen3": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_flux2=True)

        k_predictor = PredictionFlux2(estimated_config)

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_gemma = KleinTextProcessingEngine(
            text_encoder=clip.cond_stage_model.qwen3,
            tokenizer=clip.tokenizer.qwen3,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        if not prompt.is_negative_prompt:
            _references = [*self.ref_latents]
            if self.ini_latent is not None:
                _references.insert(0, self.ini_latent)
                self.ini_latent = None
            dynamic_args["ref_latents"] = _references.copy()

        return self.text_processing_engine_gemma(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_gemma.tokenize([prompt])[0])
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)

        if dynamic_args["is_referencing"]:
            self.ref_latents.append(sample.cpu())
        else:
            self.ini_latent = sample.cpu()

        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)
