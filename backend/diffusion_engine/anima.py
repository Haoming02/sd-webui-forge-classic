import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionDiscreteFlow
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.anima_engine import AnimaTextProcessingEngine

# ComfyUI-derived implementation pieces are documented in:
# - backend/nn/anima.py
# - backend/text_processing/anima_engine.py


class Anima(ForgeDiffusionEngine):
    matched_guesses = [model_list.Anima]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        clip = CLIP(model_dict={"qwen3_06b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3_06b": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_wan=True)
        vae.first_stage_model.latent_format = self.model_config.latent_format

        k_predictor = PredictionDiscreteFlow(estimated_config)

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        t5_tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "huggingface", "circlestone-labs", "Anima", "tokenizer_t5")

        self.text_processing_engine_qwen3 = AnimaTextProcessingEngine(
            text_encoder=clip.cond_stage_model.qwen3_06b,
            qwen_tokenizer=clip.tokenizer.qwen3_06b,
            t5_tokenizer_path=t5_tokenizer_path,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        # The Wan-style VAE path is used for this model.
        self.is_wan = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine_qwen3(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_qwen3.tokenize([prompt])[0])
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        if sample.ndim == 5:
            # For image workflows, keep a single temporal slice.
            sample = sample[:, 0]
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        if x.ndim == 4:
            # Wan21 latent_format expects (B, C, T, H, W). Keep image path as T=1.
            x = x.unsqueeze(2)
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 2) * 2.0 - 1.0
        if sample.ndim == 5:
            # VAE may decode to (B, T, C, H, W). Use first frame for image output.
            sample = sample[:, 0]
        return sample.to(x)
