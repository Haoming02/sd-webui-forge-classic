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
from backend.text_processing.anima_engine import AnimaTextProcessingEngine
from modules.shared import opts


class Anima(ForgeDiffusionEngine):
    matched_guesses = [model_list.Anima]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen3_06b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3_06b": huggingface_components["tokenizer"], "t5xxl": huggingface_components["tokenizer_2"]})

        vae = VAE(model=huggingface_components["vae"], is_wan=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_anima = AnimaTextProcessingEngine(
            text_encoder=clip.cond_stage_model.qwen3_06b,
            qwen_tokenizer=clip.tokenizer.qwen3_06b,
            t5_tokenizer=clip.tokenizer.t5xxl,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_wan = True
        self.use_shift = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        if not prompt.is_negative_prompt:
            if not opts.anima_do_reference:
                dynamic_args.ref_latents.clear()
            else:
                _references = [*self.ref_latents]
                if self.ini_latent is not None:
                    _references.insert(0, self.ini_latent)
                    self.ini_latent = None
                dynamic_args.ref_latents = _references.copy()

        return self.text_processing_engine_anima(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_anima.tokenize([prompt])[0][0])
        return token_count, max(512, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        samples: torch.Tensor = super().encode_first_stage(x)

        if opts.anima_do_reference:
            sample = samples[0].detach().clone().unsqueeze(0).cpu()
            if dynamic_args.is_referencing:
                self.ref_latents.append(sample)
            else:
                self.ini_latent = sample

        return samples
