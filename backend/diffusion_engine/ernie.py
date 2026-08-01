import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.ministral3_engine import Ministral3TextProcessingEngine


class ErnieImage(ForgeDiffusionEngine):
    matched_guesses = [model_list.ErnieImage]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"ministral3": huggingface_components["text_encoder"]}, tokenizer_dict={"ministral3": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_flux2=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_ministral = Ministral3TextProcessingEngine(
            text_encoder=clip.cond_stage_model.ministral3,
            tokenizer=clip.tokenizer.ministral3,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.use_shift = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine_ministral(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_ministral.tokenize([prompt])[0])
        return token_count, max(999, token_count)
