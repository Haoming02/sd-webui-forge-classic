import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.classic_engine import ClassicTextProcessingEngine


class StableDiffusion(ForgeDiffusionEngine):
    matched_guesses = [model_list.SD15]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"clip_l": huggingface_components["text_encoder"]}, tokenizer_dict={"clip_l": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"])

        unet = UnetPatcher.from_model(model=huggingface_components["unet"], diffusers_scheduler=huggingface_components["scheduler"], config=estimated_config)

        self.text_processing_engine = ClassicTextProcessingEngine(
            text_encoder=clip.cond_stage_model.clip_l,
            tokenizer=clip.tokenizer.clip_l,
            embedding_dir=dynamic_args.embedding_dir,
            embedding_key="clip_l",
            embedding_expected_shape=768,
            text_projection=False,
            minimal_clip_skip=1,
            clip_skip=1,
            return_pooled=False,
            final_layer_norm=True,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        # WebUI Legacy
        self.is_sd1 = True

    def set_clip_skip(self, clip_skip):
        self.text_processing_engine.clip_skip = clip_skip

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        cond = self.text_processing_engine(prompt)
        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        _, token_count = self.text_processing_engine.process_texts([prompt])
        return token_count, self.text_processing_engine.get_target_prompt_token_count(token_count)
