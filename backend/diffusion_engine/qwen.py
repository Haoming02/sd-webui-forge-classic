from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.misc.image_resize import resize_reference
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.qwen_image_engine import Qwen25VL7BEngine
from modules.shared import opts


class QwenImage(ForgeDiffusionEngine):
    matched_guesses = [model_list.QwenImage]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen25_7b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen25_7b": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_wan=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_qwen = Qwen25VL7BEngine(
            text_encoder=clip.cond_stage_model.qwen25_7b,
            tokenizer=clip.tokenizer.qwen25_7b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_wan = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        if not prompt.is_negative_prompt:
            _references = [*self.ref_latents]
            if self.ini_latent is not None:
                _references.insert(0, self.ini_latent)
                self.ini_latent = None

            if dynamic_args.edit and bool(_references):
                return self.get_learned_conditioning_with_image(prompt, _references)
            else:
                dynamic_args.ref_latents.clear()

        return self.text_processing_engine_qwen(prompt)

    @torch.inference_mode()
    def get_learned_conditioning_with_image(self, prompt: list[str], images: list[torch.Tensor]):
        images_vl, ref_latents, image_prompts = [], [], []
        for i, image in enumerate(images):
            v, r, p = self.encode_vision(image, i)
            images_vl.append(v)
            ref_latents.append(r)
            image_prompts.append(p)

        dynamic_args.ref_latents = ref_latents.copy()
        return self.text_processing_engine_qwen(["\n".join([*image_prompts, *prompt])], images=images_vl)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_qwen.tokenize(prompt))
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_vision(self, image: torch.Tensor, i: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        samples = image.movedim(-1, 1)

        _vision = resize_reference(samples, 1, 384).movedim(1, -1)[:, :, :, :3]

        s = resize_reference(samples, resolution=1024)
        sample = self.forge_objects.vae.encode(s.movedim(1, -1)[:, :, :, :3])
        _latent = self.forge_objects.vae.first_stage_model.process_in(sample)

        _prompt = f"Picture {i}: <|vision_start|><|image_pad|><|vision_end|>"

        return (_vision, _latent, _prompt)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        if dynamic_args.edit:
            self.store_reference(x[:1].movedim(1, -1).mul(0.5).add(0.5).cpu())

        if not dynamic_args.is_referencing:
            return super().encode_first_stage(x)
