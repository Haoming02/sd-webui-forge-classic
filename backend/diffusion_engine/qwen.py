import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionDiscreteFlow
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.qwen_engine import QwenTextProcessingEngine
from modules.shared import opts


class QwenImage(ForgeDiffusionEngine):
    matched_guesses = [model_list.QwenImage]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        clip = CLIP(model_dict={"qwen25_7b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen25_7b": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_wan=True)
        vae.first_stage_model.latent_format = self.model_config.latent_format

        k_predictor = PredictionDiscreteFlow(estimated_config)

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_qwen = QwenTextProcessingEngine(
            text_encoder=clip.cond_stage_model.qwen25_7b,
            tokenizer=clip.tokenizer.qwen25_7b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_wan = True

        self.images_vl = []
        self.ref_latents = []
        self.image_prompt = ""

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        if not prompt.is_negative_prompt:
            if self.image_prompt:
                return self.get_learned_conditioning_with_image(prompt)
            else:
                dynamic_args["ref_latents"].clear()
                self.ref_latents.clear()
                self.image_prompt = ""
        return self.text_processing_engine_qwen(prompt)

    @torch.inference_mode()
    def get_learned_conditioning_with_image(self, prompt: list[str]):
        cond = self.text_processing_engine_qwen([self.image_prompt + "".join(prompt)], images=self.images_vl)
        self.images_vl.clear()
        dynamic_args["ref_latents"] = self.ref_latents.copy()
        self.ref_latents.clear()
        self.image_prompt = ""
        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_qwen.tokenize([prompt])[0])
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_vision(self, image: torch.Tensor):
        samples = image.movedim(-1, 1)  # b, c, h, w

        total = int(384 * 384)
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)

        s = torch.nn.functional.interpolate(samples, size=(height, width), mode="area")
        self.images_vl.append(s.movedim(1, -1))

        if opts.qwen_vae_resize:
            total = int(1024 * 1024)
            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by / 32.0) * 32
            height = round(samples.shape[2] * scale_by / 32.0) * 32

            s = torch.nn.functional.interpolate(samples, size=(height, width), mode="area")
        else:
            s = samples.clone()
        sample = self.forge_objects.vae.encode(s.movedim(1, -1)[:, :, :, :3])
        self.ref_latents.append(self.forge_objects.vae.first_stage_model.process_in(sample))

        self.image_prompt += f"Picture {len(self.images_vl)}: <|vision_start|><|image_pad|><|vision_end|>"

    @torch.inference_mode()
    def encode_first_stage(self, x):
        if x.size(0) > 1:
            x = x[0].unsqueeze(0)  # enforce batch_size of 1
        start_image = x.movedim(1, -1) * 0.5 + 0.5
        self.encode_vision(start_image)
        sample = self.forge_objects.vae.encode(start_image)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 2) * 2.0 - 1.0
        return sample.to(x)
