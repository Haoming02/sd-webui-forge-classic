# https://github.com/Comfy-Org/ComfyUI/blob/v0.38.0/comfy_extras/nodes_qwen.py

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
from backend.text_processing.qwen_image_21_engine import Qwen3VL8BEngine
from modules.shared import opts


class QwenImage21(ForgeDiffusionEngine):
    matched_guesses = [model_list.QwenImage21]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen3vl_8b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3vl_8b": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_qwen21=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_qwen = Qwen3VL8BEngine(
            text_encoder=clip.cond_stage_model.qwen3vl_8b,
            tokenizer=clip.tokenizer.qwen3vl_8b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    def set_shift(self, shift: float, width: int, height: int):
        seq_len = width * height / (16 * 16)

        # shift = base_shift + (max_shift - base_shift) * (sequence_length - base_image_seq_len) / (max_image_seq_len - base_image_seq_len)
        shift = 0.5 + (0.9 - 0.5) * (seq_len - 256) / (8192 - 256)

        self.forge_objects.unet.model.predictor.set_parameters(shift=shift)
        memory_management.logger.debug(f"Shift: {shift}")

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        if not prompt.is_negative_prompt:
            _references = [*self.ref_latents]
            if self.ini_latent is not None:
                _references.insert(0, self.ini_latent)
                self.ini_latent = None

            if opts.qwen21_do_reference and bool(_references):
                return self.get_learned_conditioning_with_image(prompt, _references)
            else:
                dynamic_args.ref_latents.clear()
                dynamic_args.image_slots = None

        return self.text_processing_engine_qwen(prompt)

    @torch.inference_mode()
    def get_learned_conditioning_with_image(self, prompt: list[str], images: list[torch.Tensor]):
        images_vl, ref_latents = [], []
        for image in images:
            v, r = self.encode_vision(image)
            images_vl.append(v)
            ref_latents.append(r)

        dynamic_args.ref_latents = ref_latents.copy()
        return self.text_processing_engine_qwen(prompt, images=images_vl)

    @torch.inference_mode()
    def encode_vision(self, image: torch.Tensor, resolution: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
        samples = image[:1].movedim(-1, 1)  # b, c, h, w

        if resolution > 0:
            ratio = samples.shape[3] / samples.shape[2]
            width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
            height = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
        else:
            width, height = round(samples.shape[3] / 32) * 32, round(samples.shape[2] / 32) * 32

        if (width, height) == (samples.shape[3], samples.shape[2]):
            s = image[:1]
        else:
            s = torch.nn.functional.interpolate(samples, size=(height, width), mode="area").movedim(1, -1)

        rgb = s[:, :, :, :3]
        if s.shape[-1] > 3:
            rgb = rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])

        sample = self.forge_objects.vae.encode(s)
        _latent = self.forge_objects.vae.first_stage_model.process_in(sample)

        return (rgb, _latent)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_qwen.tokenize(prompt))
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        if x.size(1) == 3:
            a = torch.ones((1, 1, x.size(2), x.size(3)), dtype=x.dtype, device=x.device)
            x = torch.cat([x, a], dim=1)

        if opts.qwen21_do_reference:
            start_image = x[0].movedim(0, -1).mul(0.5).add(0.5).unsqueeze(0)
            if dynamic_args.is_referencing:
                self.ref_latents.append(start_image.cpu())
            else:
                self.ini_latent = start_image.cpu()

        return super().encode_first_stage(x)
