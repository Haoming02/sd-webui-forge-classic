import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionDiscreteFlow
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.anima_engine import AnimaTextProcessingEngine


class Anima(ForgeDiffusionEngine):
    matched_guesses = [model_list.Anima]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen3_06b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3_06b": huggingface_components["tokenizer"], "t5xxl": huggingface_components["tokenizer_2"]})

        vae = VAE(model=huggingface_components["vae"], is_wan=True)

        k_predictor = PredictionDiscreteFlow(estimated_config)

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
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine_anima(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_anima.tokenize([prompt])[0][0])
        return token_count, max(512, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        samples: list[torch.Tensor] = []
        batch: int = x.size(0)

        for b in range(batch):
            y = x[b].unsqueeze(0)
            sample = self.forge_objects.vae.encode(y.movedim(1, -1) * 0.5 + 0.5)
            sample = self.forge_objects.vae.first_stage_model.process_in(sample)
            samples.append(sample)

        return torch.cat(samples).to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor):
        samples: list[torch.Tensor] = []
        batch: int = x.size(0)

        for b in range(batch):
            y = x[b].unsqueeze(0)
            sample = self.forge_objects.vae.first_stage_model.process_out(y)
            sample = self.forge_objects.vae.decode(sample).movedim(-1, 2) * 2.0 - 1.0
            samples.append(sample)

        return torch.cat(samples).to(x)
