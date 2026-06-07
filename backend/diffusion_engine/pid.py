import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionDiscreteFlow
from backend.nn.vae import AutoencoderKLFlux2, IntegratedAutoencoderKL
from backend.nn.wan_vae import WanVAE
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.gemma_it_engine import GemmaTextProcessingEngine
from modules_forge.packages.huggingface_guess import latent


class PiD(ForgeDiffusionEngine):
    matched_guesses = [model_list.PiD]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"gemma2": huggingface_components["text_encoder"]}, tokenizer_dict={"gemma2": huggingface_components["tokenizer"]})

        _vae: torch.nn.Module = huggingface_components["vae"]

        if _wan := isinstance(_vae, WanVAE):
            _vae.latent_format = latent.Wan21()
        if _f2 := isinstance(_vae, AutoencoderKLFlux2):
            _vae.latent_format = latent.Flux2()
        if isinstance(_vae, IntegratedAutoencoderKL):
            _vae.latent_format = latent.SDXL() if _vae.embed_dim == 4 else latent.Flux()

        vae = VAE(model=_vae, is_wan=_wan, is_flux2=_f2)

        k_predictor = PredictionDiscreteFlow(estimated_config)

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_gemma = GemmaTextProcessingEngine(
            text_encoder=clip.cond_stage_model.gemma2,
            tokenizer=clip.tokenizer.gemma2,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.use_shift = True

        self.degrade_sigma: float = None

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine_gemma(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_gemma.tokenize([prompt])[0])
        return token_count, max(300, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        if not dynamic_args.is_referencing:
            raise SystemError("PiD only supports txt2img")

        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        if sample.ndim == 5:
            sample = sample[:, :, 0]

        dynamic_args.lq_latent = (sample, torch.tensor([float(self.degrade_sigma)], dtype=torch.float32))
        return None

    @torch.inference_mode()
    def decode_first_stage(self, x):
        return x
