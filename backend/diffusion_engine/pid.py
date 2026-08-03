import torch
from huggingface_guess import latent, model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.misc.context_windows import IndexListContextHandler
from backend.nn.vae import AutoencoderKLFlux2, IntegratedAutoencoderKL
from backend.nn.wan_vae import WanVAE
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.gemma_it_engine import GemmaTextProcessingEngine
from modules.shared import opts


class PiD(ForgeDiffusionEngine):
    matched_guesses = [model_list.PiD]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"gemma2": huggingface_components["text_encoder"]}, tokenizer_dict={"gemma2": huggingface_components["tokenizer"]})

        ae: torch.nn.Module = huggingface_components["vae"]

        if is_wan := type(ae) is WanVAE:
            ae.latent_format = latent.Wan21()
        if is_flux2 := type(ae) is AutoencoderKLFlux2:
            ae.latent_format = latent.Flux2()
        if type(ae) is IntegratedAutoencoderKL:
            if ae.embed_dim == 4:
                ae.latent_format = latent.SDXL()
            if ae.embed_dim == 16:
                ae.latent_format = latent.Flux()

        vae = VAE(model=ae, is_wan=is_wan, is_flux2=is_flux2)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_gemma = GemmaTextProcessingEngine(
            text_encoder=clip.cond_stage_model.gemma2,
            tokenizer=clip.tokenizer.gemma2,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.use_shift = True

        dynamic_args.context_handler = IndexListContextHandler()

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine_gemma(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_gemma.tokenize([prompt])[0])
        return token_count, max(300, token_count)

    @torch.inference_mode()
    def _validate(self, x: torch.Tensor):
        _, _, h, w = x.shape
        if h * w < 512 * 512:
            memory_management.logger.warning(f"Input Resolution ({w}x{h}) is too small, this may cause discoloration/artifacts...")
        if h * w > 2048 * 2048:
            memory_management.logger.warning(f"Input Resolution ({w}x{h}) is too large, this may cause discoloration/artifacts...")
        step: int = getattr(opts, "res_step", 64)
        h, w = round(h / step) * step, round(w / step) * step
        return torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear")

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        start_image = self._validate(x[:1]).movedim(1, -1).mul_(0.5).add_(0.5)
        sample = self.forge_objects.vae.encode(start_image)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample).squeeze(2)
        dynamic_args.lq_latent[0] = sample.detach().clone()

        return sample

    @torch.inference_mode()
    def decode_first_stage(self, x):
        return x
