import contextlib

import torch
from omegaconf import OmegaConf

import ldm_patched.modules.clip_vision
import ldm_patched.modules.utils
from ldm_patched.ldm.util import instantiate_from_config
from ldm_patched.modules import model_detection, model_management
from ldm_patched.modules.model_base import ModelType, model_sampling
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.sd import CLIP, VAE, load_model_weights
from modules import sd_hijack, shared
from modules.sd_models_config import find_checkpoint_config
from modules.sd_models_types import WebuiSdModel
from modules_forge import forge_clip
from modules_forge.unet_patcher import UnetPatcher


class FakeObject:
    def __init__(self, *args, **kwargs):
        return

    def eval(self, *args, **kwargs):
        return self

    def parameters(self, *args, **kwargs):
        return []


class ForgeObjects:
    def __init__(self, unet, clip, vae, clipvision):
        self.unet: UnetPatcher = unet
        self.clip: CLIP = clip
        self.vae: VAE = vae
        self.clipvision: ModelPatcher = clipvision

    def shallow_copy(self):
        return ForgeObjects(self.unet, self.clip, self.vae, self.clipvision)


@torch.no_grad()
def load_checkpoint_guess_config(sd, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True) -> ForgeObjects:
    clip = None
    clipvision = None
    vae = None
    model = None
    unet = None
    clip_target = None

    parameters = ldm_patched.modules.utils.calculate_parameters(sd, "model.diffusion_model.")
    unet_dtype = model_management.unet_dtype(model_params=parameters)
    load_device = model_management.get_torch_device()
    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device)

    class WeightsLoader(torch.nn.Module):
        pass

    model_config = model_detection.model_config_from_unet(sd, "model.diffusion_model.", unet_dtype)
    model_config.set_manual_cast(manual_cast_dtype)

    if model_config is None:
        raise RuntimeError("Could not detect model type")

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = ldm_patched.modules.clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    if output_model:
        initial_load_device = model_management.unet_initial_load_device(parameters, unet_dtype)
        print("UNet dtype:", unet_dtype)
        model = model_config.get_model(sd, "model.diffusion_model.", device=initial_load_device)
        model.load_model_weights(sd, "model.diffusion_model.")

        unet = UnetPatcher(
            model,
            load_device=load_device,
            offload_device=model_management.unet_offload_device(),
            current_device=initial_load_device,
            weight_inplace_update=shared.opts.extra_networks_patch_inplace,
        )
        if initial_load_device != torch.device("cpu"):
            print("loaded straight to GPU")
            model_management.load_model_gpu(unet)

    if output_vae:
        vae_sd = ldm_patched.modules.utils.state_dict_prefix_replace(sd, {"first_stage_model.": ""}, filter_keys=True)
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VAE(sd=vae_sd)
        del vae_sd

    if output_clip:
        w = WeightsLoader()
        clip_target = model_config.clip_target()
        if clip_target is not None:
            clip = CLIP(clip_target, embedding_directory=embedding_directory)
            w.cond_stage_model = clip.cond_stage_model
            sd = model_config.process_clip_state_dict(sd)
            load_model_weights(w, sd)

    left_over = sd.keys()
    if len(left_over) > 0:
        print("left over keys:", left_over)

    return ForgeObjects(unet, clip, vae, clipvision)


@torch.no_grad()
def load_model_for_a1111(timer, checkpoint_info=None, state_dict=None) -> WebuiSdModel:
    ztsnr = False
    if state_dict is not None:
        ztsnr = state_dict.pop("ztsnr", None) is not None

    a1111_config_filename = find_checkpoint_config(state_dict, checkpoint_info)
    a1111_config = OmegaConf.load(a1111_config_filename)
    timer.record("forge solving config")

    for obj in ("unet_config", "network_config", "first_stage_config"):
        if hasattr(a1111_config.model.params, obj):
            getattr(a1111_config.model.params, obj).target = "modules_forge.forge_loader.FakeObject"

    sd_model: WebuiSdModel = instantiate_from_config(a1111_config.model)

    del a1111_config
    timer.record("forge instantiate config")

    forge_objects = load_checkpoint_guess_config(
        state_dict,
        output_vae=True,
        output_clip=True,
        output_clipvision=True,
        embedding_directory=shared.cmd_opts.embeddings_dir,
        output_model=True,
    )

    sd_model.forge_objects_original = forge_objects
    sd_model.forge_objects = sd_model.forge_objects_original.shallow_copy()
    sd_model.forge_objects_after_applying_lora = None

    del state_dict
    timer.record("forge load real models")

    sd_model.first_stage_model = forge_objects.vae.first_stage_model
    sd_model.model.diffusion_model = forge_objects.unet.model.diffusion_model

    conditioner = getattr(sd_model, "conditioner", None)
    sd_model.is_sdxl = conditioner is not None

    if sd_model.is_sdxl:
        for i in range(len(conditioner.embedders)):
            embedder = conditioner.embedders[i]
            typename = type(embedder).__name__

            if typename == "FrozenCLIPEmbedder":  # Clip L
                embedder.tokenizer = forge_objects.clip.tokenizer.clip_l.tokenizer
                embedder.transformer = forge_objects.clip.cond_stage_model.clip_l.transformer
                model_embeddings = embedder.transformer.text_model.embeddings
                model_embeddings.token_embedding = sd_hijack.EmbeddingsWithFixes(model_embeddings.token_embedding, sd_hijack.model_hijack)
                conditioner.embedders[i] = forge_clip.CLIP_SD_XL_L(embedder, sd_hijack.model_hijack)

            elif typename == "FrozenOpenCLIPEmbedder2":  # Clip G
                embedder.tokenizer = forge_objects.clip.tokenizer.clip_g.tokenizer
                embedder.transformer = forge_objects.clip.cond_stage_model.clip_g.transformer
                embedder.text_projection = forge_objects.clip.cond_stage_model.clip_g.text_projection
                model_embeddings = embedder.transformer.text_model.embeddings
                model_embeddings.token_embedding = sd_hijack.EmbeddingsWithFixes(model_embeddings.token_embedding, sd_hijack.model_hijack, textual_inversion_key="clip_g")
                conditioner.embedders[i] = forge_clip.CLIP_SD_XL_G(embedder, sd_hijack.model_hijack)

            elif typename == "ConcatTimestepEmbedderND":
                embedder.device = model_management.text_encoder_device()

        sd_model.cond_stage_model = conditioner

    else:
        assert type(sd_model.cond_stage_model).__name__ == "FrozenCLIPEmbedder"
        sd_model.cond_stage_model.tokenizer = forge_objects.clip.tokenizer.clip_l.tokenizer
        sd_model.cond_stage_model.transformer = forge_objects.clip.cond_stage_model.clip_l.transformer
        model_embeddings = sd_model.cond_stage_model.transformer.text_model.embeddings
        model_embeddings.token_embedding = sd_hijack.EmbeddingsWithFixes(model_embeddings.token_embedding, sd_hijack.model_hijack)

        sd_model.cond_stage_model = forge_clip.CLIP_SD_15_L(sd_model.cond_stage_model, sd_hijack.model_hijack)

    timer.record("forge set components")

    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    if getattr(sd_model, "parameterization", None) == "v":
        sd_model.forge_objects.unet.model.model_sampling = model_sampling(sd_model.forge_objects.unet.model.model_config, ModelType.V_PREDICTION)
        sd_model.alphas_cumprod_original = sd_model.alphas_cumprod

    sd_model.ztsnr = ztsnr
    sd_model.is_sd2 = False
    sd_model.is_sd1 = not sd_model.is_sdxl
    sd_model.sd_model_hash = sd_model_hash
    sd_model.sd_model_checkpoint = checkpoint_info.filename
    sd_model.sd_checkpoint_info = checkpoint_info

    @torch.inference_mode()
    def patched_decode_first_stage(x):
        sample = sd_model.forge_objects.unet.model.model_config.latent_format.process_out(x)
        sample = sd_model.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)

    @torch.inference_mode()
    def patched_encode_first_stage(x):
        sample = sd_model.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = sd_model.forge_objects.unet.model.model_config.latent_format.process_in(sample)
        return sample.to(x)

    sd_model.ema_scope = lambda *args, **kwargs: contextlib.nullcontext()
    sd_model.get_first_stage_encoding = lambda x: x
    sd_model.decode_first_stage = patched_decode_first_stage
    sd_model.encode_first_stage = patched_encode_first_stage
    sd_model.clip = sd_model.cond_stage_model
    sd_model.tiling_enabled = False
    timer.record("forge finalize")

    sd_model.current_lora_hash = str([])
    return sd_model


def rescale_zero_terminal_snr_abar(alphas_cumprod):
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas_bar[-1] = 4.8973451890853435e-08
    return alphas_bar


def rescale_zero_terminal_snr_sigmas(sigmas):
    """https://github.com/comfyanonymous/ComfyUI/blob/v0.3.48/comfy/model_sampling.py#L5"""
    alphas_cumprod = 1 / ((sigmas * sigmas) + 1)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas_bar[-1] = 4.8973451890853435e-08
    return ((1 - alphas_bar) / alphas_bar) ** 0.5


def apply_alpha_schedule_override(sd_model, p=None):
    """
    Applies an override to the alpha schedule of the model according to settings.
    - downcasts the alpha schedule to half precision
    - rescales the alpha schedule to have zero terminal SNR
    """

    if not (hasattr(sd_model, "alphas_cumprod") and hasattr(sd_model, "alphas_cumprod_original")):
        return

    sd_model.alphas_cumprod = sd_model.alphas_cumprod_original.to(shared.device)

    if shared.opts.use_downcasted_alpha_bar:
        if p is not None:
            p.extra_generation_params["Downcast alphas_cumprod"] = shared.opts.use_downcasted_alpha_bar
        sd_model.alphas_cumprod = sd_model.alphas_cumprod.half().to(shared.device)

    if getattr(sd_model, "ztsnr", False) or shared.opts.sd_noise_schedule == "Zero Terminal SNR":
        if p is not None:
            p.extra_generation_params["Noise Schedule"] = "Zero Terminal SNR"
        sd_model.alphas_cumprod = rescale_zero_terminal_snr_abar(sd_model.alphas_cumprod).to(shared.device)
        sd_model.forge_objects.unet.model.model_sampling.set_sigmas(rescale_zero_terminal_snr_sigmas(sd_model.forge_objects.unet.model.model_sampling.sigmas).to(shared.device))


ForgeSD = ForgeObjects
