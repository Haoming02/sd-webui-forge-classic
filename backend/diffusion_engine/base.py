from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from backend.patcher.clip import CLIP
    from backend.patcher.unet import UnetPatcher
    from backend.patcher.vae import VAE

from backend import memory_management, utils


class ForgeObjects:
    def __init__(self, unet, clip, vae, clipvision):
        self.unet: "UnetPatcher" = unet
        self.clip: "CLIP" = clip
        self.vae: "VAE" = vae
        self.clipvision = clipvision

    def shallow_copy(self):
        return ForgeObjects(self.unet, self.clip, self.vae, self.clipvision)


class ForgeDiffusionEngine:
    matched_guesses = []

    def __init__(self, estimated_config, huggingface_components):
        self.model_config = estimated_config
        self.is_inpaint = estimated_config.inpaint_model()

        self.forge_objects: "ForgeObjects" = None
        self.forge_objects_original: "ForgeObjects" = None
        self.forge_objects_after_applying_lora: "ForgeObjects" = None

        self.current_lora_hash = str([])

        self.fix_for_webui_backward_compatibility()

        self.ini_latent: "torch.Tensor" = None  # image from img2img input
        self.ref_latents: list["torch.Tensor"] = []  # images from ImageStitch

    def set_clip_skip(self, clip_skip):
        pass

    def get_first_stage_encoding(self, x):
        return x

    def get_learned_conditioning(self, prompt: list[str]):
        raise NotImplementedError

    def encode_first_stage(self, x):
        raise NotImplementedError

    def decode_first_stage(self, x):
        raise NotImplementedError

    def get_prompt_lengths_on_ui(self, prompt):
        return 0, 75

    def is_webui_legacy_model(self):
        return self.is_sd1 or self.is_sdxl

    def fix_for_webui_backward_compatibility(self):
        self.tiling_enabled = False
        self.use_distilled_cfg_scale = False
        self.use_shift = False
        self.is_sd1 = False
        self.is_sdxl = False
        self.is_wan = False  # affects the usage of WanVAE (B, C, T, H, W)

    @property
    def first_stage_model(self):
        try:
            return self.forge_objects.vae.first_stage_model
        except Exception:
            return None

    @property
    def cond_stage_model(self):
        try:
            return self.forge_objects.clip.cond_stage_model
        except Exception:
            return None

    def clear_references(self):
        # called by ImageStitch
        self.ref_latents.clear()
        memory_management.soft_empty_cache()

    def save_unet(self, filename):
        import safetensors.torch as sf

        sd = utils.get_state_dict_after_quant(self.forge_objects.unet.model.diffusion_model)
        sf.save_file(sd, filename)
        return filename

    def save_checkpoint(self, filename):
        import safetensors.torch as sf

        sd = {}
        sd.update(utils.get_state_dict_after_quant(self.forge_objects.unet.model.diffusion_model, prefix="model.diffusion_model."))
        sd.update(utils.get_state_dict_after_quant(self.forge_objects.clip.cond_stage_model, prefix="text_encoders."))
        sd.update(utils.get_state_dict_after_quant(self.forge_objects.vae.first_stage_model, prefix="vae."))
        sf.save_file(sd, filename)
        return filename
