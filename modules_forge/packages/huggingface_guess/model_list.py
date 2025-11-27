# reference: https://github.com/comfyanonymous/ComfyUI/blob/v0.3.52/comfy/supported_models.py

from enum import Enum

import torch

from . import diffusers_convert, latent, utils


class ModelType(Enum):
    EPS = 1
    V_PREDICTION = 2
    FLUX = 3
    FLOW = 4


class BASE:
    huggingface_repo = None
    unet_config = {}
    unet_extra_config = {
        "num_heads": -1,
        "num_head_channels": 64,
    }

    required_keys = {}

    clip_prefix = []
    clip_vision_prefix = None
    noise_aug_config = None
    sampling_settings = {}
    latent_format = latent.LatentFormat
    vae_key_prefix = ["first_stage_model."]
    text_encoder_key_prefix = ["cond_stage_model."]
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    memory_usage_factor = 2.0

    manual_cast_dtype = None
    unet_target = "unet"
    vae_target = "vae"

    @classmethod
    def matches(cls, unet_config, state_dict=None):
        for k in cls.unet_config:
            if k not in unet_config or cls.unet_config[k] != unet_config[k]:
                return False
        if state_dict is not None:
            for k in cls.required_keys:
                if k not in state_dict:
                    return False
        return True

    def model_type(self, state_dict):
        return ModelType.EPS

    def clip_target(self, state_dict: dict):
        return {}

    def inpaint_model(self):
        return self.unet_config.get("in_channels", -1) > 4

    def __init__(self, unet_config):
        self.unet_config = unet_config.copy()
        self.sampling_settings = self.sampling_settings.copy()
        self.latent_format = self.latent_format()
        for x in self.unet_extra_config:
            self.unet_config[x] = self.unet_extra_config[x]

    def process_clip_state_dict(self, state_dict):
        replace_prefix = {k: "" for k in self.text_encoder_key_prefix}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)

    def process_unet_state_dict(self, state_dict):
        return state_dict

    def process_vae_state_dict(self, state_dict):
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": self.text_encoder_key_prefix[0]}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def process_clip_vision_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        if self.clip_vision_prefix is not None:
            replace_prefix[""] = self.clip_vision_prefix
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def process_unet_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": "model.diffusion_model."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def process_vae_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": self.vae_key_prefix[0]}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)


class SD15(BASE):
    huggingface_repo = "runwayml/stable-diffusion-v1-5"

    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    unet_extra_config = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    latent_format = latent.SD15
    memory_usage_factor = 1.0

    def process_clip_state_dict(self, state_dict):
        k = list(state_dict.keys())
        for x in k:
            if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
                y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
                state_dict[y] = state_dict.pop(x)

        if "cond_stage_model.transformer.text_model.embeddings.position_ids" in state_dict:
            ids = state_dict["cond_stage_model.transformer.text_model.embeddings.position_ids"]
            if ids.dtype == torch.float32:
                state_dict["cond_stage_model.transformer.text_model.embeddings.position_ids"] = ids.round()

        replace_prefix = {"cond_stage_model.": "clip_l."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)

    def process_clip_state_dict_for_saving(self, state_dict):
        pop_keys = ["clip_l.transformer.text_projection.weight", "clip_l.logit_scale"]
        for p in pop_keys:
            if p in state_dict:
                state_dict.pop(p)

        replace_prefix = {"clip_l.": "cond_stage_model."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def clip_target(self, state_dict: dict):
        return {"clip_l": "text_encoder"}


class SDXLRefiner(BASE):
    huggingface_repo = "stabilityai/stable-diffusion-xl-refiner-1.0"

    unet_config = {
        "model_channels": 384,
        "use_linear_in_transformer": True,
        "context_dim": 1280,
        "adm_in_channels": 2560,
        "transformer_depth": [0, 0, 4, 4, 4, 4, 0, 0],
        "use_temporal_attention": False,
    }

    latent_format = latent.SDXL
    memory_usage_factor = 1.0

    def process_clip_state_dict(self, state_dict):
        replace_prefix = {"conditioner.embedders.0.model.": "clip_g."}
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)
        return utils.clip_text_transformers_convert(state_dict, "clip_g.", "clip_g.transformer.")

    def process_clip_state_dict_for_saving(self, state_dict):
        state_dict_g = diffusers_convert.convert_text_enc_state_dict_v20(state_dict, "clip_g")
        if "clip_g.transformer.text_model.embeddings.position_ids" in state_dict_g:
            state_dict_g.pop("clip_g.transformer.text_model.embeddings.position_ids")
        replace_prefix = {"clip_g": "conditioner.embedders.0.model"}
        return utils.state_dict_prefix_replace(state_dict_g, replace_prefix)

    def clip_target(self, state_dict: dict):
        return {"clip_g": "text_encoder"}


class SDXL(BASE):
    huggingface_repo = "stabilityai/stable-diffusion-xl-base-1.0"

    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }

    latent_format = latent.SDXL
    memory_usage_factor = 0.8

    def model_type(self, state_dict: dict):
        if "v_pred" in state_dict:
            return ModelType.V_PREDICTION
        else:
            return ModelType.EPS

    def process_clip_state_dict(self, state_dict):
        replace_prefix = {
            "conditioner.embedders.0.transformer.text_model": "clip_l.transformer.text_model",
            "conditioner.embedders.1.model.": "clip_g.",
        }
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)
        return utils.clip_text_transformers_convert(state_dict, "clip_g.", "clip_g.transformer.")

    def process_clip_state_dict_for_saving(self, state_dict):
        state_dict_g = diffusers_convert.convert_text_enc_state_dict_v20(state_dict, "clip_g")
        for k in state_dict:
            if k.startswith("clip_l"):
                state_dict_g[k] = state_dict[k]

        state_dict_g["clip_l.transformer.text_model.embeddings.position_ids"] = torch.arange(77).expand((1, -1))
        pop_keys = ["clip_l.transformer.text_projection.weight", "clip_l.logit_scale"]
        for p in pop_keys:
            if p in state_dict_g:
                state_dict_g.pop(p)

        replace_prefix = {
            "clip_g": "conditioner.embedders.1.model",
            "clip_l": "conditioner.embedders.0",
        }
        return utils.state_dict_prefix_replace(state_dict_g, replace_prefix)

    def clip_target(self, state_dict: dict):
        return {"clip_l": "text_encoder", "clip_g": "text_encoder_2"}


class Flux(BASE):
    huggingface_repo = "black-forest-labs/FLUX.1-dev"

    unet_config = {
        "image_model": "flux",
        "guidance_embed": True,
    }

    sampling_settings = {}

    unet_extra_config = {}
    latent_format = latent.Flux

    memory_usage_factor = 2.8

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.nunchaku: bool = self.unet_config.pop("nunchaku", False)

    def model_type(self, state_dict):
        return ModelType.FLUX

    def clip_target(self, state_dict: dict):
        result = {}
        pref = self.text_encoder_key_prefix[0]

        if "{}clip_l.transformer.text_model.final_layer_norm.weight".format(pref) in state_dict:
            result["clip_l"] = "text_encoder"

        if "{}t5xxl.transformer.encoder.final_layer_norm.weight".format(pref) in state_dict:
            result["t5xxl"] = "text_encoder_2"

        elif "{}t5xxl.transformer.encoder.final_layer_norm.qweight".format(pref) in state_dict:
            result["t5xxl"] = "text_encoder_2"

        return result


class FluxSchnell(Flux):
    huggingface_repo = "black-forest-labs/FLUX.1-schnell"

    unet_config = {
        "image_model": "flux",
        "guidance_embed": False,
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 1.0,
    }

    supported_inference_dtypes = [torch.bfloat16, torch.float32]


class Chroma(FluxSchnell):
    huggingface_repo = "Chroma"

    unet_config = {
        "image_model": "chroma",
    }

    sampling_settings = {
        "multiplier": 1.0,
    }

    memory_usage_factor = 3.2

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    text_encoder_key_prefix = ["text_encoders.", "cond_stage_model."]

    def clip_target(self, state_dict: dict):
        for pref in self.text_encoder_key_prefix:
            if "{}t5xxl.transformer.encoder.final_layer_norm.weight".format(pref) in state_dict:
                return {"t5xxl": "text_encoder"}
            elif "{}t5xxl.transformer.encoder.final_layer_norm.qweight".format(pref) in state_dict:
                return {"t5xxl": "text_encoder"}

    def process_vae_state_dict(self, state_dict):
        replace_prefix = {"first_stage_model.": "vae."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)


class Lumina2(BASE):
    huggingface_repo = "neta-art/Neta-Lumina"

    unet_config = {
        "image_model": "lumina2",
        "dim": 2304,
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 6.0,
    }

    memory_usage_factor = 1.2

    unet_extra_config = {}
    latent_format = latent.Flux

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def model_type(self, state_dict):
        return ModelType.FLOW

    def clip_target(self, state_dict: dict):
        pref = self.text_encoder_key_prefix[0]
        if "{}gemma2_2b.transformer.model.embed_tokens.weight".format(pref) in state_dict:
            state_dict.pop("{}gemma2_2b.logit_scale".format(pref), None)
            state_dict.pop("{}spiece_model".format(pref), None)
            return {"gemma2_2b.transformer": "text_encoder"}
        else:
            return {"gemma2_2b": "text_encoder"}


class ZImage(Lumina2):
    huggingface_repo = "Tongyi-MAI/Z-Image-Turbo"

    unet_config = {
        "image_model": "lumina2",
        "dim": 3840,
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 3.0,
    }

    memory_usage_factor = 1.7

    def clip_target(self, state_dict={}):
        return {"qwen3_4b.transformer": "text_encoder"}


class WAN21_T2V(BASE):
    huggingface_repo = "Wan-AI/Wan2.1-T2V-14B"

    unet_config = {
        "image_model": "wan2.1",
        "model_type": "t2v",
    }

    sampling_settings = {
        "shift": 8.0,
    }

    unet_extra_config = {}
    latent_format = latent.Wan21

    memory_usage_factor = 1.0

    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.memory_usage_factor = self.unet_config.get("dim", 2000) / 2000

    def model_type(self, state_dict):
        return ModelType.FLOW

    def clip_target(self, state_dict: dict):
        return {"umt5xxl": "text_encoder"}


class WAN21_I2V(WAN21_T2V):
    huggingface_repo = "Wan-AI/Wan2.1-I2V-14B"

    unet_config = {
        "image_model": "wan2.1",
        "model_type": "i2v",
        "in_dim": 36,
    }


class QwenImage(BASE):
    huggingface_repo = "Qwen/Qwen-Image"

    unet_config = {
        "image_model": "qwen_image",
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 1.15,
    }

    memory_usage_factor = 1.8

    unet_extra_config = {}
    latent_format = latent.Wan21

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.nunchaku: bool = self.unet_config.pop("nunchaku", False)

    def model_type(self, state_dict):
        return ModelType.FLOW

    def clip_target(self, state_dict: dict):
        pref = self.text_encoder_key_prefix[0]
        if "{}.qwen25_7b.transformer.model.embed_tokens.weight".format(pref) in state_dict:
            state_dict.pop("{}qwen25_7b.logit_scale".format(pref), None)
            return {"qwen25_7b.transformer": "text_encoder"}
        else:
            return {"qwen25_7b": "text_encoder"}


models = [
    SD15,
    SDXL,
    SDXLRefiner,
    Flux,
    FluxSchnell,
    Chroma,
    Lumina2,
    ZImage,
    WAN21_T2V,
    WAN21_I2V,
    QwenImage,
]
