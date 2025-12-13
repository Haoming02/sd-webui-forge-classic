# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.77/comfy/lora.py

"""
This file is part of ComfyUI.
Copyright (C) 2024 Comfy

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch

from modules_forge.packages.comfy import weight_adapter

from .utils import flux_to_diffusers, unet_to_diffusers, z_image_to_diffusers

LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}


def load_lora(lora, to_load):

    def convert_lora_bfl_control(sd):  # BFL loras for Flux
        sd_out = {}
        for k in sd:
            k_to = "diffusion_model.{}".format(k.replace(".lora_B.bias", ".diff_b").replace("_norm.scale", "_norm.scale.set_weight"))
            sd_out[k_to] = sd[k]

        sd_out["diffusion_model.img_in.reshape_weight"] = torch.tensor([sd["img_in.lora_B.weight"].shape[0], sd["img_in.lora_A.weight"].shape[1]])
        return sd_out

    if "img_in.lora_A.weight" in lora and "single_blocks.0.norm.key_norm.scale" in lora:
        lora = convert_lora_bfl_control(lora)

    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        alpha_name = "{}.alpha".format(x)
        alpha = None
        if alpha_name in lora.keys():
            alpha = lora[alpha_name].item()
            loaded_keys.add(alpha_name)

        dora_scale_name = "{}.dora_scale".format(x)
        dora_scale = None
        if dora_scale_name in lora.keys():
            dora_scale = lora[dora_scale_name]
            loaded_keys.add(dora_scale_name)

        for adapter_cls in weight_adapter.adapters:
            adapter = adapter_cls.load(x, lora, alpha, dora_scale, loaded_keys)
            if adapter is not None:
                patch_dict[to_load[x]] = adapter
                loaded_keys.update(adapter.loaded_keys)
                continue

        w_norm_name = "{}.w_norm".format(x)
        b_norm_name = "{}.b_norm".format(x)
        w_norm = lora.get(w_norm_name, None)
        b_norm = lora.get(b_norm_name, None)

        if w_norm is not None:
            loaded_keys.add(w_norm_name)
            patch_dict[to_load[x]] = ("diff", (w_norm,))
            if b_norm is not None:
                loaded_keys.add(b_norm_name)
                patch_dict["{}.bias".format(to_load[x][: -len(".weight")])] = ("diff", (b_norm,))

        diff_name = "{}.diff".format(x)
        diff_weight = lora.get(diff_name, None)
        if diff_weight is not None:
            patch_dict[to_load[x]] = ("diff", (diff_weight,))
            loaded_keys.add(diff_name)

        diff_bias_name = "{}.diff_b".format(x)
        diff_bias = lora.get(diff_bias_name, None)
        if diff_bias is not None:
            patch_dict["{}.bias".format(to_load[x][: -len(".weight")])] = ("diff", (diff_bias,))
            loaded_keys.add(diff_bias_name)

        set_weight_name = "{}.set_weight".format(x)
        set_weight = lora.get(set_weight_name, None)
        if set_weight is not None:
            patch_dict[to_load[x]] = ("set", (set_weight,))
            loaded_keys.add(set_weight_name)

    remaining_dict = {x: y for x, y in lora.items() if x not in loaded_keys}
    return patch_dict, remaining_dict


def model_lora_keys_clip(model, key_map={}):
    sdk = model.state_dict().keys()
    for k in sdk:
        if k.endswith(".weight"):
            key_map["text_encoders.{}".format(k[: -len(".weight")])] = k  # generic lora format without any weird key names

    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    clip_l_present = False
    clip_g_present = False
    for b in range(32):  # TODO: clean up
        for c in LORA_CLIP_MAP:
            k = "clip_h.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c)  # diffusers lora
                key_map[lora_key] = k

            k = "clip_l.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c])  # SDXL base
                key_map[lora_key] = k
                clip_l_present = True
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c)  # diffusers lora
                key_map[lora_key] = k

            k = "clip_g.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                clip_g_present = True
                if clip_l_present:
                    lora_key = "lora_te2_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c])  # SDXL base
                    key_map[lora_key] = k
                    lora_key = "text_encoder_2.text_model.encoder.layers.{}.{}".format(b, c)  # diffusers lora
                    key_map[lora_key] = k
                else:
                    lora_key = "lora_te_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c])  # TODO: test if this is correct for SDXL-Refiner
                    key_map[lora_key] = k
                    lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c)  # diffusers lora
                    key_map[lora_key] = k
                    lora_key = "lora_prior_te_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c])  # cascade lora: TODO put lora key prefix in the model config
                    key_map[lora_key] = k

    for k in sdk:
        if k.endswith(".weight") and k.startswith("t5xxl.transformer."):  # OneTrainer SD3 and Flux lora
            l_key = k[len("t5xxl.transformer.") : -len(".weight")]
            t5_index = 1
            if clip_g_present:
                t5_index += 1
            if clip_l_present:
                t5_index += 1
                if t5_index == 2:
                    key_map["lora_te{}_{}".format(t5_index, l_key.replace(".", "_"))] = k  # OneTrainer Flux
                    t5_index += 1

            key_map["lora_te{}_{}".format(t5_index, l_key.replace(".", "_"))] = k

    return key_map


def model_lora_keys_unet(model, key_map={}):
    sd = model.state_dict()
    sdk = sd.keys()

    for k in sdk:
        if k.startswith("diffusion_model."):
            if k.endswith(".weight"):
                key_lora = k[len("diffusion_model.") : -len(".weight")].replace(".", "_")
                key_map["lora_unet_{}".format(key_lora)] = k
                key_map["{}".format(k[: -len(".weight")])] = k  # generic lora format without any weird key names
            else:
                key_map["{}".format(k)] = k  # generic lora format for not .weight without any weird key names

    diffusers_keys = unet_to_diffusers(model.diffusion_model.config)
    for k in diffusers_keys:
        if k.endswith(".weight"):
            unet_key = "diffusion_model.{}".format(diffusers_keys[k])
            key_lora = k[: -len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = unet_key
            key_map["lycoris_{}".format(key_lora)] = unet_key  # simpletuner lycoris format

            diffusers_lora_prefix = ["", "unet."]
            for p in diffusers_lora_prefix:
                diffusers_lora_key = "{}{}".format(p, k[: -len(".weight")].replace(".to_", ".processor.to_"))
                if diffusers_lora_key.endswith(".to_out.0"):
                    diffusers_lora_key = diffusers_lora_key[:-2]
                key_map[diffusers_lora_key] = unet_key

    _model_name: str = model.config.huggingface_repo.lower()

    if "flux" in _model_name or "chroma" in _model_name:  # Diffusers lora Flux
        diffusers_keys = flux_to_diffusers(model.diffusion_model.config, output_prefix="diffusion_model.")
        for k in diffusers_keys:
            if k.endswith(".weight"):
                to = diffusers_keys[k]
                key_map["transformer.{}".format(k[: -len(".weight")])] = to  # simpletrainer and probably regular diffusers flux lora format
                key_map["lycoris_{}".format(k[: -len(".weight")].replace(".", "_"))] = to  # simpletrainer lycoris
                key_map["lora_transformer_{}".format(k[: -len(".weight")].replace(".", "_"))] = to  # onetrainer
        for k in sdk:
            hidden_size = model.diffusion_model.config.get("hidden_size", 0)
            if k.endswith(".weight") and ".linear1." in k:
                key_map["{}".format(k.replace(".linear1.weight", ".linear1_qkv"))] = (k, (0, 0, hidden_size * 3))

    if "qwen" in _model_name:
        for k in sdk:
            if k.startswith("diffusion_model.") and k.endswith(".weight"):  # QwenImage lora format
                key_lora = k[len("diffusion_model.") : -len(".weight")]
                # Direct mapping for transformer_blocks format (QwenImage LoRA format)
                key_map["{}".format(key_lora)] = k
                # Support transformer prefix format
                key_map["transformer.{}".format(key_lora)] = k
                key_map["lycoris_{}".format(key_lora.replace(".", "_"))] = k  # SimpleTuner lycoris format

    if "lumina" in _model_name or "z-image" in _model_name:
        diffusers_keys = z_image_to_diffusers(model.diffusion_model.config, output_prefix="diffusion_model.")
        for k in diffusers_keys:
            if k.endswith(".weight"):
                to = diffusers_keys[k]
                key_lora = k[: -len(".weight")]
                key_map["diffusion_model.{}".format(key_lora)] = to
                key_map["lycoris_{}".format(key_lora.replace(".", "_"))] = to

    return key_map
