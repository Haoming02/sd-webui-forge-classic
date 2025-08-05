# 1st Edit by. https://github.com/comfyanonymous/ComfyUI
# 2nd Edit by. Forge Official
# 3rd Edit by. Haoming02
# - Based on: https://github.com/comfyanonymous/ComfyUI/blob/v0.3.29/comfy/lora.py

import torch

import ldm_patched.modules.model_management
import ldm_patched.modules.utils

LEN_WEIGHT = len(".weight")
LEN_DIFFUSION_MODEL = len("diffusion_model.")

LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}


def load_lora(lora, known_keys, to_load, do_referential_load):
    patch_dict = {}
    loaded_keys = set()
    new_keys = set()
    remaining_dict = lora

    if do_referential_load:
        new_lora_keys = set()
        for key in known_keys:
            load_lora_key(key, to_load[key], remaining_dict, patch_dict, new_lora_keys)
            if (len(new_lora_keys) > 0):
                loaded_keys.update(new_lora_keys)
                for nk in new_lora_keys:
                    del remaining_dict[nk]
                new_lora_keys = set()

    if (len(remaining_dict) > 0):
        new_lora_keys = set()
        for key in to_load:
            if key not in known_keys:
                load_lora_key(key, to_load[key], remaining_dict, patch_dict, new_lora_keys)
                if (len(new_lora_keys) > 0):
                    if do_referential_load:
                        new_keys.add(key)
                    loaded_keys.update(new_lora_keys)
                    for nk in new_lora_keys:
                        del remaining_dict[nk]
                    new_lora_keys = set()

    return patch_dict, remaining_dict, new_keys


def load_lora_key(key, to_load_value_for_key, lora, patch_dict, loaded_keys):
    alpha_name = "{}.alpha".format(key)
    alpha = None
    if alpha_name in lora.keys():
        alpha = lora[alpha_name].item()
        loaded_keys.add(alpha_name)

    dora_scale_name = "{}.dora_scale".format(key)
    dora_scale = None
    if dora_scale_name in lora.keys():
        dora_scale = lora[dora_scale_name]
        loaded_keys.add(dora_scale_name)

    reshape_name = "{}.reshape_weight".format(key)
    reshape = None
    if reshape_name in lora.keys():
        try:
            reshape = lora[reshape_name].tolist()
            loaded_keys.add(reshape_name)
        except Exception:
            pass

    regular_lora = "{}.lora_up.weight".format(key)
    diffusers_lora = "{}_lora.up.weight".format(key)
    diffusers2_lora = "{}.lora_B.weight".format(key)
    diffusers3_lora = "{}.lora.up.weight".format(key)
    mochi_lora = "{}.lora_B".format(key)
    transformers_lora = "{}.lora_linear_layer.up.weight".format(key)
    A_name = None

    if regular_lora in lora.keys():
        A_name = regular_lora
        B_name = "{}.lora_down.weight".format(key)
        mid_name = "{}.lora_mid.weight".format(key)
    elif diffusers_lora in lora.keys():
        A_name = diffusers_lora
        B_name = "{}_lora.down.weight".format(key)
        mid_name = None
    elif diffusers2_lora in lora.keys():
        A_name = diffusers2_lora
        B_name = "{}.lora_A.weight".format(key)
        mid_name = None
    elif diffusers3_lora in lora.keys():
        A_name = diffusers3_lora
        B_name = "{}.lora.down.weight".format(key)
        mid_name = None
    elif mochi_lora in lora.keys():
        A_name = mochi_lora
        B_name = "{}.lora_A".format(key)
        mid_name = None
    elif transformers_lora in lora.keys():
        A_name = transformers_lora
        B_name = "{}.lora_linear_layer.down.weight".format(key)
        mid_name = None

    if A_name is not None:
        mid = None
        if mid_name is not None and mid_name in lora.keys():
            mid = lora[mid_name]
            loaded_keys.add(mid_name)
        patch_dict[to_load_value_for_key] = ("lora", (lora[A_name], lora[B_name], alpha, mid, dora_scale, reshape))
        loaded_keys.add(A_name)
        loaded_keys.add(B_name)

    ######## loha
    hada_w1_a_name = "{}.hada_w1_a".format(key)
    hada_w1_b_name = "{}.hada_w1_b".format(key)
    hada_w2_a_name = "{}.hada_w2_a".format(key)
    hada_w2_b_name = "{}.hada_w2_b".format(key)
    hada_t1_name = "{}.hada_t1".format(key)
    hada_t2_name = "{}.hada_t2".format(key)
    if hada_w1_a_name in lora.keys():
        hada_t1 = None
        hada_t2 = None
        if hada_t1_name in lora.keys():
            hada_t1 = lora[hada_t1_name]
            hada_t2 = lora[hada_t2_name]
            loaded_keys.add(hada_t1_name)
            loaded_keys.add(hada_t2_name)

        patch_dict[to_load_value_for_key] = (
            "loha",
            (
                lora[hada_w1_a_name],
                lora[hada_w1_b_name],
                alpha,
                lora[hada_w2_a_name],
                lora[hada_w2_b_name],
                hada_t1,
                hada_t2,
                dora_scale,
            ),
        )
        loaded_keys.add(hada_w1_a_name)
        loaded_keys.add(hada_w1_b_name)
        loaded_keys.add(hada_w2_a_name)
        loaded_keys.add(hada_w2_b_name)

    ######## lokr
    lokr_w1_name = "{}.lokr_w1".format(key)
    lokr_w2_name = "{}.lokr_w2".format(key)
    lokr_w1_a_name = "{}.lokr_w1_a".format(key)
    lokr_w1_b_name = "{}.lokr_w1_b".format(key)
    lokr_t2_name = "{}.lokr_t2".format(key)
    lokr_w2_a_name = "{}.lokr_w2_a".format(key)
    lokr_w2_b_name = "{}.lokr_w2_b".format(key)

    lokr_w1 = None
    if lokr_w1_name in lora.keys():
        lokr_w1 = lora[lokr_w1_name]
        loaded_keys.add(lokr_w1_name)

    lokr_w2 = None
    if lokr_w2_name in lora.keys():
        lokr_w2 = lora[lokr_w2_name]
        loaded_keys.add(lokr_w2_name)

    lokr_w1_a = None
    if lokr_w1_a_name in lora.keys():
        lokr_w1_a = lora[lokr_w1_a_name]
        loaded_keys.add(lokr_w1_a_name)

    lokr_w1_b = None
    if lokr_w1_b_name in lora.keys():
        lokr_w1_b = lora[lokr_w1_b_name]
        loaded_keys.add(lokr_w1_b_name)

    lokr_w2_a = None
    if lokr_w2_a_name in lora.keys():
        lokr_w2_a = lora[lokr_w2_a_name]
        loaded_keys.add(lokr_w2_a_name)

    lokr_w2_b = None
    if lokr_w2_b_name in lora.keys():
        lokr_w2_b = lora[lokr_w2_b_name]
        loaded_keys.add(lokr_w2_b_name)

    lokr_t2 = None
    if lokr_t2_name in lora.keys():
        lokr_t2 = lora[lokr_t2_name]
        loaded_keys.add(lokr_t2_name)

    if (lokr_w1 is not None) or (lokr_w2 is not None) or (lokr_w1_a is not None) or (lokr_w2_a is not None):
        patch_dict[to_load_value_for_key] = (
            "lokr",
            (lokr_w1, lokr_w2, alpha, lokr_w1_a, lokr_w1_b, lokr_w2_a, lokr_w2_b, lokr_t2, dora_scale),
        )

    # glora
    a1_name = "{}.a1.weight".format(key)
    a2_name = "{}.a2.weight".format(key)
    b1_name = "{}.b1.weight".format(key)
    b2_name = "{}.b2.weight".format(key)
    if a1_name in lora:
        patch_dict[to_load_value_for_key] = ("glora", (lora[a1_name], lora[a2_name], lora[b1_name], lora[b2_name], alpha, dora_scale))
        loaded_keys.add(a1_name)
        loaded_keys.add(a2_name)
        loaded_keys.add(b1_name)
        loaded_keys.add(b2_name)

    w_norm_name = "{}.w_norm".format(key)
    b_norm_name = "{}.b_norm".format(key)
    w_norm = lora.get(w_norm_name, None)
    b_norm = lora.get(b_norm_name, None)

    if w_norm is not None:
        loaded_keys.add(w_norm_name)
        patch_dict[to_load_value_for_key] = ("diff", (w_norm,))
        if b_norm is not None:
            loaded_keys.add(b_norm_name)
            patch_dict["{}.bias".format(to_load_value_for_key[: -LEN_WEIGHT])] = ("diff", (b_norm,))

    diff_name = "{}.diff".format(key)
    diff_weight = lora.get(diff_name, None)
    if diff_weight is not None:
        patch_dict[to_load_value_for_key] = ("diff", (diff_weight,))
        loaded_keys.add(diff_name)

    diff_bias_name = "{}.diff_b".format(key)
    diff_bias = lora.get(diff_bias_name, None)
    if diff_bias is not None:
        patch_dict["{}.bias".format(to_load_value_for_key[: -LEN_WEIGHT])] = ("diff", (diff_bias,))
        loaded_keys.add(diff_bias_name)

    set_weight_name = "{}.set_weight".format(key)
    set_weight = lora.get(set_weight_name, None)
    if set_weight is not None:
        patch_dict[to_load_value_for_key] = ("set", (set_weight,))
        loaded_keys.add(set_weight_name)



def model_lora_keys_clip(model, key_map={}):
    sdk = model.state_dict().keys()
    for k in sdk:
        if k.endswith(".weight"):
            key_map["text_encoders.{}".format(k[: -LEN_WEIGHT])] = k  # generic lora format without any weird key names

    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    clip_l_present = False

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

    return key_map


def model_lora_keys_unet(model, key_map={}):
    sd = model.state_dict()
    sdk = sd.keys()

    for k in sdk:
        if k.startswith("diffusion_model."):
            if k.endswith(".weight"):
                key_lora = k[LEN_DIFFUSION_MODEL : -LEN_WEIGHT].replace(".", "_")
                key_map["lora_unet_{}".format(key_lora)] = k
                key_map["{}".format(k[: -LEN_WEIGHT])] = k  # generic lora format without any weird key names
            else:
                key_map["{}".format(k)] = k  # generic lora format for not .weight without any weird key names

    diffusers_keys = ldm_patched.modules.utils.unet_to_diffusers(model.model_config.unet_config)
    for k in diffusers_keys:
        if k.endswith(".weight"):
            unet_key = "diffusion_model.{}".format(diffusers_keys[k])
            key_lora = k[: -LEN_WEIGHT].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = unet_key
            key_map["lycoris_{}".format(key_lora)] = unet_key  # simpletuner lycoris format

            diffusers_lora_prefix = ["", "unet."]
            for p in diffusers_lora_prefix:
                diffusers_lora_key = "{}{}".format(p, k[: -LEN_WEIGHT].replace(".to_", ".processor.to_"))
                if diffusers_lora_key.endswith(".to_out.0"):
                    diffusers_lora_key = diffusers_lora_key[:-2]
                key_map[diffusers_lora_key] = unet_key

    return key_map


def weight_decompose(dora_scale: torch.Tensor, weight: torch.Tensor, lora_diff: torch.Tensor, alpha: float, strength: float, intermediate_dtype: torch.dtype) -> torch.Tensor:
    dora_scale = ldm_patched.modules.model_management.cast_to_device(dora_scale, weight.device, intermediate_dtype)
    lora_diff *= alpha
    weight_calc = weight + lora_diff.type(weight.dtype)
    weight_norm = weight_calc.transpose(0, 1).reshape(weight_calc.shape[1], -1).norm(dim=1, keepdim=True).reshape(weight_calc.shape[1], *[1] * (weight_calc.dim() - 1)).transpose(0, 1)

    weight_calc *= (dora_scale / weight_norm).type(weight.dtype)
    if strength != 1.0:
        weight_calc -= weight
        weight += strength * (weight_calc)
    else:
        weight[:] = weight_calc
    return weight


def pad_tensor_to_shape(tensor: torch.Tensor, new_shape: list[int]) -> torch.Tensor:
    if any([new_shape[i] < tensor.shape[i] for i in range(len(new_shape))]):
        raise ValueError("The new shape must be larger than the original tensor in all dimensions")
    if len(new_shape) != len(tensor.shape):
        raise ValueError("The new shape must have the same number of dimensions as the original tensor")

    padded_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)

    orig_slices = tuple(slice(0, dim) for dim in tensor.shape)
    new_slices = tuple(slice(0, dim) for dim in tensor.shape)

    padded_tensor[new_slices] = tensor[orig_slices]
    return padded_tensor
