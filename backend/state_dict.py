import json

import torch


def load_state_dict(model, sd, ignore_errors=[], log_name=None, ignore_start=None):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [x for x in missing if x not in ignore_errors]
    unexpected = [x for x in unexpected if x not in ignore_errors]

    if isinstance(ignore_start, str):
        missing = [x for x in missing if not x.startswith(ignore_start)]
        unexpected = [x for x in unexpected if not x.startswith(ignore_start)]

    log_name = log_name or type(model).__name__
    if len(missing) > 0:
        print(f"{log_name} Missing: {missing}")
    if len(unexpected) > 0:
        print(f"{log_name} Unexpected: {unexpected}")


def state_dict_has(sd, prefix):
    return any(x.startswith(prefix) for x in sd.keys())


def filter_state_dict_with_prefix(sd, prefix, new_prefix=""):
    new_sd = {}

    for k, v in list(sd.items()):
        if k.startswith(prefix):
            new_sd[new_prefix + k[len(prefix) :]] = v
            del sd[k]

    return new_sd


def try_filter_state_dict(sd, prefix_list, new_prefix=""):
    for prefix in prefix_list:
        if state_dict_has(sd, prefix):
            return filter_state_dict_with_prefix(sd, prefix, new_prefix)
    return {}


def transformers_convert(sd, prefix_from, prefix_to, number):
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in sd:
            sd[keys_to_replace[k].format(prefix_to)] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    sd[k_to] = weights[shape_from * x : shape_from * (x + 1)]
    return sd


def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp) :])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def detect_quantization(state_dict: dict[str, torch.Tensor], *, is_unet: bool = False) -> dict | None:
    if any(k.endswith(".comfy_quant") for k in state_dict):
        return {"mixed_ops": True, "TE": not is_unet}
    return None


def convert_quantization(state_dict: dict[str, torch.Tensor], metadata: dict) -> dict[str, torch.Tensor]:
    # https://github.com/Comfy-Org/ComfyUI/blob/v0.19.0/comfy/utils.py#L1358
    if metadata is None:
        metadata = {}

    if "_quantization_metadata" in metadata:
        quant_metadata = json.loads(metadata["_quantization_metadata"]) or {}
    else:
        model_prefix = None

        for key in state_dict.keys():
            if key.endswith("scaled_fp8"):
                model_prefix = key.replace("scaled_fp8", "")
                break

        if model_prefix is None:
            return state_dict, metadata

        scaled_fp8_key = "{}scaled_fp8".format(model_prefix)
        scaled_fp8_weight = state_dict[scaled_fp8_key]
        scaled_fp8_dtype = scaled_fp8_weight.dtype
        if scaled_fp8_dtype is torch.float32:
            scaled_fp8_dtype = torch.float8_e4m3fn

        if scaled_fp8_weight.nelement() == 2:
            full_precision_matrix_mult = True
        else:
            full_precision_matrix_mult = False

        out_sd = {}
        layers = {}
        for k in list(state_dict.keys()):
            if k == scaled_fp8_key:
                continue
            if not k.startswith(model_prefix):
                out_sd[k] = state_dict[k]
                continue
            k_out = k
            w = state_dict.pop(k)
            layer = None
            if k_out.endswith(".scale_weight"):
                layer = k_out[: -len(".scale_weight")]
                k_out = "{}.weight_scale".format(layer)

            if layer is not None:
                layer_conf = {"format": "float8_e4m3fn"}
                if full_precision_matrix_mult:
                    layer_conf["full_precision_matrix_mult"] = full_precision_matrix_mult
                layers[layer] = layer_conf

            if k_out.endswith(".scale_input"):
                layer = k_out[: -len(".scale_input")]
                k_out = "{}.input_scale".format(layer)
                if w.item() == 1.0:
                    continue

            out_sd[k_out] = w

        state_dict = out_sd
        quant_metadata = {"layers": layers}

    if layers := quant_metadata.get("layers", None):
        for k, v in layers.items():
            state_dict["{}.comfy_quant".format(k)] = torch.tensor(list(json.dumps(v).encode("utf-8")), dtype=torch.uint8)

    return state_dict, metadata
