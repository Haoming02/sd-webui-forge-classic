import json
import os

import gguf
import safetensors
import torch
from einops import rearrange, repeat

from backend.args import args
from backend.operations_gguf import ParameterGGUF
from modules import safe

MMAP_TORCH_FILES = args.mmap_torch_files
DISABLE_MMAP = args.disable_mmap


def read_arbitrary_config(directory):
    config_path = os.path.join(directory, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json file found in the directory: {directory}")

    with open(config_path, "rt", encoding="utf-8") as file:
        config_data = json.load(file)

    return config_data


def load_torch_file(ckpt: str, safe_load=False, device=None, *, return_metadata=False):
    """https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/utils.py#L53"""
    if device is None:
        device = torch.device("cpu")

    metadata = None
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        try:
            with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
                sd = {}
                for k in f.keys():
                    tensor = f.get_tensor(k)
                    if DISABLE_MMAP:
                        tensor = tensor.to(device=device, copy=True)
                    sd[k] = tensor
                if return_metadata:
                    metadata = f.metadata()
        except Exception as e:
            if len(e.args) > 0:
                if "HeaderTooLarge" in e.args[0] or "MetadataIncompleteBuffer" in e.args[0]:
                    raise ValueError(f'\nModel: "{ckpt}" is corrupt or invalid...\nPlease download the model again')
            raise e

    elif ckpt.lower().endswith(".gguf"):
        reader = gguf.GGUFReader(ckpt)
        sd = {}
        for tensor in reader.tensors:
            sd[str(tensor.name)] = ParameterGGUF(tensor)

    else:
        torch_args = {}

        if not safe_load:
            torch_args["pickle_module"] = safe
        else:
            torch_args["weights_only"] = True
            if MMAP_TORCH_FILES:
                torch_args["mmap"] = True

        pl_sd = torch.load(ckpt, map_location=device, **torch_args)

        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            if len(pl_sd) == 1:
                key = list(pl_sd.keys())[0]
                sd = pl_sd[key]
                if not isinstance(sd, dict):
                    sd = pl_sd
            else:
                sd = pl_sd

    return (sd, metadata) if return_metadata else sd


def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], torch.nn.Parameter(value, requires_grad=False))


def set_attr_raw(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], value)


def copy_to_param(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)


def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def get_attr_with_parent(obj, attr):
    attrs = attr.split(".")
    parent = obj
    name = None
    for name in attrs:
        parent = obj
        obj = getattr(obj, name)
    return parent, name, obj


def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            params += sd[k].nelement()
    return params


def tensor2parameter(x):
    if isinstance(x, torch.nn.Parameter):
        return x
    else:
        return torch.nn.Parameter(x, requires_grad=False)


def fp16_fix(x):
    # avoid fp16 overflow
    # https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/ldm/chroma/layers.py#L111

    if x.dtype == torch.float16:
        return torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


def dtype_to_element_size(dtype):
    if isinstance(dtype, torch.dtype):
        return torch.tensor([], dtype=dtype).element_size()
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


def nested_compute_size(obj, element_size):
    module_mem = 0

    if isinstance(obj, dict):
        for key in obj:
            module_mem += nested_compute_size(obj[key], element_size)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for i in range(len(obj)):
            module_mem += nested_compute_size(obj[i], element_size)
    elif isinstance(obj, torch.Tensor):
        module_mem += obj.nelement() * element_size

    return module_mem


def nested_move_to_device(obj, **kwargs):
    if isinstance(obj, dict):
        for key in obj:
            obj[key] = nested_move_to_device(obj[key], **kwargs)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = nested_move_to_device(obj[i], **kwargs)
    elif isinstance(obj, tuple):
        obj = tuple(nested_move_to_device(i, **kwargs) for i in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.to(**kwargs)
    return obj


def get_state_dict_after_quant(model, prefix=""):
    for m in model.modules():
        if hasattr(m, "weight") and hasattr(m.weight, "bnb_quantized"):
            if not m.weight.bnb_quantized:
                original_device = m.weight.device
                m.cuda()
                m.to(original_device)

    sd = model.state_dict()
    sd = {(prefix + k): v.clone() for k, v in sd.items()}
    return sd


def beautiful_print_gguf_state_dict_statics(state_dict):
    type_counts = {}
    for k, v in state_dict.items():
        gguf_cls = getattr(v, "gguf_cls", None)
        if gguf_cls is not None:
            type_name = gguf_cls.__name__
            if type_name in type_counts:
                type_counts[type_name] += 1
            else:
                type_counts[type_name] = 1
    print(f"GGUF state dict: {type_counts}")
    return


def pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
    """https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/ldm/common_dit.py#L5"""
    if padding_mode == "circular" and (torch.jit.is_tracing() or torch.jit.is_scripting()):
        padding_mode = "reflect"

    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad

    return torch.nn.functional.pad(img, pad, mode=padding_mode)


def process_img(x, index=0, h_offset=0, w_offset=0):
    """https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/ldm/flux/model.py#L213"""
    bs, c, h, w = x.shape
    patch_size = 2
    x = pad_to_patch_size(x, (patch_size, patch_size))

    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
    h_len = (h + (patch_size // 2)) // patch_size
    w_len = (w + (patch_size // 2)) // patch_size

    h_offset = (h_offset + (patch_size // 2)) // patch_size
    w_offset = (w_offset + (patch_size // 2)) // patch_size

    img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
    img_ids[:, :, 0] = img_ids[:, :, 1] + index
    img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(h_offset, h_len - 1 + h_offset, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
    img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(w_offset, w_len - 1 + w_offset, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
    return img, repeat(img_ids, "h w c -> b (h w) c", b=bs)


def join_dicts(base_dict: dict | None, update_dict: dict | None) -> dict:
    if not update_dict:
        return (base_dict or {}).copy()

    result = (base_dict or {}).copy()

    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = join_dicts(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            result[key] = result[key] + value
        else:
            result[key] = value

    return result


def hash_tensor(x: torch.Tensor) -> int:
    if hasattr(torch, "hash_tensor"):
        return torch.hash_tensor(x).item()
    else:
        return hash(tuple(x.reshape(-1).tolist()))
