import glob
import os.path
from copy import deepcopy

import torch

from backend import memory_management, utils
from modules import hashes, paths, sd_models, shared

vae_path = os.path.abspath(os.path.join(paths.models_path, "VAE"))
vae_ignore_keys: set[str] = {"model_ema.decay", "model_ema.num_updates"}
vae_dict: dict[str, os.PathLike] = {}

base_vae: dict[str, torch.Tensor] = None
loaded_vae_file: os.PathLike = None
checkpoint_info: "sd_models.CheckpointInfo" = None


@torch.inference_mode()
def _load_vae_dict(model, vae_sd: dict):
    sd = {k: v for k, v in vae_sd.items() if k[0:4] != "loss" and k not in vae_ignore_keys}
    model.first_stage_model.load_state_dict(sd)


def get_loaded_vae_name() -> str:
    if loaded_vae_file is None:
        return None
    return os.path.basename(loaded_vae_file)


def get_loaded_vae_hash() -> str:
    if loaded_vae_file is None:
        return None

    sha256 = hashes.sha256(loaded_vae_file, "vae")
    return sha256[0:10] if sha256 else None


def store_base_vae(model):
    global base_vae, checkpoint_info
    assert loaded_vae_file is None
    memory_management.logger.debug("Storing Original VAE...")
    base_vae = deepcopy(model.first_stage_model.state_dict())
    checkpoint_info = model.sd_checkpoint_info


def delete_base_vae():
    global base_vae, checkpoint_info
    base_vae = None
    checkpoint_info = None
    memory_management.soft_empty_cache()


def restore_base_vae(model):
    global loaded_vae_file
    if base_vae is None:
        return
    memory_management.logger.debug("Restoring Original VAE...")
    _load_vae_dict(model, base_vae)
    loaded_vae_file = None
    delete_base_vae()


def get_filename(filepath: os.PathLike) -> str:
    return os.path.basename(filepath)


def refresh_vae_list():
    vae_dict.clear()
    paths = []

    file_extensions = ("ckpt", "pt", "pth", "bin", "safetensors", "sft", "gguf")

    for ext in file_extensions:
        paths.append(os.path.join(sd_models.model_path, f"**/*.vae.{ext}"))
        paths.append(os.path.join(vae_path, f"**/*.{ext}"))

    for _dir in shared.cmd_opts.vae_dirs:
        for ext in file_extensions:
            paths.append(os.path.join(_dir, f"**/*.{ext}"))

    candidates = []
    for path in paths:
        candidates += glob.iglob(path, recursive=True)

    for filepath in candidates:
        name = get_filename(filepath)
        vae_dict[name] = filepath

    vae_dict.update(dict(sorted(vae_dict.items(), key=lambda item: shared.natural_sort_key(item[0]))))


def reload_vae_weights(vae: str) -> bool:
    if vae in (None, "None"):
        return False

    store_base_vae(shared.sd_model)
    vae_sd = utils.load_torch_file(vae)
    _load_vae_dict(shared.sd_model, vae_sd)
    return True


def restore_vae_weights():
    restore_base_vae(shared.sd_model)
