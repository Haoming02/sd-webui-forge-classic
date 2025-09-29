import collections
import gc
import os.path
import re
import sys
import threading

import safetensors.torch
import torch
from ldm_patched.modules import model_management
from modules import cache, devices, errors, extra_networks, hashes, modelloader, paths, processing, script_callbacks, sd_vae, shared
from modules.timer import Timer
from modules_forge import forge_loader

model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))

checkpoints_list = {}
checkpoint_aliases = {}
checkpoints_loaded = collections.OrderedDict()


def replace_key(d, key, new_key, value):
    keys = list(d.keys())

    d[new_key] = value

    if key not in keys:
        return d

    index = keys.index(key)
    keys[index] = new_key

    new_d = {k: d[k] for k in keys}

    d.clear()
    d.update(new_d)
    return d


class CheckpointInfo:
    def __init__(self, filename):
        self.filename = filename
        abspath = os.path.abspath(filename)
        abs_ckpt_dir = os.path.abspath(shared.cmd_opts.ckpt_dir) if shared.cmd_opts.ckpt_dir is not None else None

        self.is_safetensors = os.path.splitext(filename)[1].lower() == ".safetensors"

        if abs_ckpt_dir and abspath.startswith(abs_ckpt_dir):
            name = abspath.replace(abs_ckpt_dir, "")
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, "")
        else:
            name = os.path.basename(filename)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        def read_metadata():
            metadata = read_metadata_from_safetensors(filename)
            self.modelspec_thumbnail = metadata.pop("modelspec.thumbnail", None)

            return metadata

        self.metadata = {}
        if self.is_safetensors:
            try:
                self.metadata = cache.cached_data_for_file("safetensors-metadata", "/".join(["checkpoint", name]), filename, read_metadata)
            except Exception as e:
                errors.display(e, f"reading metadata for {filename}")

        self.name = name
        self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
        self.model_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        self.hash = model_hash(filename)

        self.sha256 = hashes.sha256_from_cache(self.filename, "/".join(["checkpoint", name]))
        self.shorthash = self.sha256[0:10] if self.sha256 else None

        self.title = name if self.shorthash is None else f"{name} [{self.shorthash}]"
        self.short_title = self.name_for_extra if self.shorthash is None else f"{self.name_for_extra} [{self.shorthash}]"

        self.ids = [self.hash, self.model_name, self.title, name, self.name_for_extra, f"{name} [{self.hash}]"]
        if self.shorthash:
            self.ids += [self.shorthash, self.sha256, f"{self.name} [{self.shorthash}]", f"{self.name_for_extra} [{self.shorthash}]"]

    def register(self):
        checkpoints_list[self.title] = self
        for id in self.ids:
            checkpoint_aliases[id] = self

    def calculate_shorthash(self):
        self.sha256 = hashes.sha256(self.filename, "/".join(["checkpoint", self.name]))
        if self.sha256 is None:
            return

        shorthash = self.sha256[0:10]
        if self.shorthash == self.sha256[0:10]:
            return self.shorthash

        self.shorthash = shorthash

        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f"{self.name} [{self.shorthash}]", f"{self.name_for_extra} [{self.shorthash}]"]

        old_title = self.title
        self.title = f"{self.name} [{self.shorthash}]"
        self.short_title = f"{self.name_for_extra} [{self.shorthash}]"

        replace_key(checkpoints_list, old_title, self.title, self)
        self.register()

        return self.shorthash


try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import CLIPModel, logging  # noqa: F401

    logging.set_verbosity_error()
except Exception:
    pass


def setup_model():
    os.makedirs(model_path, exist_ok=True)


def checkpoint_tiles(use_short=False):
    return [x.short_title if use_short else x.title for x in checkpoints_list.values()]


def list_models():
    checkpoints_list.clear()
    checkpoint_aliases.clear()

    cmd_ckpt = shared.cmd_opts.ckpt
    if shared.cmd_opts.no_download_sd_model or cmd_ckpt != shared.sd_model_file or os.path.exists(cmd_ckpt):
        model_url = None
    else:
        model_url = "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/realisticVisionV51_v51VAE.safetensors"

    model_list = modelloader.load_models(model_path=model_path, model_url=model_url, command_path=shared.cmd_opts.ckpt_dir, ext_filter=[".ckpt", ".safetensors"], download_name="realisticVisionV51_v51VAE.safetensors", ext_blacklist=[".vae.ckpt", ".vae.safetensors"])

    if os.path.exists(cmd_ckpt):
        checkpoint_info = CheckpointInfo(cmd_ckpt)
        checkpoint_info.register()
        shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
    elif cmd_ckpt is not None and cmd_ckpt != shared.default_sd_model_file:
        print(f"Checkpoint in --ckpt argument not found (Possible it was moved to {model_path}: {cmd_ckpt}", file=sys.stderr)

    for filename in model_list:
        checkpoint_info = CheckpointInfo(filename)
        checkpoint_info.register()


re_strip_checksum = re.compile(r"\s*\[[^]]+]\s*$")


def get_closet_checkpoint_match(search_string):
    if not search_string:
        return None

    checkpoint_info = checkpoint_aliases.get(search_string, None)
    if checkpoint_info is not None:
        return checkpoint_info

    found = sorted([info for info in checkpoints_list.values() if search_string in info.title], key=lambda x: len(x.title))
    if found:
        return found[0]

    search_string_without_checksum = re.sub(re_strip_checksum, "", search_string)
    found = sorted([info for info in checkpoints_list.values() if search_string_without_checksum in info.title], key=lambda x: len(x.title))
    if found:
        return found[0]

    return None


model_hash = hashes.partial_hash_from_cache
"""old hash that only looks at a small part of the file and is prone to collisions"""


def select_checkpoint():
    """Raises `FileNotFoundError` if no checkpoints are found."""
    model_checkpoint = shared.opts.sd_model_checkpoint

    checkpoint_info = checkpoint_aliases.get(model_checkpoint, None)
    if checkpoint_info is not None:
        return checkpoint_info

    if len(checkpoints_list) == 0:
        error_message = "No checkpoints found. When searching for checkpoints, looked at:"
        if shared.cmd_opts.ckpt is not None:
            error_message += f"\n - file {os.path.abspath(shared.cmd_opts.ckpt)}"
        error_message += f"\n - directory {model_path}"
        if shared.cmd_opts.ckpt_dir is not None:
            error_message += f"\n - directory {os.path.abspath(shared.cmd_opts.ckpt_dir)}"
        error_message += "Can't run without a checkpoint. Find and place a .ckpt or .safetensors file into any of those locations."
        raise FileNotFoundError(error_message)

    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        print(f"Checkpoint {model_checkpoint} not found; loading fallback {checkpoint_info.title}", file=sys.stderr)

    return checkpoint_info


checkpoint_dict_replacements_sd1 = {
    "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
    "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
    "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
}


def transform_checkpoint_dict_key(k, replacements):
    for text, replacement in replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text) :]

    return k


def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k, checkpoint_dict_replacements_sd1)
        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd


def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"
        json_data = json_start + file.read(metadata_len - 2)
        json_obj = json.loads(json_data)

        res = {}
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            if isinstance(v, str) and v[0:1] == "{":
                try:
                    res[k] = json.loads(v)
                except Exception:
                    pass

        return res


def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
    _, extension = os.path.splitext(checkpoint_file)
    if extension.lower() == ".safetensors":
        device = map_location or shared.weight_load_location or devices.get_optimal_device_name()

        if not shared.opts.disable_mmap_load_safetensors:
            pl_sd = safetensors.torch.load_file(checkpoint_file, device=device)
        else:
            pl_sd = safetensors.torch.load(open(checkpoint_file, "rb").read())
            pl_sd = {k: v.to(device) for k, v in pl_sd.items()}
    else:
        pl_sd = torch.load(checkpoint_file, map_location=map_location or shared.weight_load_location)

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = get_state_dict_from_checkpoint(pl_sd)
    return sd


def get_checkpoint_state_dict(checkpoint_info: CheckpointInfo, timer):
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    if checkpoint_info in checkpoints_loaded:
        # use checkpoint cache
        print(f"Loading weights [{sd_model_hash}] from cache")
        # move to end as latest
        checkpoints_loaded.move_to_end(checkpoint_info)
        return checkpoints_loaded[checkpoint_info]

    print(f"Loading weights [{sd_model_hash}] from {checkpoint_info.filename}")
    res = read_state_dict(checkpoint_info.filename)
    timer.record("load weights from disk")

    return res


class SkipWritingToConfig:
    """This context manager prevents load_model_weights from writing checkpoint name to the config when it loads weight."""

    skip = False
    previous = None

    def __enter__(self):
        self.previous = SkipWritingToConfig.skip
        SkipWritingToConfig.skip = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        SkipWritingToConfig.skip = self.previous


class SdModelData:
    def __init__(self):
        self.sd_model = None
        self.was_loaded_at_least_once = False
        self.lock = threading.Lock()

    def get_sd_model(self):
        if self.was_loaded_at_least_once:
            return self.sd_model

        if self.sd_model is None:
            with self.lock:
                if self.sd_model is not None or self.was_loaded_at_least_once:
                    return self.sd_model

                try:
                    load_model()

                except Exception as e:
                    errors.display(e, "loading stable diffusion model", full_traceback=True)
                    print("", file=sys.stderr)
                    print("Stable diffusion model failed to load", file=sys.stderr)
                    self.sd_model = None

        return self.sd_model

    def set_sd_model(self, mdl, already_loaded=False):
        self.sd_model = mdl
        if already_loaded:
            sd_vae.base_vae = getattr(mdl, "base_vae", None)
            sd_vae.loaded_vae_file = getattr(mdl, "loaded_vae_file", None)
            sd_vae.checkpoint_info = mdl.sd_checkpoint_info


model_data = SdModelData()


def get_empty_cond(sd_model):

    p = processing.StableDiffusionProcessingTxt2Img()
    extra_networks.activate(p, {})

    if hasattr(sd_model, "conditioner"):
        d = sd_model.get_learned_conditioning([""])
        return d["crossattn"]
    else:
        return sd_model.cond_stage_model([""])


def send_model_to_cpu(m):
    pass


def model_target_device(m):
    return devices.device


def send_model_to_device(m):
    pass


def send_model_to_trash(m):
    pass


@torch.inference_mode()
def load_model(checkpoint_info=None, already_loaded_state_dict=None):
    from modules import sd_hijack

    checkpoint_info = checkpoint_info or select_checkpoint()

    timer = Timer()

    if model_data.sd_model is not None:
        if model_data.sd_model.filename == checkpoint_info.filename:
            return model_data.sd_model

        model_management.unload_all_models()
        sd_vae.delete_base_vae()
        sd_vae.clear_loaded_vae()

        try:
            del model_data.sd_model.model.diffusion_model
        except AttributeError:
            pass

        for junk in (
            "clip",
            "first_stage_model",
            "cond_stage_model",
            "alphas_cumprod",
            "conditioner",
            "model",
            "forge_objects",
            "forge_objects_original",
            "forge_objects_after_applying_lora",
        ):
            try:
                delattr(model_data.sd_model, junk)
            except AttributeError:
                pass

        del model_data.sd_model
        model_management.soft_empty_cache()
        gc.collect()

    timer.record("unload existing model")

    if already_loaded_state_dict is not None:
        state_dict = already_loaded_state_dict
    else:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

    if shared.opts.sd_checkpoint_cache > 0:
        # cache newly loaded model
        checkpoints_loaded[checkpoint_info] = state_dict.copy()
        # clean up cache if limit is reached
        while len(checkpoints_loaded) > shared.opts.sd_checkpoint_cache:
            checkpoints_loaded.popitem(last=False)

    sd_model = forge_loader.load_model_for_a1111(timer=timer, checkpoint_info=checkpoint_info, state_dict=state_dict)
    sd_model.filename = checkpoint_info.filename

    del state_dict

    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256

    vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename).tuple()
    sd_vae.load_vae(sd_model, vae_file, vae_source)
    timer.record("load VAE")

    model_data.set_sd_model(sd_model)
    model_data.was_loaded_at_least_once = True
    shared.cached_lora_hash.clear()

    # Reload embeddings after model load as they may or may not fit the model
    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
    timer.record("load textual inversion embeddings")

    script_callbacks.model_loaded_callback(sd_model)
    timer.record("scripts callbacks")

    sd_model.cond_stage_model_empty_prompt = get_empty_cond(sd_model)

    timer.record("calculate empty prompt")

    print(f"Model loaded in {timer.summary()}.")

    gc.collect()
    return sd_model


def reuse_model_from_already_loaded(sd_model, checkpoint_info, timer):
    pass


def reload_model_weights(sd_model=None, info=None, forced_reload=False):
    return load_model(info)


def unload_model_weights(sd_model=None, info=None):
    before = model_management.get_free_memory()
    model_management.unload_all_models()
    after = model_management.get_free_memory()
    print(f"\nUnload Weights - Free up {(after - before) // (2 ** 20)} (MB) VRAM\n")

    return sd_model


def list_loaded_weights():
    if len(model_management.current_loaded_models) == 0:
        return

    from rich.console import Console
    from rich.table import Table

    table = Table(title="Currently Loaded Weights")
    table.add_column("Model", justify="left")
    table.add_column("VRAM", justify="right")
    table.add_column("Device", justify="right")

    for mdl in model_management.current_loaded_models:
        table.add_row(
            str(mdl.model.model.__class__.__name__),
            f"{int(mdl.model_memory() / 2 ** 20)} (MB)" if mdl.model_memory() > 0 else "n.a.",
            str(mdl.device),
        )

    print("")
    console = Console()
    console.print(table)


def apply_token_merging(sd_model, token_merging_ratio):
    if token_merging_ratio <= 0:
        return

    from ldm_patched.modules.tomesd import TomePatcher

    sd_model.forge_objects.unet = TomePatcher.patch(model=sd_model.forge_objects.unet, ratio=token_merging_ratio)
    print(f"token_merging_ratio = {token_merging_ratio}")
