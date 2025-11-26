import gc
import math
import os
import re
import sys

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.loader import forge_loader
from modules import cache, devices, errors, extra_networks, hashes, modelloader, patches, paths, processing, script_callbacks, sd_hijack, sd_unet, sd_vae, shared  # noqa
from modules.shared import cmd_opts, opts
from modules.timer import Timer

model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))

checkpoints_list = {}
checkpoint_aliases = {}


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
        abs_ckpt_dirs = (*cmd_opts.ckpt_dirs, model_path)

        self.is_safetensors = os.path.splitext(filename)[1].lower() == ".safetensors"

        for _dir in abs_ckpt_dirs:
            if abspath.startswith(str(_dir)):
                name = abspath.replace(str(_dir), "")
                break

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        def read_metadata():
            metadata = read_metadata_from_safetensors(filename)
            self.modelspec_thumbnail = metadata.pop("modelspec.thumbnail", None)

            return metadata

        self.metadata = {}
        if self.is_safetensors:
            try:
                self.metadata = cache.cached_data_for_file("safetensors-metadata", "checkpoint/" + name, filename, read_metadata)
            except Exception as e:
                errors.display(e, f"reading metadata for {filename}")

        self.name = name
        self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
        self.model_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        self.hash = model_hash(filename)

        self.sha256 = hashes.sha256_from_cache(self.filename, f"checkpoint/{name}")
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
        self.sha256 = hashes.sha256(self.filename, f"checkpoint/{self.name}")
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

    def __str__(self):
        return str(dict(filename=self.filename, hash=self.hash))

    def __repr__(self):
        return str(dict(filename=self.filename, hash=self.hash))


def setup_model():
    os.makedirs(model_path, exist_ok=True)


def checkpoint_tiles(use_short=False):
    return [x.short_title if use_short else x.name for x in checkpoints_list.values()]


def list_models():
    checkpoints_list.clear()
    checkpoint_aliases.clear()
    model_list: set[str] = set()

    for _dir in (*cmd_opts.ckpt_dirs, model_path):
        model_list.update(modelloader.load_models(model_path=_dir, ext_filter=[".ckpt", ".safetensors", ".gguf"], ext_blacklist=[".vae.ckpt", ".vae.safetensors"]))

    for filename in model_list:
        checkpoint_info = CheckpointInfo(filename)
        checkpoint_info.register()


re_strip_checksum = re.compile(r"\s*\[[^]]+]\s*$")


def match_checkpoint_to_name(name):
    name = name.split(" [")[0]

    for ckptname in checkpoints_list.values():
        title = ckptname.title.split(" [")[0]
        if (name in title) or (title in name):
            return ckptname.short_title if shared.opts.sd_checkpoint_dropdown_use_short else ckptname.name.split(" [")[0]

    return name


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


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""

    try:
        with open(filename, "rb") as file:
            import hashlib

            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return "NOFILE"


def select_checkpoint():
    """Raises `FileNotFoundError` if no checkpoints are found."""
    model_checkpoint = shared.opts.sd_model_checkpoint

    checkpoint_info = checkpoint_aliases.get(model_checkpoint, None)
    if checkpoint_info is not None:
        return checkpoint_info

    if len(checkpoints_list) == 0:
        print("You do not have any model!")
        return None

    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        print(f"Checkpoint {model_checkpoint} not found; loading fallback {checkpoint_info.title}", file=sys.stderr)

    return checkpoint_info


def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"

        res = {}

        try:
            json_data = json_start + file.read(metadata_len - 2)
            json_obj = json.loads(json_data)
            for k, v in json_obj.get("__metadata__", {}).items():
                res[k] = v
                if isinstance(v, str) and v[0:1] == "{":
                    try:
                        res[k] = json.loads(v)
                    except Exception:
                        pass
        except Exception:
            errors.report(f"Error reading metadata from file: {filename}", exc_info=True)

        return res


class FakeInitialModel:
    """a dummy class for compatibility when no model is loaded yet"""

    def __init__(self):
        self.cond_stage_model = None
        self.chunk_length = 75

    def get_prompt_lengths_on_ui(self, prompt):
        r = len(prompt.strip("!,. ").replace(" ", ",").replace(".", ",").replace("!", ",").replace(",,", ",").replace(",,", ",").replace(",,", ",").replace(",,", ",").split(","))
        return r, math.ceil(max(r, 1) / self.chunk_length) * self.chunk_length


class SdModelData:
    def __init__(self):
        self.sd_model = FakeInitialModel()
        self.forge_loading_parameters = {}
        self.forge_hash = ""

    def get_sd_model(self):
        return self.sd_model

    def set_sd_model(self, v):
        self.sd_model = v


model_data = SdModelData()


def unload_model_weights(sd_model=None, info=None):
    memory_management.unload_all_models()
    return


def list_loaded_weights():
    if len(memory_management.current_loaded_models) == 0:
        return

    from rich.console import Console
    from rich.table import Table

    table = Table(title="Currently Loaded Weights")
    table.add_column("Model", justify="left")
    table.add_column("VRAM", justify="right")
    table.add_column("Device", justify="right")

    for mdl in memory_management.current_loaded_models:
        mdl.compute_inclusive_exclusive_memory()
        table.add_row(
            str(mdl.model.model.__class__.__name__),
            f"{int(mdl.inclusive_memory / 2 ** 20)} (MB)" if mdl.inclusive_memory > 0 else "n.a.",
            str(mdl.device),
        )

    print("")
    console = Console()
    console.print(table)


def apply_token_merging(sd_model, token_merging_ratio):
    if token_merging_ratio > 0.0:
        from backend.misc.tomesd import TomePatcher

        sd_model.forge_objects.unet = TomePatcher.patch(model=sd_model.forge_objects.unet, ratio=token_merging_ratio)
        print(f"token_merging_ratio = {token_merging_ratio}")

    if opts.scaling_factor > 1.0 and sd_model.model_config.model_type.name == "EPS":
        from backend.misc.eps import EpsilonScaling

        sd_model.forge_objects.unet = EpsilonScaling.patch(model=sd_model.forge_objects.unet, scaling_factor=opts.scaling_factor)
        print(f"eps_scaling_factor = {opts.scaling_factor}")


@torch.inference_mode()
def forge_model_reload():
    current_hash = str(model_data.forge_loading_parameters)

    if model_data.forge_hash == current_hash:
        return model_data.sd_model, False

    print("Loading Model: " + str(model_data.forge_loading_parameters))

    timer = Timer()

    if model_data.sd_model is not None:
        if not isinstance(model_data.sd_model, FakeInitialModel):
            model_data.sd_model.forge_objects.unet.model.cleanup()
            del model_data.sd_model.forge_objects.clip.tokenizer
            del model_data.sd_model.forge_objects.clip.cond_stage_model
            del model_data.sd_model.forge_objects.vae.first_stage_model

        memory_management.unload_all_models()

        for junk in (
            "model_config",
            "forge_objects",
            "forge_objects_original",
            "forge_objects_after_applying_lora",
            "text_processing_engine",
            "text_processing_engine_l",
            "text_processing_engine_g",
            "text_processing_engine_t5",
            "model",
        ):
            try:
                delattr(model_data.sd_model, junk)
            except AttributeError:
                pass

        model_data.sd_model = None
        model_data.forge_hash = ""
        memory_management.soft_empty_cache()
        gc.collect()

    timer.record("unload existing model")

    checkpoint_info = model_data.forge_loading_parameters["checkpoint_info"]

    if checkpoint_info is None:
        raise ValueError("You do not have any model! Please download at least one model in [models/Stable-diffusion].")

    state_dict = checkpoint_info.filename
    additional_state_dicts = model_data.forge_loading_parameters.get("additional_modules", [])

    timer.record("cache state dict")

    dynamic_args["forge_unet_storage_dtype"] = model_data.forge_loading_parameters.get("unet_storage_dtype", None)
    dynamic_args["embedding_dir"] = cmd_opts.embeddings_dir
    sd_model = forge_loader(state_dict, additional_state_dicts=additional_state_dicts)
    timer.record("forge model load")

    sd_model.extra_generation_params = {}
    sd_model.comments = []
    sd_model.sd_checkpoint_info = checkpoint_info
    sd_model.filename = checkpoint_info.filename
    sd_model.sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256

    model_data.set_sd_model(sd_model)

    script_callbacks.model_loaded_callback(sd_model)

    timer.record("scripts callbacks")

    print(f"Model loaded in {timer.summary()}.")

    model_data.forge_hash = current_hash

    return sd_model, True
