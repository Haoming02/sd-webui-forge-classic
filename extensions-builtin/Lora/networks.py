import os.path
import re

from ldm_patched.modules.sd import load_lora_for_models
from ldm_patched.modules.utils import load_torch_file
from modules import errors, scripts, sd_models, shared

import network


def load_lora_state_dict(filename):
    return load_torch_file(filename, safe_load=True)


def load_network(name, network_on_disk):
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)
    return net


def get_networks_on_desk(names: list[str], *, tried: bool = True) -> list["network.NetworkOnDisk"]:
    networks_on_disk = [(available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None)) for name in names]

    if tried or all(x is not None for x in networks_on_disk):
        return networks_on_disk

    list_available_networks()
    return get_networks_on_desk(names, tried=True)


def load_networks(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    current_sd = sd_models.model_data.get_sd_model()
    if current_sd is None:
        return

    loaded_networks.clear()

    networks_on_disk = get_networks_on_desk(names, tried=False)

    for network_on_disk, name in zip(networks_on_disk, names):
        try:
            net = load_network(name, network_on_disk)
            net.mentioned_name = name
            network_on_disk.read_hash()
            loaded_networks.append(net)
        except Exception:
            print(f'\nFailed to load LoRA: "{name}"\n')
            continue

    compiled_lora_targets = []
    for a, b, c in zip(networks_on_disk, unet_multipliers, te_multipliers):
        if a is None:
            continue
        compiled_lora_targets.append((a.filename, b, c))

    if shared.cached_lora_hash == compiled_lora_targets:
        return

    shared.cached_lora_hash = compiled_lora_targets
    current_sd.current_lora_hash = str(compiled_lora_targets)

    if not bool(shared.cached_lora_hash):
        del current_sd.forge_objects_after_applying_lora
        current_sd.forge_objects_after_applying_lora = None
        return

    for filename, strength_model, strength_clip in compiled_lora_targets:
        lora_sd = load_lora_state_dict(filename)
        current_sd.forge_objects.unet, current_sd.forge_objects.clip = load_lora_for_models(
            current_sd.forge_objects.unet,
            current_sd.forge_objects.clip,
            lora_sd,
            strength_model,
            strength_clip,
            filename,
        )

    current_sd.forge_objects_after_applying_lora = current_sd.forge_objects.shallow_copy()


def list_available_networks():
    available_networks.clear()
    available_network_aliases.clear()
    available_network_hash_lookup.clear()
    forbidden_network_aliases.clear()
    forbidden_network_aliases.update({"none": 1, "Addams": 1})

    candidates = list(
        shared.walk_files(
            shared.cmd_opts.lora_dir,
            allowed_extensions=[".pt", ".ckpt", ".safetensors"],
        )
    )

    for filename in candidates:
        if os.path.isdir(filename):
            continue

        name = os.path.splitext(os.path.basename(filename))[0]
        try:
            entry = network.NetworkOnDisk(name, filename)
        except OSError:  # should catch FileNotFoundError and PermissionError, etc.
            errors.report(f"Failed to load network {name} from {filename}", exc_info=True)
            continue

        available_networks[name] = entry

        if entry.alias in available_network_aliases:
            forbidden_network_aliases[entry.alias.lower()] = 1

        available_network_aliases[name] = entry
        available_network_aliases[entry.alias] = entry


re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")


def infotext_pasted(infotext, params):
    if "AddNet Module 1" in [x[1] for x in scripts.scripts_txt2img.infotext_fields]:
        return  # if the other extension is active, it will handle those fields, no need to do anything

    added = []

    for k in params:
        if not k.startswith("AddNet Model "):
            continue

        num = k[13:]

        if params.get("AddNet Module " + num) != "LoRA":
            continue

        name = params.get("AddNet Model " + num)
        if name is None:
            continue

        m = re_network_name.match(name)
        if m:
            name = m.group(1)

        multiplier = params.get("AddNet Weight A " + num, "1.0")

        added.append(f"<lora:{name}:{multiplier}>")

    if added:
        params["Prompt"] += "\n" + "".join(added)


extra_network_lora = None

available_networks = {}
available_network_aliases = {}
loaded_networks = []
loaded_bundle_embeddings = {}
networks_in_memory = {}
available_network_hash_lookup = {}
forbidden_network_aliases = {}

list_available_networks()
