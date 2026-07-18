import json
import os
import re
from enum import Enum

import gradio as gr
import torch
import tqdm
from huggingface_guess.detection import unet_prefix_from_state_dict
from safetensors.torch import save_file

from backend.state_dict import state_dict_prefix_replace
from backend.utils import load_torch_file
from modules import images, sd_models, shared
from modules.ui_common import plaintext_to_html


def run_pnginfo(image):
    if image is None:
        return "", "", ""

    geninfo, items = images.read_info_from_image(image)
    items = {**{"parameters": geninfo}, **items}

    info = ""
    for key, text in items.items():
        info += f"""<div>
            <p><b>{plaintext_to_html(str(key))}</b></p>
            <p>{plaintext_to_html(str(text))}</p>
        </div>""".strip() + "\n"

    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    return "", geninfo, info


class InterpolationMethod(Enum):
    no_interpolation = "No Interpolation"
    weighted_sum = "Weighted Sum"
    add_difference = "Add Difference"

    @staticmethod
    def titles() -> list[str]:
        return [m.value for m in InterpolationMethod]

    @staticmethod
    def desc(value: str) -> str:
        match value:
            case InterpolationMethod.no_interpolation.value:
                return "Require 1 Model ; Mainly for format conversion"
            case InterpolationMethod.weighted_sum.value:
                return "Require 2 Model ; Result is calculated as A * (1 - M) + B * M"
            case InterpolationMethod.add_difference.value:
                return "Require 3 Model ; Result is calculated as A + (B - C) * M"


def read_metadata(primary_model_name: str, secondary_model_name: str, tertiary_model_name: str) -> str:
    metadata = {}

    for checkpoint_name in (primary_model_name, secondary_model_name, tertiary_model_name):
        if (checkpoint_info := sd_models.checkpoints_list.get(checkpoint_name, None)) is not None:
            metadata.update({checkpoint_name: checkpoint_info.metadata})

    return gr.update(value=json.dumps(metadata, indent=4, ensure_ascii=False), visible=True)


def run_modelmerger(id_task, primary_model_name: str, secondary_model_name: str, tertiary_model_name: str, interp_method: str, multiplier: float, custom_name: str, discard_weights: str, save_metadata: bool, config_source: list[str], add_merge_recipe: bool):
    shared.state.begin(job="model-merge")

    def fail(message: str):
        shared.state.textinfo = message
        shared.state.end()
        return [gr.skip(), gr.skip(), gr.skip(), gr.skip(), message]

    if not primary_model_name:
        return fail("Failed: Missing Primary Model")

    if interp_method != InterpolationMethod.no_interpolation.value and not secondary_model_name:
        return fail("Failed: Missing Secondary Model")

    if interp_method == InterpolationMethod.add_difference.value and not tertiary_model_name:
        return fail("Failed: Missing Tertiary Model")

    def weighted_sum(theta0: torch.Tensor, theta1: torch.Tensor, alpha: float) -> torch.Tensor:
        return torch.lerp(theta0, theta1, alpha)

    def get_difference(theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        return torch.subtract(theta1, theta2)

    def add_difference(theta0: torch.Tensor, theta1_2_diff: torch.Tensor, alpha: float) -> torch.Tensor:
        return torch.add(theta0, theta1_2_diff, alpha=alpha)

    def filename_weighted_sum() -> str:
        a = primary_model_info.model_name
        b = secondary_model_info.model_name
        Ma = round(1 - multiplier, 2)
        Mb = round(multiplier, 2)

        return f"{Ma}({a}) + {Mb}({b})"

    def filename_add_difference() -> str:
        a = primary_model_info.model_name
        b = secondary_model_info.model_name
        c = tertiary_model_info.model_name
        m = round(multiplier, 2)

        return f"{a} + {m}({b} - {c})"

    def filename_nothing() -> str:
        return primary_model_info.model_name

    THETA_FUNCS = {
        InterpolationMethod.no_interpolation.value: (filename_nothing, None, None),
        InterpolationMethod.weighted_sum.value: (filename_weighted_sum, None, weighted_sum),
        InterpolationMethod.add_difference.value: (filename_add_difference, get_difference, add_difference),
    }

    filename_generator, theta_func1, theta_func2 = THETA_FUNCS[interp_method]
    shared.state.job_count = (1 if theta_func1 else 0) + (1 if theta_func2 else 0)

    primary_model_info = sd_models.checkpoint_aliases[primary_model_name]
    secondary_model_info = sd_models.checkpoint_aliases[secondary_model_name] if theta_func2 else None
    tertiary_model_info = sd_models.checkpoint_aliases[tertiary_model_name] if theta_func1 else None

    def _load_state_dict(filename: str) -> dict[str, torch.Tensor]:
        sd = load_torch_file(filename)
        prefix = unet_prefix_from_state_dict(sd)
        return state_dict_prefix_replace(sd, {prefix: "model.diffusion_model."})

    if theta_func2:
        shared.state.textinfo = "Loading B"
        print(f"Loading {secondary_model_info.filename}...")
        theta_1 = _load_state_dict(secondary_model_info.filename)
    else:
        theta_1 = None

    if theta_func1:
        _keys = list(theta_1.keys())

        shared.state.textinfo = "Loading C"
        print(f"Loading {tertiary_model_info.filename}...")
        theta_2 = _load_state_dict(tertiary_model_info.filename)

        shared.state.textinfo = "Merging B and C"
        shared.state.sampling_steps = total = len(_keys)
        missing = 0

        for key in tqdm.tqdm(_keys):
            shared.state.sampling_step += 1

            if key in theta_2:
                a = theta_1.pop(key)
                b = theta_2.pop(key)

                if a.shape != b.shape:
                    raise ValueError(f"Shape Mismatch ({tuple(a.shape)} != {tuple(b.shape)})")

                theta_1[key] = theta_func1(a.float(), b.float()).to(a)
            else:
                w = theta_1.pop(key)
                theta_1[key] = torch.zeros_like(w)
                missing += 1

        del theta_2
        shared.state.nextjob()

        if missing > total * 0.25:
            raise SystemError("Keys Mismatch between B & C...")

    shared.state.textinfo = "Loading A"
    print(f"Loading {primary_model_info.filename}...")
    theta_0 = _load_state_dict(primary_model_info.filename)

    if theta_func2:
        _keys = list(theta_0.keys())

        shared.state.textinfo = "Merging A and B"
        shared.state.sampling_steps = total = len(_keys)
        missing = 0

        for key in tqdm.tqdm(_keys):
            shared.state.sampling_step += 1

            if key not in theta_1:
                missing += 1
                continue

            a = theta_0.pop(key)
            b = theta_1.pop(key)

            if a.shape != b.shape:
                raise ValueError(f"Shape Mismatch ({tuple(a.shape)} != {tuple(b.shape)})")

            theta_0[key] = theta_func2(a.float(), b.float(), multiplier).to(a)

        del theta_1
        shared.state.nextjob()

        if missing > total * 0.25:
            raise SystemError("Keys Mismatch between A & B...")

    if discard_weights:
        regex = re.compile(discard_weights)
        for key in list(theta_0.keys()):
            if re.search(regex, key):
                theta_0.pop(key)

    filename: str = custom_name or filename_generator()
    if not filename.endswith(".safetensors"):
        filename += ".safetensors"

    output_modelname = os.path.join(sd_models.model_path, filename)

    shared.state.textinfo = "Saving"
    print(f"Saving to {output_modelname}...")

    metadata = {}

    if save_metadata:
        if "A" in config_source and primary_model_info is not None:
            metadata.update(primary_model_info.metadata)
        if "B" in config_source and secondary_model_info is not None:
            metadata.update(secondary_model_info.metadata)
        if "C" in config_source and tertiary_model_info is not None:
            metadata.update(tertiary_model_info.metadata)

        if add_merge_recipe:
            merge_recipe = {
                "type": "Neo",
                "interp_method": interp_method,
                "multiplier": multiplier,
                "discard_weights": discard_weights,
                "config_source": config_source,
            }

            sd_merge_models = {}

            def add_model_metadata(key: str, checkpoint_info: sd_models.CheckpointInfo):
                checkpoint_info.calculate_shorthash()
                merge_recipe[key] = checkpoint_info.sha256

                sd_merge_models[checkpoint_info.sha256] = {
                    "name": checkpoint_info.name,
                    "legacy_hash": checkpoint_info.hash,
                }

                if (r := checkpoint_info.metadata.get("sd_merge_recipe", None)) is not None:
                    sd_merge_models["sd_merge_recipe"] = r
                if (m := checkpoint_info.metadata.get("sd_merge_models", None)) is not None:
                    sd_merge_models["sd_merge_models"] = m

            if primary_model_info:
                add_model_metadata("primary_model_hash", primary_model_info)
            if secondary_model_info:
                add_model_metadata("secondary_model_hash", secondary_model_info)
            if tertiary_model_info:
                add_model_metadata("tertiary_model_hash", tertiary_model_info)

            metadata["sd_merge_recipe"] = json.dumps(merge_recipe)
            metadata["sd_merge_models"] = json.dumps(sd_merge_models)

    def sanitize_metadata(meta_dict: dict | None) -> dict | None:
        if not meta_dict:
            return None

        sanitized = {}
        for key, value in meta_dict.items():
            if value is None:
                continue
            elif isinstance(value, str):
                sanitized[key] = value
            elif isinstance(value, (dict, list)):
                sanitized[key] = json.dumps(value)
            else:
                sanitized[key] = str(value)

        return sanitized

    save_file(theta_0, output_modelname, metadata=sanitize_metadata(metadata))
    print(f"Checkpoint saved to {output_modelname}")

    shared.state.textinfo = "Checkpoint Saved"
    shared.state.end()

    sd_models.list_models()
    return [gr.update(choices=sorted(sd_models.checkpoint_tiles()))] * 4 + [f"Checkpoint saved to {output_modelname}"]
