import glob
import os
from collections import OrderedDict
from functools import lru_cache

from modules import shared
from modules_forge.shared import controlnet_dir, supported_preprocessors

CNET_MODEL_EXTS = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}

controlnet_filename_dict: dict[str, str] = {"None": None}
controlnet_names: list[str] = ["None"]


def traverse_all_files(path: str) -> list[str]:
    files = glob.glob(os.path.join(path, "**", "*"), recursive=True)
    return [file for file in files if os.path.splitext(file)[1] in CNET_MODEL_EXTS]


def get_all_models(path: str, sort_by: str, filter_by: None | str = None) -> dict:
    result = OrderedDict()
    models = traverse_all_files(path)

    if filter_by:
        filter_by = filter_by.strip().lower()
        models = [m for m in models if filter_by in os.path.basename(m).lower()]

    assert sort_by == "name"
    models = sorted(models, key=lambda m: os.path.basename(m))

    for filename in models:
        name = os.path.splitext(os.path.basename(filename))[0]
        result[name] = filename

    result.pop("None", None)
    return result


def update_controlnet_filenames():
    global controlnet_filename_dict, controlnet_names
    controlnet_filename_dict = {"None": None}

    ext_dirs = (
        shared.opts.data.get("control_net_models_path", None),
        getattr(shared.cmd_opts, "controlnet_dir", None),
    )
    extra_paths = (extra_path for extra_path in ext_dirs if extra_path is not None and os.path.exists(extra_path))

    for path in [controlnet_dir, *extra_paths]:
        found = get_all_models(path, "name")
        controlnet_filename_dict.update(found)

    controlnet_names = sorted(controlnet_filename_dict.keys(), key=lambda mdl: mdl)


def get_all_controlnet_names() -> list[str]:
    return controlnet_names


def get_controlnet_filename(controlnet_name: str) -> str:
    return controlnet_filename_dict[controlnet_name]


def get_filtered_controlnet_names(tag: str) -> list[str]:
    filename_filters = ["union", "promax"]

    filtered_preprocessors = get_filtered_preprocessors(tag)
    for p in filtered_preprocessors.values():
        filename_filters.extend(p.model_filename_filters)

    return [cnet for cnet in controlnet_names if cnet == "None" or any(f.lower() in cnet.lower() for f in filename_filters)]


def get_all_preprocessor_tags() -> list[str]:
    tags = []
    for p in supported_preprocessors.values():
        tags.extend(p.tags)
    tags = sorted(list(set(tags)))
    return ["All"] + tags


def get_preprocessor(name: str):
    return supported_preprocessors[name]


def get_default_preprocessor(tag: str) -> str:
    ps = get_filtered_preprocessor_names(tag)
    assert len(ps) > 0
    return ps[0] if len(ps) == 1 else ps[1]


@lru_cache(maxsize=1, typed=False)
def get_sorted_preprocessors() -> dict:
    results = OrderedDict({"None": supported_preprocessors["None"]})
    preprocessors = [p for (k, p) in supported_preprocessors.items() if k != "None"]
    preprocessors = sorted(preprocessors, key=lambda mdl: mdl.name)
    for p in preprocessors:
        results[p.name] = p
    return results


def get_all_preprocessor_names() -> list[str]:
    return list(get_sorted_preprocessors().keys())


def get_filtered_preprocessor_names(tag: str) -> list[str]:
    return list(get_filtered_preprocessors(tag).keys())


def get_filtered_preprocessors(tag: str) -> dict:
    if tag == "All":
        return supported_preprocessors
    return {k: v for (k, v) in get_sorted_preprocessors().items() if tag in v.tags or k == "None"}


def select_control_type(control_type: str) -> tuple[list[str], list[str], str, str]:
    global controlnet_names

    pattern = control_type.lower()
    all_models = list(controlnet_names)

    if pattern == "all":
        preprocessors = get_sorted_preprocessors().values()
        return [[p.name for p in preprocessors], all_models, "none", "None"]

    filtered_model_list = get_filtered_controlnet_names(control_type)

    if pattern == "none":
        filtered_model_list.append("None")

    assert len(filtered_model_list) > 0
    if len(filtered_model_list) == 1:
        default_model = "None"

    else:
        default_model = filtered_model_list[1]
        for x in filtered_model_list:
            if "11" in x.split("[")[0]:
                default_model = x
                break

    return (get_filtered_preprocessor_names(control_type), filtered_model_list, get_default_preprocessor(control_type), default_model)
