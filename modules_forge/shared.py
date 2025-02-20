import os
import ldm_patched.modules.utils
import argparse

from modules.paths_internal import models_path


parser = argparse.ArgumentParser()

parser.add_argument(
    "--controlnet-dir",
    type=str,
    help="Path to directory with ControlNet models",
    default=None,
)

parser.add_argument(
    "--controlnet-preprocessor-models-dir",
    type=str,
    help="Path to directory with annotator model directories",
    default=None,
)

cmd_opts, _ = parser.parse_known_args()

if cmd_opts.controlnet_dir:
    controlnet_dir = cmd_opts.controlnet_dir
else:
    controlnet_dir = os.path.join(models_path, "ControlNet")
    os.makedirs(controlnet_dir, exist_ok=True)

if cmd_opts.controlnet_preprocessor_models_dir:
    preprocessor_dir = cmd_opts.controlnet_preprocessor_models_dir
else:
    preprocessor_dir = os.path.join(models_path, "ControlNetPreprocessor")
    os.makedirs(preprocessor_dir, exist_ok=True)

diffusers_dir = os.path.join(models_path, "diffusers")
os.makedirs(diffusers_dir, exist_ok=True)

supported_preprocessors = {}
supported_control_models = []


def add_supported_preprocessor(preprocessor):
    global supported_preprocessors
    p = preprocessor
    supported_preprocessors[p.name] = p


def add_supported_control_model(control_model):
    global supported_control_models
    supported_control_models.append(control_model)


def try_load_supported_control_model(ckpt_path):
    global supported_control_models
    state_dict = ldm_patched.modules.utils.load_torch_file(ckpt_path, safe_load=True)
    for supported_type in supported_control_models:
        state_dict_copy = {k: v for k, v in state_dict.items()}
        model = supported_type.try_build_from_state_dict(state_dict_copy, ckpt_path)
        if model is not None:
            return model
    return None
