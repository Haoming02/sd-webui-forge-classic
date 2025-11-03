import argparse
import os

from backend import utils
from modules.paths_internal import models_path, normalized_filepath

parser = argparse.ArgumentParser()

parser.add_argument(
    "--controlnet-dir",
    type=normalized_filepath,
    help="Path to directory with ControlNet models",
    default=os.path.join(models_path, "ControlNet"),
)
parser.add_argument(
    "--controlnet-preprocessor-models-dir",
    type=normalized_filepath,
    help="Path to directory with Annotator models",
    default=os.path.join(models_path, "ControlNetPreprocessor"),
)

cmd_opts, _ = parser.parse_known_args()

controlnet_dir: str = cmd_opts.controlnet_dir
os.makedirs(controlnet_dir, exist_ok=True)

preprocessor_dir: str = cmd_opts.controlnet_preprocessor_models_dir
os.makedirs(preprocessor_dir, exist_ok=True)

diffusers_dir: str = os.path.join(models_path, "diffusers")
os.makedirs(diffusers_dir, exist_ok=True)

supported_preprocessors = {}
supported_control_models = []


def add_supported_preprocessor(preprocessor):
    supported_preprocessors[preprocessor.name] = preprocessor


def add_supported_control_model(control_model):
    supported_control_models.append(control_model)


def try_load_supported_control_model(ckpt_path):
    state_dict = utils.load_torch_file(ckpt_path, safe_load=True)
    for supported_type in supported_control_models:
        state_dict_copy = {k: v for k, v in state_dict.items()}
        model = supported_type.try_build_from_state_dict(state_dict_copy, ckpt_path)
        if model is not None:
            return model

    return None
