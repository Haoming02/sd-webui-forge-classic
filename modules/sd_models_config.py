from modules import shared
import os.path


sd_configs_path = shared.sd_configs_path

config_sd1 = shared.sd_default_config
config_inpainting = os.path.join(sd_configs_path, "v1-inpainting-inference.yaml")
config_sdxl = os.path.join(sd_configs_path, "sd_xl_base.yaml")
config_sdxlv = os.path.join(sd_configs_path, "sd_xl_v.yaml")
config_sdxl_refiner = os.path.join(sd_configs_path, "sd_xl_refiner.yaml")
config_sdxl_inpainting = os.path.join(sd_configs_path, "sd_xl_inpaint.yaml")


def guess_model_config_from_state_dict(sd: dict, *args, **kwargs):
    diffusion_model_input = sd.get("model.diffusion_model.input_blocks.0.0.weight", None)

    if sd.get("conditioner.embedders.1.model.ln_final.weight", None) is not None:
        if diffusion_model_input.shape[1] == 9:
            return config_sdxl_inpainting
        elif "v_pred" in sd:
            return config_sdxlv
        else:
            return config_sdxl

    if sd.get("conditioner.embedders.0.model.ln_final.weight", None) is not None:
        return config_sdxl_refiner

    if diffusion_model_input is not None and diffusion_model_input.shape[1] == 9:
        return config_inpainting

    return config_sd1


def find_checkpoint_config(state_dict, info):
    if info is None:
        return guess_model_config_from_state_dict(state_dict, "")

    config = find_checkpoint_config_near_filename(info)
    if config is not None:
        return config

    return guess_model_config_from_state_dict(state_dict, info.filename)


def find_checkpoint_config_near_filename(info):
    if info is None:
        return None

    config = f"{os.path.splitext(info.filename)[0]}.yaml"
    if os.path.isfile(config):
        return config

    return None
