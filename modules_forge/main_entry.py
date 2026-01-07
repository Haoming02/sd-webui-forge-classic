import logging
import os.path

import gradio as gr
import torch
from gradio.context import Context
from rich import print_json

from backend import memory_management, operations
from backend.args import dynamic_args
from backend.logging import setup_logger
from modules import infotext_utils, paths, processing, sd_models, shared, shared_items, ui_common

logger = logging.getLogger("ui_models")
setup_logger(logger)

total_vram = int(memory_management.total_vram)

ui_forge_preset: gr.Radio
ui_checkpoint: gr.Dropdown
ui_vae: gr.Dropdown
ui_forge_unet_dtype: gr.Radio

forge_unet_storage_dtype_options: dict[str, tuple[torch.dtype, bool]] = {
    "Automatic": (None, False),
    "Automatic (fp16 LoRA)": (None, True),
    "float8-e4m3fn": (torch.float8_e4m3fn, False),
    "float8-e4m3fn (fp16 LoRA)": (torch.float8_e4m3fn, True),
    "float8-e5m2": (torch.float8_e5m2, False),
    "float8-e5m2 (fp16 LoRA)": (torch.float8_e5m2, True),
}

if operations.bnb_available:
    forge_unet_storage_dtype_options.update(
        {
            "bnb-nf4": ("nf4", False),
            "bnb-nf4 (fp16 LoRA)": ("nf4", True),
            "bnb-fp4": ("fp4", False),
            "bnb-fp4 (fp16 LoRA)": ("fp4", True),
        }
    )


module_list: dict[str, os.PathLike] = {}


def bind_to_opts(comp: gr.components.Component, k: str, save: bool = False, callback=None):
    def on_change(v):
        shared.opts.set(k, v)
        if save:
            shared.opts.save(shared.config_filename)
        if callback is not None:
            callback()

    comp.change(on_change, inputs=[comp], queue=False, show_progress=False)


def make_checkpoint_manager_ui():
    global ui_forge_preset, ui_checkpoint, ui_vae, ui_forge_unet_dtype

    if shared.opts.sd_model_checkpoint in [None, "None", "none", ""]:
        if len(sd_models.checkpoints_list) == 0:
            sd_models.list_models()
        if len(sd_models.checkpoints_list) > 0:
            shared.opts.set("sd_model_checkpoint", next(iter(sd_models.checkpoints_list.values())).name)

    ui_forge_preset = gr.Radio(label="UI Preset", value=lambda: shared.opts.forge_preset, choices=("sd", "xl", "flux", "qwen", "lumina", "wan"), elem_id="forge_ui_preset")

    ui_checkpoint = gr.Dropdown(label="Checkpoint", value=None, choices=None, elem_id="setting_sd_model_checkpoint", elem_classes=["model_selection"])

    ui_vae = gr.Dropdown(label="VAE / Text Encoder", value=None, choices=None, multiselect=True)

    def refresh_model_list():
        ckpt_list, vae_list = refresh_models()
        return gr.update(choices=ckpt_list), gr.update(choices=vae_list)

    refresh_button = ui_common.ToolButton(value=ui_common.refresh_symbol, elem_id="forge_refresh_checkpoint", tooltip="Refresh")
    refresh_button.click(fn=refresh_model_list, outputs=[ui_checkpoint, ui_vae], queue=False)

    def model_on_load():
        ckpt_list, vae_list = refresh_models()
        return [gr.update(value=shared.opts.sd_model_checkpoint, choices=ckpt_list), gr.update(value=[os.path.basename(x) for x in shared.opts.forge_additional_modules], choices=vae_list)]

    Context.root_block.load(fn=model_on_load, outputs=[ui_checkpoint, ui_vae], show_progress=False, queue=False)

    ui_forge_unet_dtype = gr.Dropdown(label="Diffusion in Low Bits", value=lambda: shared.opts.forge_unet_storage_dtype, choices=list(forge_unet_storage_dtype_options.keys()))
    bind_to_opts(ui_forge_unet_dtype, "forge_unet_storage_dtype", save=True, callback=refresh_model_loading_parameters)

    ui_checkpoint.change(checkpoint_change, inputs=[ui_checkpoint, ui_forge_preset], show_progress=False)
    ui_vae.change(modules_change, inputs=[ui_vae, ui_forge_preset], queue=False, show_progress=False)


def find_files_with_extensions(base_path: os.PathLike, extensions: list[str]) -> dict[str, os.PathLike]:
    found_files = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, file)
                found_files[file] = full_path
    return found_files


def refresh_models() -> tuple[list[os.PathLike], list[os.PathLike]]:
    shared_items.refresh_checkpoints()
    ckpt_list = shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)

    file_extensions = ("ckpt", "pt", "pth", "bin", "safetensors", "sft", "gguf")

    module_list.clear()

    module_paths: set[os.PathLike] = {
        os.path.abspath(os.path.join(paths.models_path, "VAE")),
        os.path.abspath(os.path.join(paths.models_path, "text_encoder")),
        *shared.cmd_opts.vae_dirs,
        *shared.cmd_opts.text_encoder_dirs,
    }

    for vae_path in module_paths:
        vae_files = find_files_with_extensions(vae_path, file_extensions)
        module_list.update(vae_files)

    return sorted(ckpt_list), sorted(module_list.keys())


def refresh_model_loading_parameters():
    from modules.sd_models import model_data, select_checkpoint

    checkpoint_info = select_checkpoint()

    unet_storage_dtype, lora_fp16 = forge_unet_storage_dtype_options.get(shared.opts.forge_unet_storage_dtype, (None, False))

    ckpt: str = checkpoint_info.filename
    if ckpt.endswith(("gguf", "GGUF")) and not lora_fp16:
        logger.warning("GGUF requires fp16 LoRA ; overriding option")
        lora_fp16 = True

    dynamic_args["online_lora"] = lora_fp16

    model_data.forge_loading_parameters = dict(checkpoint_info=checkpoint_info, additional_modules=shared.opts.forge_additional_modules, unet_storage_dtype=unet_storage_dtype)

    modules: list[str] = [os.path.basename(x) for x in shared.opts.forge_additional_modules]

    logger.info("Model Selected:")
    print_json(data=dict(checkpoint=os.path.basename(ckpt), modules=modules))
    logger.info(f"Patch LoRAs on-the-fly: {lora_fp16}")
    processing.need_global_unload = True


def checkpoint_change(ckpt_name: str, preset: str, save=True, refresh=True) -> bool:
    """`ckpt_name` accepts valid aliases; returns `True` if checkpoint changed"""
    new_ckpt_info = sd_models.get_closet_checkpoint_match(ckpt_name)
    current_ckpt_info = sd_models.get_closet_checkpoint_match(shared.opts.data.get("sd_model_checkpoint", ""))
    if new_ckpt_info == current_ckpt_info:
        return False

    shared.opts.set("sd_model_checkpoint", ckpt_name)
    if preset is not None:
        shared.opts.set(f"forge_checkpoint_{preset}", ckpt_name)

    if save:
        shared.opts.save(shared.config_filename)
    if refresh:
        refresh_model_loading_parameters()
    return True


def modules_change(module_values: list, preset: str, save=True, refresh=True) -> bool:
    """`module_values` accepts file paths or just the module names; returns `True` if modules changed"""
    modules = []
    for v in module_values:
        module_name = os.path.basename(v)  # If the input is a filepath, extract the file name
        if module_name in module_list:
            modules.append(module_list[module_name])
    modules.sort()

    # skip further processing if value unchanged
    if modules == shared.opts.data.get("forge_additional_modules", []):
        return False

    shared.opts.set("forge_additional_modules", modules)
    if preset is not None:
        shared.opts.set(f"forge_additional_modules_{preset}", modules)

    if save:
        shared.opts.save(shared.config_filename)
    if refresh:
        refresh_model_loading_parameters()
    return True


def get_a1111_ui_component(tab: str, label: str) -> gr.components.Component:
    fields = infotext_utils.paste_fields[tab]["fields"]
    for f in fields:
        if f.label == label or f.api == label:
            return f.component


def forge_main_entry():
    ui_txt2img_width = get_a1111_ui_component("txt2img", "Size-1")
    ui_txt2img_height = get_a1111_ui_component("txt2img", "Size-2")
    ui_txt2img_cfg = get_a1111_ui_component("txt2img", "CFG scale")
    ui_txt2img_distilled_cfg = get_a1111_ui_component("txt2img", "Distilled CFG Scale")
    ui_txt2img_sampler = get_a1111_ui_component("txt2img", "sampler_name")
    ui_txt2img_scheduler = get_a1111_ui_component("txt2img", "scheduler")

    ui_img2img_width = get_a1111_ui_component("img2img", "Size-1")
    ui_img2img_height = get_a1111_ui_component("img2img", "Size-2")
    ui_img2img_cfg = get_a1111_ui_component("img2img", "CFG scale")
    ui_img2img_distilled_cfg = get_a1111_ui_component("img2img", "Distilled CFG Scale")
    ui_img2img_sampler = get_a1111_ui_component("img2img", "sampler_name")
    ui_img2img_scheduler = get_a1111_ui_component("img2img", "scheduler")

    ui_txt2img_hr_cfg = get_a1111_ui_component("txt2img", "Hires CFG Scale")
    ui_txt2img_hr_distilled_cfg = get_a1111_ui_component("txt2img", "Hires Distilled CFG Scale")

    ui_txt2img_batch_size = get_a1111_ui_component("txt2img", "Batch size")
    ui_img2img_batch_size = get_a1111_ui_component("img2img", "Batch size")

    output_targets = [
        ui_checkpoint,
        ui_vae,
        ui_forge_unet_dtype,
        ui_txt2img_width,
        ui_img2img_width,
        ui_txt2img_height,
        ui_img2img_height,
        ui_txt2img_cfg,
        ui_img2img_cfg,
        ui_txt2img_distilled_cfg,
        ui_img2img_distilled_cfg,
        ui_txt2img_sampler,
        ui_img2img_sampler,
        ui_txt2img_scheduler,
        ui_img2img_scheduler,
        ui_txt2img_hr_cfg,
        ui_txt2img_hr_distilled_cfg,
        ui_txt2img_batch_size,
        ui_img2img_batch_size,
    ]

    ui_forge_preset.change(on_preset_change, inputs=[ui_forge_preset], outputs=output_targets, queue=False, show_progress=False).then(js="clickLoraRefresh", fn=None, queue=False, show_progress=False)
    Context.root_block.load(on_preset_change, inputs=[ui_forge_preset], outputs=output_targets, queue=False, show_progress=False)

    refresh_model_loading_parameters()


def on_preset_change(preset: str):
    assert preset is not None
    shared.opts.set("forge_preset", preset)
    shared.opts.save(shared.config_filename)

    additional_modules = [os.path.basename(x) for x in getattr(shared.opts, f"forge_additional_modules_{preset}", [])]

    extra_slider = preset in ("flux", "lumina", "wan")
    distill_label = "Distilled CFG Scale" if preset == "flux" else "Shift"
    batch_args = {"minimum": 1, "maximum": 97, "step": 16, "label": "Frames", "value": 1} if preset == "wan" else {"minimum": 1, "maximum": 8, "step": 1, "label": "Batch size", "value": 1}

    return [
        gr.update(value=getattr(shared.opts, f"forge_checkpoint_{preset}", shared.opts.sd_model_checkpoint)),  # ui_checkpoint
        gr.update(value=additional_modules),  # ui_vae
        gr.update(value=getattr(shared.opts, "forge_unet_storage_dtype", "Automatic")),  # ui_forge_unet_dtype
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_width", 768)),  # ui_txt2img_width
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_width", 768)),  # ui_img2img_width
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_height", 768)),  # ui_txt2img_height
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_height", 768)),  # ui_img2img_height
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_cfg", 1.0)),  # ui_txt2img_cfg
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_cfg", 1.0)),  # ui_img2img_cfg
        gr.update(visible=extra_slider, label=distill_label, value=getattr(shared.opts, f"{preset}_t2i_d_cfg", 3.0)),  # ui_txt2img_distilled_cfg
        gr.update(visible=extra_slider, label=distill_label, value=getattr(shared.opts, f"{preset}_i2i_d_cfg", 3.0)),  # ui_img2img_distilled_cfg
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_sampler", "Euler")),  # ui_txt2img_sampler
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_sampler", "Euler")),  # ui_img2img_sampler
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_scheduler", "Simple")),  # ui_txt2img_scheduler
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_scheduler", "Simple")),  # ui_img2img_scheduler
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_hr_cfg", 1.0)),  # ui_txt2img_hr_cfg
        gr.update(visible=extra_slider, label=distill_label, value=getattr(shared.opts, f"{preset}_t2i_hr_d_cfg", 3.0)),  # ui_txt2img_hr_distilled_cfg
        gr.update(**batch_args),  # ui_txt2img_batch_size
        gr.update(**batch_args),  # ui_img2img_batch_size
    ]
