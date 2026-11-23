import logging
import os.path
from typing import Callable, OrderedDict

import gradio as gr
import torch
from gradio.context import Context
from gradio.events import Dependency
from rich import print_json

from backend.args import dynamic_args
from backend.logging import setup_logger
from modules import (
    infotext_utils,
    paths,
    processing,
    sd_models,
    shared,
    shared_items,
    ui_common,
)
from modules_forge.presets import Preset, is_video, load_presets, use_distill, use_shift

logger = logging.getLogger("ui_models")
setup_logger(logger)

DEFAULTS: dict[str, Preset] = {}
ALL_PRESETS: dict[str, Preset] = {}

ui_forge_preset: gr.Dropdown
ui_checkpoint: gr.Dropdown
ui_vae: gr.Dropdown
ui_forge_unet_dtype: gr.Dropdown
_event_refresh: Dependency
_event_load: Dependency

forge_unet_storage_dtype_options: dict[str, tuple[torch.dtype, bool]] = {
    "Automatic": (None, False),
    "Automatic (fp16 LoRA)": (None, True),
    "float8-e4m3fn": (torch.float8_e4m3fn, False),
    "float8-e4m3fn (fp16 LoRA)": (torch.float8_e4m3fn, True),
    "float8-e5m2": (torch.float8_e5m2, False),
    "float8-e5m2 (fp16 LoRA)": (torch.float8_e5m2, True),
}


module_list: OrderedDict[str, os.PathLike] = OrderedDict()
module_map: OrderedDict[str, str] = OrderedDict()


def make_checkpoint_manager_ui():
    global ui_forge_preset, ui_checkpoint, ui_vae, ui_forge_unet_dtype, _event_refresh, _event_load
    global DEFAULTS, ALL_PRESETS

    if shared.opts.sd_model_checkpoint in [None, "None", "none", ""]:
        if len(sd_models.checkpoints_list) == 0:
            sd_models.list_models()
        if len(sd_models.checkpoints_list) > 0:
            shared.opts.set("sd_model_checkpoint", next(iter(sd_models.checkpoints_list.values())).name)

    DEFAULTS, ALL_PRESETS = load_presets(set(shared.cmd_opts.disable_preset))
    if not ALL_PRESETS:
        raise ValueError("No Preset is Available...")

    choices: list[str] = sorted(ALL_PRESETS) if shared.cmd_opts.sort_preset else list(ALL_PRESETS)
    selected: str = next((p for p in choices if p == shared.opts.forge_preset), choices[0])

    ui_forge_preset = gr.Dropdown(label="UI Preset", value=selected, choices=choices, elem_id="forge_ui_preset")

    ui_checkpoint = gr.Dropdown(label="Checkpoint", value=None, choices=None, elem_id="setting_sd_model_checkpoint", elem_classes=["model_selection"])

    ui_vae = gr.Dropdown(label="VAE / Text Encoder", value=None, choices=None, multiselect=True, elem_id="setting_sd_modules", elem_classes=["model_selection"])

    def refresh_model_list():
        ckpt_list, vae_list = refresh_models()
        return [gr.update(choices=ckpt_list), gr.update(choices=vae_list)]

    refresh_button = ui_common.ToolButton(value=ui_common.refresh_symbol, elem_id="forge_refresh_checkpoint", tooltip="Refresh")
    _event_refresh = refresh_button.click(fn=refresh_model_list, outputs=[ui_checkpoint, ui_vae], queue=False)
    _event_load = Context.root_block.load(fn=refresh_model_list, outputs=[ui_checkpoint, ui_vae], queue=False)

    ui_forge_unet_dtype = gr.Dropdown(label="Diffusion in Low Bits", value=None, choices=list(forge_unet_storage_dtype_options.keys()), elem_id="forge_ui_dtype")

    ui_checkpoint.input(checkpoint_change, inputs=[ui_checkpoint, ui_forge_preset], queue=False, show_progress=False)
    ui_vae.input(modules_change, inputs=[ui_vae, ui_forge_preset], queue=False, show_progress=False)
    ui_forge_unet_dtype.input(dtype_change, inputs=[ui_forge_unet_dtype, ui_forge_preset], queue=False, show_progress=False)


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
    shared_items.refresh_vae_list()
    ckpt_list = shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)

    file_extensions = ("ckpt", "pt", "pth", "bin", "safetensors", "sft", "gguf")

    _module_list: dict[str, os.PathLike] = {}
    _module_map: dict[str, str] = {}

    module_paths: set[os.PathLike] = {
        os.path.abspath(os.path.join(paths.models_path, "VAE")),
        os.path.abspath(os.path.join(paths.models_path, "text_encoder")),
        *shared.cmd_opts.vae_dirs,
        *shared.cmd_opts.text_encoder_dirs,
    }

    for vae_path in module_paths:
        vae_files = find_files_with_extensions(vae_path, file_extensions)
        _module_list.update(vae_files)

        for file, path in vae_files.items():
            rel = os.path.normpath(os.path.relpath(path, vae_path))
            _module_map[file] = rel

    module_list.clear()
    module_map.clear()

    module_list.update({title: _module_list[title] for title in sorted(_module_list)})
    module_map.update({title: _module_map[title] for title in sorted(_module_map)})

    if shared.opts.sd_modules_dropdown_use_short:
        return ckpt_list, list(module_list.keys())
    else:
        return ckpt_list, list(module_map.values())


def refresh_model_loading_parameters(*, refresh: bool = True, disable_log: bool = False):
    if not refresh:
        return

    from modules.sd_models import model_data, select_checkpoint

    checkpoint_info = select_checkpoint()
    if checkpoint_info is None:
        logger.critical('You do not have any model... Please download models to "models/Stable-diffusion"')
        return

    unet_storage_dtype, lora_fp16 = forge_unet_storage_dtype_options.get(shared.opts.forge_unet_storage_dtype, (None, False))

    model_data.forge_loading_parameters = dict(checkpoint_info=checkpoint_info, additional_modules=shared.opts.forge_additional_modules, unet_storage_dtype=unet_storage_dtype)

    ckpt: str = checkpoint_info.filename
    modules: list[str] = [os.path.basename(x) for x in shared.opts.forge_additional_modules]
    dtype = str(unet_storage_dtype or [torch.float16, torch.bfloat16])

    if not disable_log:
        logger.info("Model Selected:")
        print_json(data=dict(checkpoint=os.path.basename(ckpt), modules=modules, dtype=dtype))

    if ckpt.endswith(("gguf", "GGUF")) and not lora_fp16:
        lora_fp16 = True
        if not disable_log:
            logger.warning("GGUF requires fp16 LoRA ; overriding option")

    dynamic_args.online_lora = lora_fp16

    if not disable_log:
        logger.info(f"Patch LoRAs on-the-fly: {lora_fp16}")
        if not ckpt.endswith(("gguf", "GGUF")) and lora_fp16:
            logger.warning("on-the-fly WILL be slower ; enable only if you know what you are doing")

    processing.need_global_unload = True


def _save_models_to_preset(preset: str, key: str, value: str | list[str]):
    preset_models: dict = shared.opts.data.get("forge_preset_models", {})
    models: dict = preset_models.setdefault(preset, {})
    models.update({key: value})
    shared.opts.set("forge_preset_models", preset_models)


def checkpoint_change(ckpt_name: str, preset: str, save=True, refresh=True) -> bool:
    """`ckpt_name` accepts valid aliases; returns `True` if checkpoint changed"""
    new_ckpt_info = sd_models.get_closet_checkpoint_match(ckpt_name)
    current_ckpt_info = sd_models.get_closet_checkpoint_match(getattr(shared.opts, "sd_model_checkpoint", ""))
    if new_ckpt_info == current_ckpt_info:
        return False

    shared.opts.set("sd_model_checkpoint", ckpt_name)
    if preset is not None:
        _save_models_to_preset(preset, "checkpoint", ckpt_name)

    if save:
        shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters(refresh=refresh)
    return True


def modules_change(module_values: list, preset: str, save=True, refresh=True) -> bool:
    """`module_values` accepts file paths or just the module names; returns `True` if modules changed"""
    modules = []
    for v in module_values:
        module_name = os.path.basename(v)  # If the input is a filepath, extract the filename
        if module_name in module_list:
            modules.append(module_list[module_name])
    modules.sort()

    # skip further processing if value unchanged
    if modules == getattr(shared.opts, "forge_additional_modules", []):
        return False

    shared.opts.set("forge_additional_modules", modules)
    if preset is not None:
        _save_models_to_preset(preset, "modules", modules)

    if save:
        shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters(refresh=refresh)
    return True


def dtype_change(dtype: str, preset: str, save=True, refresh=True) -> bool:
    shared.opts.set("forge_unet_storage_dtype", dtype)
    if preset is not None:
        _save_models_to_preset(preset, "dtype", dtype)

    if save:
        shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters(refresh=refresh)
    return True


def get_a1111_ui_component(tab: str, label: str) -> gr.components.Component:
    fields = infotext_utils.paste_fields[tab]["fields"]
    for f in fields:
        if f.label == label or f.api == label:
            return f.component


def forge_main_entry():
    ui_txt2img_steps = get_a1111_ui_component("txt2img", "Steps")
    ui_txt2img_hr_steps = get_a1111_ui_component("txt2img", "Hires steps")
    ui_img2img_steps = get_a1111_ui_component("img2img", "Steps")

    ui_txt2img_sampler = get_a1111_ui_component("txt2img", "sampler_name")
    ui_img2img_sampler = get_a1111_ui_component("img2img", "sampler_name")
    ui_txt2img_scheduler = get_a1111_ui_component("txt2img", "scheduler")
    ui_img2img_scheduler = get_a1111_ui_component("img2img", "scheduler")

    ui_txt2img_width = get_a1111_ui_component("txt2img", "Size-1")
    ui_img2img_width = get_a1111_ui_component("img2img", "Size-1")
    ui_txt2img_height = get_a1111_ui_component("txt2img", "Size-2")
    ui_img2img_height = get_a1111_ui_component("img2img", "Size-2")

    ui_txt2img_cfg = get_a1111_ui_component("txt2img", "CFG scale")
    ui_txt2img_hr_cfg = get_a1111_ui_component("txt2img", "Hires CFG Scale")
    ui_img2img_cfg = get_a1111_ui_component("img2img", "CFG scale")

    ui_txt2img_distilled_cfg = get_a1111_ui_component("txt2img", "Distilled CFG Scale")
    ui_txt2img_hr_distilled_cfg = get_a1111_ui_component("txt2img", "Hires Distilled CFG Scale")
    ui_img2img_distilled_cfg = get_a1111_ui_component("img2img", "Distilled CFG Scale")

    ui_txt2img_batch_size = get_a1111_ui_component("txt2img", "Batch size")
    ui_img2img_batch_size = get_a1111_ui_component("img2img", "Batch size")

    output_targets = [
        ui_checkpoint,
        ui_vae,
        ui_forge_unet_dtype,
        ui_txt2img_steps,
        ui_txt2img_hr_steps,
        ui_img2img_steps,
        ui_txt2img_sampler,
        ui_img2img_sampler,
        ui_txt2img_scheduler,
        ui_img2img_scheduler,
        ui_txt2img_width,
        ui_img2img_width,
        ui_txt2img_height,
        ui_img2img_height,
        ui_txt2img_cfg,
        ui_txt2img_hr_cfg,
        ui_img2img_cfg,
        ui_txt2img_distilled_cfg,
        ui_txt2img_hr_distilled_cfg,
        ui_img2img_distilled_cfg,
        ui_txt2img_batch_size,
        ui_img2img_batch_size,
    ]

    ui_forge_preset.change(on_preset_change, inputs=[ui_forge_preset], outputs=output_targets, queue=False, show_progress=False).success(
        fn=_load_presets,
        inputs=[ui_checkpoint, ui_vae, ui_forge_unet_dtype, ui_forge_preset],
        queue=False,
        show_progress=False,
    ).then(js="clickLoraRefresh", fn=None, queue=False, show_progress=False)
    _event_load.then(on_preset_change, inputs=[ui_forge_preset], outputs=output_targets, queue=False, show_progress=False)

    refresh_model_loading_parameters()
    _refreshed_callback()


def _load_presets(ui_checkpoint: str, ui_vae: list[str], ui_forge_unet_dtype: str, ui_forge_preset: str):
    dtype_change(ui_forge_unet_dtype, ui_forge_preset, save=False, refresh=False)
    modules_change(ui_vae, ui_forge_preset, save=False, refresh=False)
    checkpoint_change(ui_checkpoint, ui_forge_preset, save=False, refresh=False)
    shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters()


def _parse_value(preset: "Preset", default: "Preset", *, s: str, k: str):
    v1, v2 = getattr(getattr(preset, s), k), getattr(getattr(default, s), k)
    v: int | float = v1 if v1 is not None else v2
    return gr.skip() if v is None or v <= 0 else gr.update(value=v)


def _parse_cfg2(preset: "Preset", default: "Preset", *, s: str):
    if use_distill(preset, default):
        args = {"visible": True, "label": "Distilled CFG Scale"}
        v1, v2 = getattr(preset, s).dcfg, getattr(default, s).dcfg
    elif use_shift(preset, default):
        args = {"visible": True, "label": "Shift"}
        v1, v2 = getattr(preset, s).shift, getattr(default, s).shift
    else:
        return gr.update(visible=False)

    v: float = v1 if v1 is not None else v2
    val = {} if v is None or v <= 0 else {"value": v}
    return gr.update(**val, **args)


def on_preset_change(preset_name: str):
    assert preset_name is not None
    shared.opts.set("forge_preset", preset_name)
    shared.opts.save(shared.config_filename)

    preset: Preset = ALL_PRESETS[preset_name]
    default: Preset = DEFAULTS[preset.arch]

    models: dict[str, str | list[str]] = shared.opts.forge_preset_models.get(preset.name, {})
    checkpoint = models.get("checkpoint", shared.opts.sd_model_checkpoint)
    modules = models.get("modules", [])
    dtype = models.get("dtype", "Automatic")

    if (fps := is_video(preset, default)) > 1:
        batch_args_t2i = {"minimum": 1, "maximum": fps * 15 + 1, "step": fps, "label": "Frames", "value": preset.t2i.batch_size or default.t2i.batch_size or 1}
    else:
        batch_args_t2i = {"minimum": 1, "maximum": 8, "step": 1, "label": "Batch Size", "value": preset.t2i.batch_size or default.t2i.batch_size or 1}

    batch_args_i2i = batch_args_t2i.copy()
    batch_args_i2i["value"] = preset.i2i.batch_size or default.i2i.batch_size or 1

    modules_value = [os.path.basename(m) for m in modules]
    if not shared.opts.sd_modules_dropdown_use_short:
        modules_value = [module_map[m] for m in modules_value if m in module_map]

    return [
        # ui_checkpoint, ui_vae, ui_forge_unet_dtype
        gr.update(value=checkpoint),
        gr.update(value=modules_value),
        gr.update(value=dtype),
        # ui_txt2img_steps, ui_txt2img_hr_steps, ui_img2img_steps
        _parse_value(preset, default, s="t2i", k="step"),
        _parse_value(preset, default, s="hr", k="step"),
        _parse_value(preset, default, s="i2i", k="step"),
        # ui_txt2img_sampler, ui_img2img_sampler, ui_txt2img_scheduler, ui_img2img_scheduler
        gr.update(value=preset.t2i.sampler or default.t2i.sampler),
        gr.update(value=preset.i2i.sampler or default.i2i.sampler),
        gr.update(value=preset.t2i.scheduler or default.t2i.scheduler),
        gr.update(value=preset.i2i.scheduler or default.i2i.scheduler),
        # ui_txt2img_width, ui_img2img_width, ui_txt2img_height, ui_img2img_height
        _parse_value(preset, default, s="t2i", k="width"),
        _parse_value(preset, default, s="i2i", k="width"),
        _parse_value(preset, default, s="t2i", k="height"),
        _parse_value(preset, default, s="i2i", k="height"),
        # ui_txt2img_cfg, ui_txt2img_hr_cfg, ui_img2img_cfg
        _parse_value(preset, default, s="t2i", k="cfg"),
        _parse_value(preset, default, s="hr", k="cfg"),
        _parse_value(preset, default, s="i2i", k="cfg"),
        # ui_txt2img_distilled_cfg, ui_txt2img_hr_distilled_cfg, ui_img2img_distilled_cfg
        _parse_cfg2(preset, default, s="t2i"),
        _parse_cfg2(preset, default, s="hr"),
        _parse_cfg2(preset, default, s="i2i"),
        # ui_txt2img_batch_size, ui_img2img_batch_size
        gr.update(**batch_args_t2i),
        gr.update(**batch_args_i2i),
    ]


# region Callback


_CALLBACK_SET = False

_refresh_callbacks: list[tuple[Callable, list[gr.components.Component], list[gr.components.Component]]] = []


def capture_callbacks():
    global _CALLBACK_SET
    _CALLBACK_SET = False
    _refresh_callbacks.clear()


def on_refresh_checkpoint(callback: Callable, inputs: list[gr.components.Component] = None, outputs: list[gr.components.Component] = None):
    """register a function to be called after the refresh button for Checkpoint / VAE / Text Encoder is clicked"""
    if _CALLBACK_SET:
        raise RuntimeError("Cannot register callback after forge_main_entry")
    _refresh_callbacks.append((callback, inputs, outputs))


def _refreshed_callback():
    global _CALLBACK_SET

    event = _event_refresh

    for callback, inputs, outputs in _refresh_callbacks:
        event = event.then(fn=callback, inputs=inputs, outputs=outputs, queue=False, show_progress=False)

    _CALLBACK_SET = True
