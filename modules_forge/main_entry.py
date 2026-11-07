import logging
import os.path
from typing import Callable, Optional, OrderedDict

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
from modules_forge.presets import PresetArch, is_video, use_distill, use_shift

logger = logging.getLogger("ui_models")
setup_logger(logger)

ui_forge_preset: gr.Radio
ui_checkpoint: gr.Dropdown
ui_vae: gr.Dropdown
ui_forge_unet_dtype: gr.Radio
_event_refresh: Dependency

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
    global ui_forge_preset, ui_checkpoint, ui_vae, ui_forge_unet_dtype, _event_refresh

    if shared.opts.sd_model_checkpoint in [None, "None", "none", ""]:
        if len(sd_models.checkpoints_list) == 0:
            sd_models.list_models()
        if len(sd_models.checkpoints_list) > 0:
            shared.opts.set("sd_model_checkpoint", next(iter(sd_models.checkpoints_list.values())).name)

    def preset_choices():
        presets = [p for p in PresetArch.choices() if p not in shared.cmd_opts.disable_preset]
        return sorted(presets) if shared.cmd_opts.sort_preset else presets

    ui_forge_preset = gr.Dropdown(label="UI Preset", value=shared.opts.forge_preset, choices=preset_choices(), elem_id="forge_ui_preset")

    ui_checkpoint = gr.Dropdown(label="Checkpoint", value=None, choices=None, elem_id="setting_sd_model_checkpoint", elem_classes=["model_selection"])

    ui_vae = gr.Dropdown(label="VAE / Text Encoder", value=None, choices=None, multiselect=True, elem_id="setting_sd_modules", elem_classes=["model_selection"])

    def refresh_model_list():
        ckpt_list, vae_list = refresh_models()
        return [gr.update(choices=ckpt_list), gr.update(choices=vae_list)]

    refresh_button = ui_common.ToolButton(value=ui_common.refresh_symbol, elem_id="forge_refresh_checkpoint", tooltip="Refresh")
    _event_refresh = refresh_button.click(fn=refresh_model_list, outputs=[ui_checkpoint, ui_vae], queue=False)
    Context.root_block.load(fn=refresh_model_list, outputs=[ui_checkpoint, ui_vae], queue=False)

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


def resolve_dtype(dtype_name: str, checkpoint_info) -> tuple[torch.dtype, bool, bool]:
    """A dtype option carries both the storage dtype and the on-the-fly LoRA mode; they must be restored together.
    Returns (unet_storage_dtype, online_lora, gguf_overridden)"""
    unet_storage_dtype, lora_fp16 = forge_unet_storage_dtype_options.get(dtype_name, (None, False))

    filename = getattr(checkpoint_info, "filename", "") or ""
    gguf_overridden = filename.endswith(("gguf", "GGUF")) and not lora_fp16

    return unet_storage_dtype, True if gguf_overridden else lora_fp16, gguf_overridden


def refresh_model_loading_parameters(*, refresh: bool = True, disable_log: bool = False, checkpoint=None, modules=None, dtype=None):
    """checkpoint/modules/dtype default to shared.opts; task-scoped callers must pass snapshot values,
    otherwise a mid-job UI switch would be taken as the value to load"""
    if not refresh:
        return

    from modules.sd_models import model_data, select_checkpoint

    if checkpoint is None:
        checkpoint_info = select_checkpoint()
        if checkpoint_info is None:
            logger.critical('You do not have any model... Please download models to "models/Stable-diffusion"')
            return
    else:
        checkpoint_info = sd_models.get_closet_checkpoint_match(checkpoint) if isinstance(checkpoint, str) else checkpoint
        if checkpoint_info is None:
            logger.warning(f'Checkpoint "{checkpoint}" not found ; loading parameters unchanged')
            return

    if modules is None:
        modules = shared.opts.forge_additional_modules
    if dtype is None:
        dtype = shared.opts.forge_unet_storage_dtype

    unet_storage_dtype, lora_fp16, gguf_overridden = resolve_dtype(dtype, checkpoint_info)

    # key order must match enforce_task_loading_parameters / xyz_grid: forge_hash compares str(dict)
    model_data.forge_loading_parameters = dict(checkpoint_info=checkpoint_info, additional_modules=list(modules), unet_storage_dtype=unet_storage_dtype)

    ckpt: str = checkpoint_info.filename
    module_names: list[str] = [os.path.basename(x) for x in modules]
    dtype_label = str(unet_storage_dtype or [torch.float16, torch.bfloat16])

    if not disable_log:
        logger.info("Model Selected:")
        print_json(data=dict(checkpoint=os.path.basename(ckpt), modules=module_names, dtype=dtype_label))

        if gguf_overridden:
            logger.warning("GGUF requires fp16 LoRA ; overriding option")

    dynamic_args.online_lora = lora_fp16

    if not disable_log:
        logger.info(f"Patch LoRAs on-the-fly: {lora_fp16}")
        if not ckpt.endswith(("gguf", "GGUF")) and lora_fp16:
            logger.warning("on-the-fly WILL be slower ; enable only if you know what you are doing")

    processing.need_global_unload = True


def snapshot_task_loading_parameters() -> dict:
    """Capture the effective model loading parameters after overrides as the task-scoped source of truth; never written back to shared.opts.

    All three keys are captured unconditionally: a plain generation has no override, but is equally affected by UI switches during sampling"""
    return dict(
        sd_model_checkpoint=shared.opts.sd_model_checkpoint,
        forge_additional_modules=list(shared.opts.forge_additional_modules or []),
        forge_unet_storage_dtype=shared.opts.forge_unet_storage_dtype,
    )


def enforce_task_loading_parameters(task_params: dict) -> Optional[dict]:
    """Build loading parameters from the task snapshot, store them in the slot and return them for forge_model_reload
    to consume, so loading no longer depends on a shared slot that UI callbacks can rewrite.
    Returns None on an invalid snapshot to keep current behaviour"""
    if not task_params:
        return None

    checkpoint_value = task_params.get("sd_model_checkpoint")
    checkpoint_info = sd_models.get_closet_checkpoint_match(checkpoint_value) if checkpoint_value else None
    if checkpoint_info is None:
        # the snapshotted model no longer exists (e.g. deleted while generating): fall back to the current slot so the job is not aborted
        checkpoint_info = sd_models.model_data.forge_loading_parameters.get("checkpoint_info")
    if checkpoint_info is None:
        return None

    modules = list(task_params.get("forge_additional_modules", shared.opts.forge_additional_modules) or [])
    dtype = task_params.get("forge_unet_storage_dtype", shared.opts.forge_unet_storage_dtype)
    unet_storage_dtype, lora_fp16, _ = resolve_dtype(dtype, checkpoint_info)

    params = dict(checkpoint_info=checkpoint_info, additional_modules=modules, unet_storage_dtype=unet_storage_dtype)

    current = sd_models.model_data.forge_loading_parameters
    changed = (
        getattr(current.get("checkpoint_info"), "filename", None) != checkpoint_info.filename
        or current.get("additional_modules") != modules
        or current.get("unet_storage_dtype") != unet_storage_dtype
    )

    # only the slot and the LoRA mode are rewritten, never need_global_unload: enforcing restores the pre-override state, it is not a new memory intent
    if changed:
        sd_models.model_data.forge_loading_parameters = params
        logger.info(f"Task-scoped model loading parameters enforced: {task_params}")

    dynamic_args.online_lora = lora_fp16

    return params


def checkpoint_change(ckpt_name: str, preset: str, save=True, refresh=True) -> bool:
    """`ckpt_name` accepts valid aliases; returns `True` if checkpoint changed"""
    new_ckpt_info = sd_models.get_closet_checkpoint_match(ckpt_name)
    current_ckpt_info = sd_models.get_closet_checkpoint_match(getattr(shared.opts, "sd_model_checkpoint", ""))
    if new_ckpt_info == current_ckpt_info:
        return False

    shared.opts.set("sd_model_checkpoint", ckpt_name)
    if preset is not None:
        shared.opts.set(f"forge_checkpoint_{preset}", ckpt_name)

    if save:
        shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters(refresh=refresh)
    return True


def normalize_modules(module_values: list) -> list[str]:
    """Accept file paths or module names; return the normalized module list (unknown entries filtered out, sorted)"""
    modules = []
    for v in module_values:
        module_name = os.path.basename(v)  # If the input is a filepath, extract the filename
        if module_name in module_list:
            modules.append(module_list[module_name])
    modules.sort()
    return modules


def modules_change(module_values: list, preset: str, save=True, refresh=True) -> bool:
    """`module_values` accepts file paths or just the module names; returns `True` if modules changed"""
    modules = normalize_modules(module_values)

    # skip further processing if value unchanged
    if modules == getattr(shared.opts, "forge_additional_modules", []):
        return False

    shared.opts.set("forge_additional_modules", modules)
    if preset is not None:
        shared.opts.set(f"forge_additional_modules_{preset}", modules)

    if save:
        shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters(refresh=refresh)
    return True


def dtype_change(dtype: str, preset: str, save=True, refresh=True) -> bool:
    shared.opts.set("forge_unet_storage_dtype", dtype)
    if preset is not None:
        shared.opts.set(f"forge_unet_storage_dtype_{preset}", dtype)

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
    Context.root_block.load(on_preset_change, inputs=[ui_forge_preset], outputs=output_targets, queue=False, show_progress=False)

    refresh_model_loading_parameters()
    _refreshed_callback()


def _load_presets(ui_checkpoint: str, ui_vae: list[str], ui_forge_unet_dtype: str, ui_forge_preset: str):
    dtype_change(ui_forge_unet_dtype, ui_forge_preset, save=False, refresh=False)
    modules_change(ui_vae, ui_forge_preset, save=False, refresh=False)
    checkpoint_change(ui_checkpoint, ui_forge_preset, save=True, refresh=True)


def on_preset_change(preset: str):
    assert preset is not None
    shared.opts.set("forge_preset", preset)
    shared.opts.save(shared.config_filename)

    if use_shift(preset):
        d_args = {"visible": getattr(shared.opts, f"{preset}_show_shift", True), "label": "Shift"}
    elif use_distill(preset):
        d_args = {"visible": True, "label": "Distilled CFG Scale"}
    else:
        d_args = {"visible": False}

    if (fps := is_video(preset)) > 1:
        batch_args_t2i = {"minimum": 1, "maximum": fps * 15 + 1, "step": fps, "label": "Frames", "value": getattr(shared.opts, f"{preset}_t2i_batch_size", 1)}
    else:
        batch_args_t2i = {"minimum": 1, "maximum": 8, "step": 1, "label": "Batch Size", "value": getattr(shared.opts, f"{preset}_t2i_batch_size", 1)}

    batch_args_i2i = batch_args_t2i.copy()
    batch_args_i2i["value"] = getattr(shared.opts, f"{preset}_i2i_batch_size", 1)

    modules_value = [os.path.basename(m) for m in getattr(shared.opts, f"forge_additional_modules_{preset}", [])]
    if not shared.opts.sd_modules_dropdown_use_short:
        modules_value = [module_map[m] for m in modules_value]

    return [
        # ui_checkpoint, ui_vae, ui_forge_unet_dtype
        gr.update(value=getattr(shared.opts, f"forge_checkpoint_{preset}", shared.opts.sd_model_checkpoint)),
        gr.update(value=modules_value),
        gr.update(value=getattr(shared.opts, f"forge_unet_storage_dtype_{preset}", "Automatic")),
        # ui_txt2img_steps, ui_txt2img_hr_steps, ui_img2img_steps
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_step", 20)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_hr_step", 20)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_i2i_step", 20)) > 0 else gr.skip(),
        # ui_txt2img_sampler, ui_img2img_sampler, ui_txt2img_scheduler, ui_img2img_scheduler
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_sampler", "Euler")),
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_sampler", "Euler")),
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_scheduler", "Simple")),
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_scheduler", "Simple")),
        # ui_txt2img_width, ui_img2img_width, ui_txt2img_height, ui_img2img_height
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_width", 1024)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_i2i_width", 1024)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_height", 1024)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_i2i_height", 1024)) > 0 else gr.skip(),
        # ui_txt2img_cfg, ui_txt2img_hr_cfg, ui_img2img_cfg
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_cfg", 1.0)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_hr_cfg", 1.0)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_i2i_cfg", 1.0)) > 0 else gr.skip(),
        # ui_txt2img_distilled_cfg, ui_img2img_distilled_cfg, ui_txt2img_hr_distilled_cfg
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_dcfg", 3.0), **d_args),
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_hr_dcfg", 3.0), **d_args),
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_dcfg", 3.0), **d_args),
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
