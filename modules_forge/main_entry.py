import os
import torch
import gradio as gr

from gradio.context import Context
from modules import shared_items, shared, ui_common, sd_models, processing, infotext_utils, paths, ui_loadsave
from backend import memory_management, stream
from backend.args import dynamic_args
from modules.shared import cmd_opts
from modules_forge.presets import PresetArch, register_ui_presets


total_vram = int(memory_management.total_vram)

ui_forge_preset: gr.Radio = None

ui_checkpoint: gr.Dropdown = None
ui_vae: gr.Dropdown = None
ui_clip_skip: gr.Slider = None

ui_forge_unet_storage_dtype_options: gr.Radio = None
ui_forge_async_loading: gr.Radio = None
ui_forge_pin_shared_memory: gr.Radio = None
ui_forge_inference_memory: gr.Slider = None



forge_unet_storage_dtype_options = {
    'Automatic': (None, False),
    'Automatic (fp16 LoRA)': (None, True),
    'bnb-nf4': ('nf4', False),
    'bnb-nf4 (fp16 LoRA)': ('nf4', True),
    'float8-e4m3fn': (torch.float8_e4m3fn, False),
    'float8-e4m3fn (fp16 LoRA)': (torch.float8_e4m3fn, True),
    'bnb-fp4': ('fp4', False),
    'bnb-fp4 (fp16 LoRA)': ('fp4', True),
    'float8-e5m2': (torch.float8_e5m2, False),
    'float8-e5m2 (fp16 LoRA)': (torch.float8_e5m2, True),
}

module_list = {}


def bind_to_opts(comp, k, save=False, callback=None):
    def on_change(v):
        shared.opts.set(k, v)
        if save:
            shared.opts.save(shared.config_filename)
        if callback is not None:
            callback()
        return

    comp.change(on_change, inputs=[comp], queue=False, show_progress=False)
    return


def make_checkpoint_manager_ui():
    global ui_checkpoint, ui_vae, ui_clip_skip, ui_forge_unet_storage_dtype_options, ui_forge_async_loading, ui_forge_pin_shared_memory, ui_forge_inference_memory, ui_forge_preset

    if shared.opts.sd_model_checkpoint in [None, 'None', 'none', '']:
        if len(sd_models.checkpoints_list) == 0:
            sd_models.list_models()
        if len(sd_models.checkpoints_list) > 0:
            shared.opts.set('sd_model_checkpoint', next(iter(sd_models.checkpoints_list.values())).name)

    ui_forge_preset = gr.Radio(label="UI Preset", value=lambda: shared.opts.forge_preset, choices=PresetArch.choices(), elem_id="forge_ui_preset")

    ui_checkpoint = gr.Dropdown(
        value=None,
        label="Checkpoint / UNet",
        elem_classes=['model_selection'],
        choices=None
    )

    ui_vae = gr.Dropdown(
        value=None,
        multiselect=True,
        label="VAE / Text Encoder",
        choices=None
    )

    def gr_refresh_models():
        ckpt_list, vae_list = refresh_models()
        return gr.update(choices=ckpt_list), gr.update(choices=vae_list)

    refresh_button = ui_common.ToolButton(value=ui_common.refresh_symbol, elem_id=f"forge_refresh_checkpoint", tooltip="Refresh")
    refresh_button.click(
        fn=gr_refresh_models,
        outputs=[ui_checkpoint, ui_vae],
        queue=False
    )

    def _on_load_models():
        ckpt_list, vae_list = refresh_models()
        return (
            gr.update(
                value=shared.opts.sd_model_checkpoint,
                choices=ckpt_list
            ), gr.update(
                value=[os.path.basename(x) for x in shared.opts.forge_additional_modules],
                choices=vae_list
            )
        )

    Context.root_block.load(
        fn=_on_load_models,
        outputs=[ui_checkpoint, ui_vae],
        show_progress=False,
        queue=False
    )

    ui_forge_unet_storage_dtype_options = gr.Dropdown(label="Diffusion in Low Bits", value=lambda: shared.opts.forge_unet_storage_dtype, choices=list(forge_unet_storage_dtype_options.keys()))
    bind_to_opts(ui_forge_unet_storage_dtype_options, 'forge_unet_storage_dtype', save=True, callback=refresh_model_loading_parameters)

    ui_forge_async_loading = gr.Radio(label="Swap Method", value=lambda: shared.opts.forge_async_loading, choices=['Queue', 'Async'])
    ui_forge_pin_shared_memory = gr.Radio(label="Swap Location", value=lambda: shared.opts.forge_pin_shared_memory, choices=['CPU', 'Shared'])
    ui_forge_inference_memory = gr.Slider(label="GPU Weights (MB)", value=lambda: total_vram - shared.opts.forge_inference_memory, minimum=0, maximum=int(memory_management.total_vram), step=1)

    mem_comps = [ui_forge_inference_memory, ui_forge_async_loading, ui_forge_pin_shared_memory]

    ui_forge_inference_memory.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)
    ui_forge_async_loading.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)
    ui_forge_pin_shared_memory.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)

    ui_clip_skip = gr.Slider(label="Clip skip", value=lambda: shared.opts.CLIP_stop_at_last_layers, **{"minimum": 1, "maximum": 12, "step": 1})
    bind_to_opts(ui_clip_skip, 'CLIP_stop_at_last_layers', save=True)

    ui_checkpoint.change(checkpoint_change, inputs=[ui_checkpoint, ui_forge_preset], show_progress=False)
    ui_vae.change(modules_change, inputs=[ui_vae, ui_forge_preset], queue=False, show_progress=False)

    return


def find_files_with_extensions(base_path, extensions):
    found_files = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, file)
                found_files[file] = full_path
    return found_files


def refresh_models():
    global module_list

    shared_items.refresh_checkpoints()
    ckpt_list = shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)

    file_extensions = ['ckpt', 'pt', 'pth', 'bin', 'safetensors', 'sft', 'gguf']

    module_list.clear()

    module_paths = [
        os.path.abspath(os.path.join(paths.models_path, "VAE")),
        os.path.abspath(os.path.join(paths.models_path, "text_encoder")),
    ]

    if isinstance(shared.cmd_opts.vae_dir, str):
        module_paths.append(os.path.abspath(shared.cmd_opts.vae_dir))
    if isinstance(shared.cmd_opts.text_encoder_dir, str):
        module_paths.append(os.path.abspath(shared.cmd_opts.text_encoder_dir))

    for vae_path in module_paths:
        vae_files = find_files_with_extensions(vae_path, file_extensions)
        module_list.update(vae_files)

    return ckpt_list, module_list.keys()


def ui_refresh_memory_management_settings(model_memory, async_loading, pin_shared_memory):
    """ Passes precalculated 'model_memory' from "GPU Weights" UI slider (skip redundant calculation) """
    refresh_memory_management_settings(
        async_loading=async_loading,
        pin_shared_memory=pin_shared_memory,
        model_memory=model_memory  # Use model_memory directly from UI slider value
    )

def refresh_memory_management_settings(async_loading=None, inference_memory=None, pin_shared_memory=None, model_memory=None):
    # Fallback to defaults if values are not passed
    async_loading = async_loading if async_loading is not None else shared.opts.forge_async_loading
    inference_memory = inference_memory if inference_memory is not None else shared.opts.forge_inference_memory
    pin_shared_memory = pin_shared_memory if pin_shared_memory is not None else shared.opts.forge_pin_shared_memory

    # If model_memory is provided, calculate inference memory accordingly, otherwise use inference_memory directly
    if model_memory is None:
        model_memory = total_vram - inference_memory
    else:
        inference_memory = total_vram - model_memory

    shared.opts.set('forge_async_loading', async_loading)
    shared.opts.set('forge_inference_memory', inference_memory)
    shared.opts.set('forge_pin_shared_memory', pin_shared_memory)

    stream.stream_activated = async_loading == 'Async'
    memory_management.current_inference_memory = inference_memory * 1024 * 1024  # Convert MB to bytes
    memory_management.PIN_SHARED_MEMORY = pin_shared_memory == 'Shared'

    log_dict = dict(
        stream=stream.should_use_stream(),
        inference_memory=memory_management.minimum_inference_memory() / (1024 * 1024),
        pin_shared_memory=memory_management.PIN_SHARED_MEMORY
    )

    print(f'Environment vars changed: {log_dict}')

    if inference_memory < min(512, total_vram * 0.05):
        print('------------------')
        print(f'[Low VRAM Warning] You just set Forge to use 100% GPU memory ({model_memory:.2f} MB) to load model weights.')
        print('[Low VRAM Warning] This means you will have 0% GPU memory (0.00 MB) to do matrix computation. Computations may fallback to CPU or go Out of Memory.')
        print('[Low VRAM Warning] In many cases, image generation will be 10x slower.')
        print("[Low VRAM Warning] To solve the problem, you can set the 'GPU Weights' (on the top of page) to a lower value.")
        print("[Low VRAM Warning] If you cannot find 'GPU Weights', you can click the 'all' option in the 'UI' area on the left-top corner of the webpage.")
        print('[Low VRAM Warning] Make sure that you know what you are testing.')
        print('------------------')
    else:
        compute_percentage = (inference_memory / total_vram) * 100.0
        print(f'[GPU Setting] You will use {(100 - compute_percentage):.2f}% GPU memory ({model_memory:.2f} MB) to load weights, and use {compute_percentage:.2f}% GPU memory ({inference_memory:.2f} MB) to do matrix computation.')

    processing.need_global_unload = True
    return


def refresh_model_loading_parameters():
    from modules.sd_models import select_checkpoint, model_data

    checkpoint_info = select_checkpoint()

    unet_storage_dtype, lora_fp16 = forge_unet_storage_dtype_options.get(shared.opts.forge_unet_storage_dtype, (None, False))

    dynamic_args['online_lora'] = lora_fp16

    model_data.forge_loading_parameters = dict(
        checkpoint_info=checkpoint_info,
        additional_modules=shared.opts.forge_additional_modules,
        unet_storage_dtype=unet_storage_dtype
    )

    print(f'Model selected: {model_data.forge_loading_parameters}')
    print(f'Using online LoRAs in FP16: {lora_fp16}')
    processing.need_global_unload = True

    return


def checkpoint_change(ckpt_name:str, preset:str, save=True, refresh=True):
    """ checkpoint name can be a number of valid aliases. Returns True if checkpoint changed. """
    new_ckpt_info = sd_models.get_closet_checkpoint_match(ckpt_name)
    current_ckpt_info = sd_models.get_closet_checkpoint_match(shared.opts.data.get('sd_model_checkpoint', ''))
    if new_ckpt_info == current_ckpt_info:
        return False

    shared.opts.set('sd_model_checkpoint', ckpt_name)
    shared.opts.set(f'forge_checkpoint_{preset}', ckpt_name)

    if save:
        shared.opts.save(shared.config_filename)
    if refresh:
        refresh_model_loading_parameters()
    return True


def modules_change(module_values:list, preset:str, save=True, refresh=True) -> bool:
    """ module values may be provided as file paths, or just the module names. Returns True if modules changed. """
    modules = []
    for v in module_values:
        module_name = os.path.basename(v) # If the input is a filepath, extract the file name
        if module_name in module_list:
            modules.append(module_list[module_name])

    # skip further processing if value unchanged
    if sorted(modules) == sorted(shared.opts.data.get('forge_additional_modules', [])):
        return False

    shared.opts.set(f'forge_additional_modules_{preset}', modules)

    if save:
        shared.opts.save(shared.config_filename)
    if refresh:
        refresh_model_loading_parameters()
    return True


def get_a1111_ui_component(tab, label):
    fields = infotext_utils.paste_fields[tab]['fields']
    for f in fields:
        if f.label == label or f.api == label:
            return f.component


def forge_main_entry():
    ui_txt2img_width = get_a1111_ui_component('txt2img', 'Size-1')
    ui_txt2img_height = get_a1111_ui_component('txt2img', 'Size-2')
    ui_txt2img_cfg = get_a1111_ui_component('txt2img', 'CFG scale')
    ui_txt2img_distilled_cfg = get_a1111_ui_component('txt2img', 'Distilled CFG Scale')
    ui_txt2img_sampler = get_a1111_ui_component('txt2img', 'sampler_name')
    ui_txt2img_scheduler = get_a1111_ui_component('txt2img', 'scheduler')

    ui_img2img_width = get_a1111_ui_component('img2img', 'Size-1')
    ui_img2img_height = get_a1111_ui_component('img2img', 'Size-2')
    ui_img2img_cfg = get_a1111_ui_component('img2img', 'CFG scale')
    ui_img2img_distilled_cfg = get_a1111_ui_component('img2img', 'Distilled CFG Scale')
    ui_img2img_sampler = get_a1111_ui_component('img2img', 'sampler_name')
    ui_img2img_scheduler = get_a1111_ui_component('img2img', 'scheduler')

    ui_txt2img_hr_cfg = get_a1111_ui_component('txt2img', 'Hires CFG Scale')
    ui_txt2img_hr_distilled_cfg = get_a1111_ui_component('txt2img', 'Hires Distilled CFG Scale')

    output_targets = [
        ui_checkpoint,
        ui_vae,
        ui_clip_skip,
        ui_forge_unet_storage_dtype_options,
        ui_forge_async_loading,
        ui_forge_pin_shared_memory,
        ui_forge_inference_memory,
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
    ]

    ui_forge_preset.change(on_preset_change, inputs=[ui_forge_preset], outputs=output_targets, queue=False, show_progress=False)
    ui_forge_preset.change(js="clickLoraRefresh", fn=None, queue=False, show_progress=False)
    Context.root_block.load(on_preset_change, inputs=[ui_forge_preset], outputs=output_targets, queue=False, show_progress=False)

    refresh_model_loading_parameters()
    return


def on_preset_change(preset):
    assert preset is not None
    shared.opts.set('forge_preset', preset)
    shared.opts.save(shared.config_filename)

    model_mem = getattr(shared.opts, f"{preset}_gpu_mb", total_vram - 1024)
    if model_mem < 0 or model_mem > total_vram:
        model_mem = total_vram - 1024

    show_clip_skip = (preset != "flux")
    show_basic_mem = (preset != "sd")
    show_adv_mem = (preset == "flux")
    distilled = (preset == "flux")

    return [
        gr.update(value=getattr(shared.opts, f"forge_checkpoint_{preset}", shared.opts.sd_model_checkpoint)),   # ui_checkpoint
        gr.update(visible=True, value=getattr(shared.opts, f"forge_additional_modules_{preset}", [])),          # ui_vae
        gr.update(visible=show_clip_skip, value=getattr(shared.opts, "CLIP_stop_at_last_layers", 1)),           # ui_clip_skip
        gr.update(visible=show_basic_mem, value=getattr(shared.opts, "forge_unet_storage_dtype", 'Automatic')), # ui_forge_unet_storage_dtype_options
        gr.update(visible=show_adv_mem, value=getattr(shared.opts, "forge_async_loading", 'Queue')),            # ui_forge_async_loading
        gr.update(visible=show_adv_mem, value=getattr(shared.opts, "forge_pin_shared_memory", 'CPU')),          # ui_forge_pin_shared_memory
        gr.update(visible=show_basic_mem, value=model_mem),                                                     # ui_forge_inference_memory
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_width", 896)),                                      # ui_txt2img_width
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_width", 1024)),                                     # ui_img2img_width
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_height", 1152)),                                    # ui_txt2img_height
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_height", 1024)),                                    # ui_img2img_height
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_cfg", 1.0)),                                        # ui_txt2img_cfg
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_cfg", 1.0)),                                        # ui_img2img_cfg
        gr.update(visible=distilled, value=getattr(shared.opts, f"{preset}_t2i_d_cfg", 3.5)),                   # ui_txt2img_distilled_cfg
        gr.update(visible=distilled, value=getattr(shared.opts, f"{preset}_i2i_d_cfg", 3.5)),                   # ui_img2img_distilled_cfg
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_sampler", 'Euler')),                                # ui_txt2img_sampler
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_sampler", 'Euler')),                                # ui_img2img_sampler
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_scheduler", 'Simple')),                             # ui_txt2img_scheduler
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_scheduler", 'Simple')),                             # ui_img2img_scheduler
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_hr_cfg", 1.0)),                                     # ui_txt2img_hr_cfg
        gr.update(visible=distilled, value=getattr(shared.opts, f"{preset}_t2i_hr_d_cfg", 3.5)),                # ui_txt2img_hr_distilled_cfg
    ]


register_ui_presets(total_vram)
