import json
from contextlib import closing

import modules.scripts
from modules import processing, infotext_utils
from modules.infotext_utils import create_override_settings_dict, parse_generation_parameters
from modules.shared import opts
import modules.shared as shared
from modules.ui import plaintext_to_html
import gradio as gr
from modules_forge import main_thread


def txt2img_create_processing(
    id_task: str,
    request: gr.Request,
    prompt: str,
    negative_prompt: str,
    prompt_styles,
    n_iter: int,
    batch_size: int,
    cfg_scale: float,
    height: int,
    width: int,
    enable_hr: bool,
    denoising_strength: float,
    hr_scale: float,
    hr_upscaler: str,
    hr_second_pass_steps: int,
    hr_resize_x: int,
    hr_resize_y: int,
    hr_checkpoint_name: str,
    hr_sampler_name: str,
    hr_scheduler: str,
    hr_cfg_scale: float,
    hr_rescale_cfg: float,
    hr_prompt: str,
    hr_negative_prompt: str,
    override_settings_texts,
    *args,
    force_enable_hr=False,
):
    override_settings = create_override_settings_dict(override_settings_texts)

    if force_enable_hr:
        enable_hr = True

    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        batch_size=batch_size,
        n_iter=n_iter,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        hr_checkpoint_name=None if hr_checkpoint_name == "Use same checkpoint" else hr_checkpoint_name,
        hr_sampler_name=None if hr_sampler_name == "Use same sampler" else hr_sampler_name,
        hr_scheduler=None if hr_scheduler == "Use same scheduler" else hr_scheduler,
        hr_cfg_scale=hr_cfg_scale if opts.hires_fix_show_sampler else cfg_scale,
        hr_rescale_cfg=hr_rescale_cfg if opts.hires_fix_show_sampler else None,
        hr_prompt=hr_prompt,
        hr_negative_prompt=hr_negative_prompt,
        override_settings=override_settings,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args

    p.user = request.username

    if shared.opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    return p


def txt2img_upscale_function(id_task: str, request: gr.Request, gallery: list[dict], gallery_index: int, generation_info: str, *args):
    _gallery = [infotext_utils.image_from_url_text(info) for info in gallery]

    if len(gallery) == 0:
        return _gallery, generation_info, "No image to upscale...", ""
    if not (0 <= gallery_index < len(gallery)):
        return _gallery, generation_info, f"Bad Index: {gallery_index}", ""
    if len(gallery) > 1 and opts.return_grid and gallery_index == 0:
        return _gallery, generation_info, "Cannot upscale the grid image...", ""

    p = txt2img_create_processing(id_task, request, *args, force_enable_hr=True)
    p.batch_size = 1
    p.n_iter = 1
    p.txt2img_upscale = True

    p.firstpass_image = _gallery[gallery_index]
    p.width, p.height = p.firstpass_image.size

    geninfo = json.loads(generation_info)
    parameters = parse_generation_parameters(geninfo.get("infotexts")[gallery_index], [])
    p.seed = parameters.get("Seed", -1)
    p.subseed = parameters.get("Variation seed", -1)

    p.override_settings["save_images_before_highres_fix"] = False

    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)

        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    new_gallery = []
    for i, image in enumerate(_gallery):
        if i == gallery_index:
            if shared.opts.hires_button_gallery_insert:
                new_gallery.append(image)
            new_gallery.extend(processed.images)
        else:
            new_gallery.append(image)

    if shared.opts.hires_button_gallery_insert:
        geninfo["infotexts"].insert(gallery_index + 1, processed.info)
    else:
        geninfo["infotexts"][gallery_index] = processed.info

    return new_gallery, json.dumps(geninfo), plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")


def txt2img_function(id_task: str, request: gr.Request, *args):
    p = txt2img_create_processing(id_task, request, *args)

    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)

        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images + processed.extra_images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")


def txt2img_upscale(id_task: str, request: gr.Request, gallery, gallery_index, generation_info, *args):
    return main_thread.run_and_wait_result(txt2img_upscale_function, id_task, request, gallery, gallery_index, generation_info, *args)


def txt2img(id_task: str, request: gr.Request, *args):
    return main_thread.run_and_wait_result(txt2img_function, id_task, request, *args)
