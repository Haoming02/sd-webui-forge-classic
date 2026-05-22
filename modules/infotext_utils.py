from __future__ import annotations

import json
import os
import re
from ast import literal_eval
from functools import partial
from typing import Any

import gradio as gr
from PIL import Image

from backend.text_processing.emphasis import uses_emphasis
from modules import errors, images, processing, script_callbacks, shared, ui_tempdir
from modules.paths import data_path
from modules_forge import main_entry

re_param_code = r'\s*([\w\s\-\/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
re_imagesize = re.compile(r"^(\d+)x(\d+)$")
re_cfg = re.compile(r"CFG scale:\s*([\d\.]+)")
type_of_gr_update = type(gr.skip())


class ParamBinding:
    def __init__(self, paste_button: gr.Button, tabname: str, source_text_component: gr.Textbox = None, source_image_component: gr.Gallery | gr.Image = None, source_tabname: str = None, override_settings_component: gr.Dropdown = None, paste_field_names: list[str] = None):
        self.paste_button = paste_button
        self.tabname = tabname
        self.source_text_component = source_text_component
        self.source_image_component = source_image_component
        self.source_tabname = source_tabname
        self.override_settings_component = override_settings_component
        self.paste_field_names = paste_field_names or []


class PasteField(tuple):
    def __new__(cls, component, target, *, api=None):
        return super().__new__(cls, (component, target))

    def __init__(self, component, target, *, api=None):
        self.component: gr.components.Component = component
        self.label = target if isinstance(target, str) else None
        self.function = target if callable(target) else None
        self.api = api


paste_fields: dict[str, dict] = {}
registered_param_bindings: list[ParamBinding] = []


def reset():
    paste_fields.clear()
    registered_param_bindings.clear()


def quote(text: str) -> str:
    if "," not in str(text) and "\n" not in str(text) and ":" not in str(text):
        return text

    try:
        return json.dumps(text, ensure_ascii=False)
    except Exception:
        return text


def unquote(text: str) -> str:
    if not text or not (text.startswith('"') and text.endswith('"')):
        return text

    try:
        return json.loads(text)
    except Exception:
        return text


def _parse_info(output: gr.components.Component, key: str, params: dict[str, Any]) -> gr.update:
    if not callable(key):
        v = params.get(key, None)
    else:
        try:
            v = key(params)
        except Exception:
            errors.report(f'Error executing "{key}"', exc_info=True)
            v = None

    if v is None:
        return gr.skip()
    elif isinstance(v, type_of_gr_update):
        return v
    else:
        try:
            valtype = type(output.value)

            if valtype == bool and v == "False":
                val = False
            elif valtype == int:
                val = float(v)
            else:
                val = valtype(v)

            return gr.update(value=val)
        except Exception:
            return gr.skip()


def image_from_url_text(filedata) -> Image.Image:
    if filedata is None:
        return None

    if isinstance(filedata, list):
        if len(filedata) == 0:
            return None
        filedata = filedata[0]

    filename: os.PathLike = None

    if isinstance(filedata, dict) and filedata.get("is_file", False):
        filename = filedata["name"]

    elif isinstance(filedata, tuple) and len(filedata) == 2:  # Gradio 4 sends images from Gallery as a list of tuples
        return filedata[0]

    if filename:
        is_in_right_dir = ui_tempdir.check_tmp_file(shared.demo, filename)
        assert is_in_right_dir, "trying to open image file outside of allowed directories"
        filename = filename.rsplit("?", 1)[0]
        return images.read(filename)

    if isinstance(filedata, str):
        from modules.api.api import decode_base64_to_image

        return decode_base64_to_image(filedata)

    return None


def add_paste_fields(tabname: str, init_img: gr.Image, fields: list[gr.components.Component], override_settings_component: gr.Dropdown = None):

    if fields:
        for i in range(len(fields)):
            if not isinstance(fields[i], PasteField):
                fields[i] = PasteField(*fields[i])

    paste_fields[tabname] = {"init_img": init_img, "fields": fields, "override_settings_component": override_settings_component}

    # backwards compatibility for existing extensions
    import modules.ui

    if tabname == "txt2img":
        modules.ui.txt2img_paste_fields = fields
    elif tabname == "img2img":
        modules.ui.img2img_paste_fields = fields


def create_buttons(tabs_list: list[str]) -> dict[str, gr.Button]:
    return {tab: gr.Button(f"Send to {tab}", elem_id=f"{tab}_tab") for tab in tabs_list}


def register_paste_params_button(binding: ParamBinding):
    registered_param_bindings.append(binding)


def _connect_paste_params_buttons(binding: ParamBinding):
    fields: list[PasteField] = paste_fields[binding.tabname]["fields"]
    dest_image: gr.Image = paste_fields[binding.tabname]["init_img"]
    override_settings: gr.Dropdown = binding.override_settings_component or paste_fields[binding.tabname]["override_settings_component"]

    dest_width: gr.Slider = next(iter([field for field, name in fields if name == "Size-1"] if fields else []), None)
    dest_height: gr.Slider = next(iter([field for field, name in fields if name == "Size-2"] if fields else []), None)

    if binding.source_image_component and dest_image:
        need_dimensions: bool = binding.tabname != "inpaint" and (dest_width and dest_height)

        if isinstance(binding.source_image_component, gr.Gallery):
            func = send_image_and_dimensions if need_dimensions else image_from_url_text
            jsfunc = "extract_image_from_gallery"
        else:
            func = send_image_and_dimensions if need_dimensions else lambda x: x
            jsfunc = None

        binding.paste_button.click(
            fn=func,
            inputs=[binding.source_image_component],
            outputs=[dest_image, dest_width, dest_height] if need_dimensions else [dest_image],
            show_progress=False,
            js=jsfunc,
        )

    if binding.source_text_component is not None and fields is not None:
        connect_paste(binding.paste_button, fields, binding.source_text_component, override_settings, binding.tabname)

    if binding.source_tabname is not None and fields is not None:
        paste_field_names = [
            *["Prompt", "Negative prompt", "Steps", "Face restoration"],
            *(["Seed"] if shared.opts.send_seed else []),
            *(["CFG scale"] if shared.opts.send_cfg else []),
            *binding.paste_field_names,
        ]

        if isinstance(binding.source_image_component, gr.Gallery) and shared.opts.send_image_info_not_ui:

            def read_infotext(x: Any, paste_fields: list[tuple]) -> list[gr.update]:
                image: Image.Image = x if isinstance(x, Image.Image) else image_from_url_text(x)
                if image is None:
                    return [gr.skip() for _ in paste_fields]

                info, _ = images.read_info_from_image(image)
                if not info:
                    return [gr.skip() for _ in paste_fields]

                params = parse_generation_parameters(info)
                script_callbacks.infotext_pasted_callback(info, params)

                res = []
                for output, key in paste_fields:
                    res.append(_parse_info(output, key, params))

                return res

            binding.paste_button.click(
                fn=partial(read_infotext, paste_fields=[(field, name) for field, name in fields if name in paste_field_names]),
                inputs=[binding.source_image_component],
                outputs=[field for field, name in fields if name in paste_field_names],
                js="extract_image_from_gallery",
                show_progress=False,
            ).then(fn=None, _js=f"switch_to_{binding.tabname}")

        else:
            binding.paste_button.click(
                fn=lambda *x: x,
                inputs=[field for field, name in paste_fields[binding.source_tabname]["fields"] if name in paste_field_names],
                outputs=[field for field, name in fields if name in paste_field_names],
                show_progress=False,
            ).then(fn=None, _js=f"switch_to_{binding.tabname}")

    else:
        binding.paste_button.click(fn=None, _js=f"switch_to_{binding.tabname}")


def connect_paste_params_buttons():
    for binding in registered_param_bindings:
        _connect_paste_params_buttons(binding)


def send_image_and_dimensions(x) -> tuple[Image.Image, int, int]:
    if isinstance(x, Image.Image):
        img = x
    else:
        img = image_from_url_text(x)

    if img is None:
        return None, gr.skip(), gr.skip()

    if img.mode != "RGB":
        img = img.convert("RGB")

    if shared.opts.send_size and isinstance(img, Image.Image):
        w = round(img.width / 64.0) * 64
        h = round(img.height / 64.0) * 64
    else:
        w = gr.skip()
        h = gr.skip()

    return img, w, h


def restore_old_hires_fix_params(res: dict):
    if shared.opts.use_old_hires_fix_width_height:
        hires_width = int(res.get("Hires resize-1", 0))
        hires_height = int(res.get("Hires resize-2", 0))

        if hires_width and hires_height:
            res["Size-1"] = hires_width
            res["Size-2"] = hires_height
            return

    try:
        firstpass_width = int(res.get("First pass size-1", None))
        firstpass_height = int(res.get("First pass size-2", None))
    except TypeError:
        return

    width = int(res.get("Size-1", 512))
    height = int(res.get("Size-2", 512))

    if firstpass_width == 0 or firstpass_height == 0:
        firstpass_width, firstpass_height = processing.old_hires_fix_first_pass_dimensions(width, height)

    res["Size-1"] = firstpass_width
    res["Size-2"] = firstpass_height
    res["Hires resize-1"] = width
    res["Hires resize-2"] = height


def _extract_styles(res: dict, prompt: str, negative_prompt: str) -> tuple[str, str]:
    if shared.opts.infotext_styles == "Ignore":
        return prompt, negative_prompt

    found_styles, prompt_no_styles, negative_prompt_no_styles = shared.prompt_styles.extract_styles_from_prompt(prompt, negative_prompt)

    same_hr_styles = True
    if "Hires prompt" in res or "Hires negative prompt" in res:
        hr_prompt, hr_negative_prompt = res.get("Hires prompt", prompt), res.get("Hires negative prompt", negative_prompt)
        hr_found_styles, hr_prompt_no_styles, hr_negative_prompt_no_styles = shared.prompt_styles.extract_styles_from_prompt(hr_prompt, hr_negative_prompt)
        if same_hr_styles := (found_styles == hr_found_styles):
            res["Hires prompt"] = "" if hr_prompt_no_styles == prompt_no_styles else hr_prompt_no_styles
            res["Hires negative prompt"] = "" if hr_negative_prompt_no_styles == negative_prompt_no_styles else hr_negative_prompt_no_styles

    if same_hr_styles:
        prompt, negative_prompt = prompt_no_styles, negative_prompt_no_styles
        if (shared.opts.infotext_styles == "Apply if any" and found_styles) or shared.opts.infotext_styles == "Apply":
            res["Styles array"] = found_styles

    return prompt, negative_prompt


def _populate_defaults(res: dict):
    if "Sampler" not in res:
        res["Sampler"] = "Euler"

    if "Schedule type" not in res:
        res["Schedule type"] = "Simple"

    if "RNG" not in res:
        res["RNG"] = "CPU"

    if "Hires resize-1" not in res:
        res["Hires resize-1"] = 0
        res["Hires resize-2"] = 0

    restore_old_hires_fix_params(res)

    if "Hires sampler" not in res:
        res["Hires sampler"] = "Use same sampler"

    if "Hires schedule type" not in res:
        res["Hires schedule type"] = "Use same scheduler"

    if "Hires checkpoint" not in res:
        res["Hires checkpoint"] = "Use same checkpoint"

    if "Hires prompt" not in res:
        res["Hires prompt"] = ""

    if "Hires negative prompt" not in res:
        res["Hires negative prompt"] = ""

    if "MaHiRo" not in res:
        res["MaHiRo"] = False

    if "Rescale CFG" not in res:
        res["Rescale CFG"] = 0.0


def parse_generation_parameters(x: str, skip_fields: list[str] | None = None):
    """
    parses infotext (the string under the Gallery in UI)
    returns a dict with field values
    """
    if skip_fields is None:
        skip_fields = shared.opts.infotext_skip_pasting

    *lines, lastline = x.strip().split("\n")
    if len(re_param.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ""

    _prompts: list[str] = []
    _negative_prompts: list[str] = []
    _neg: bool = False

    for line in lines:
        line = line.strip()
        if line.startswith("Negative prompt:"):
            line = line.replace("Negative prompt:", "").strip()
            _neg = True
        (_negative_prompts if _neg else _prompts).append(line)

    prompt: str = "\n".join(_prompts)
    negative_prompt: str = "\n".join(_negative_prompts)

    # region CivitAI
    if "flux" in lastline.lower():
        m = re.search(re_cfg, lastline)
        if m and float(m.group(1)) > 1.0:
            lastline = lastline.replace("CFG scale: ", "CFG scale: 1.0, Distilled CFG Scale: ")

    lastline = lastline.replace("Sampler: Undefined,", "Sampler: Euler, Schedule type: Simple,")
    lastline = lastline.replace(", width:", ", Size-1:").replace(", height:", ", Size-2:")
    # endregion

    res: dict[str, Any] = {}

    for k, v in re_param.findall(lastline):
        if k == "Noise Schedule":
            continue
        try:
            v = unquote(v)
            if (m := re_imagesize.match(v)) is not None:
                res[f"{k}-1"] = m.group(1)
                res[f"{k}-2"] = m.group(2)
            else:
                res[k] = v
        except Exception:
            print(f'Error parsing "{k}: {v}"')

    res["Prompt"], res["Negative prompt"] = prompt, negative_prompt = _extract_styles(res, prompt, negative_prompt)

    _populate_defaults(res)

    prompt_uses_emphasis: bool = uses_emphasis(prompt) or uses_emphasis(negative_prompt)
    if prompt_uses_emphasis and "Emphasis" not in res:
        res["Emphasis"] = "Original"

    if "Shift" in res:
        res["Distilled CFG Scale"] = res.pop("Shift")

    if "Hires Shift" in res:
        res["Hires Distilled CFG Scale"] = res.pop("Hires Shift")

    if "sd_model_name" in res:
        res["Model"] = res.pop("sd_model_name")

    if res.get("Model", None) == os.path.splitext(shared.opts.sd_model_checkpoint)[0]:
        res.pop("Model", None)

    for key in [*skip_fields, "Clip skip", "CLIP_stop_at_last_layers"]:
        res.pop(key, None)

    # VAE / TE
    modules, hr_modules = [], []

    if (vae := res.pop("VAE", None)) is not None:
        modules.append(vae)  # Classic

    _keys = list(res.keys())
    known_modules = {os.path.splitext(m)[0]: m for m in main_entry.module_list.keys()}

    for key in _keys:
        if key.startswith("Module "):
            if (m := known_modules.get(res.pop(key), None)) is not None:
                modules.append(m)
        elif key.startswith("Hires Module "):
            if (m := known_modules.get(res.pop(key), None)) is not None:
                hr_modules.append(m)

    if modules != []:
        current_modules = shared.opts.forge_additional_modules
        basename_modules = []
        for m in current_modules:
            basename_modules.append(os.path.basename(m))

        if sorted(modules) != sorted(basename_modules):
            res["VAE/TE"] = modules

    # processing.py/StableDiffusionProcessingTxt2Img/init()
    if "Hires Module 1" in res:
        if res["Hires Module 1"] == "Use same choices":
            hr_modules = ["Use same choices"]
        elif res["Hires Module 1"] == "Built-in":
            hr_modules = []

        res["Hires VAE/TE"] = hr_modules
    else:
        res["Hires VAE/TE"] = ["Use same choices"]

    return res


INFOTEXT_TO_SETTING = [("VAE/TE", "forge_additional_modules")]


def create_override_settings_dict(text_pairs: list[str]) -> dict[str, Any]:
    """
    creates processing's override_settings parameters from gradio's multiselect
    >>> ["VAE/TE: []"]
    {"forge_additional_modules": []}
    """

    if not text_pairs:
        return {}

    params = {}

    for pair in text_pairs:
        k, v = pair.split(":", 1)
        params[k.strip()] = v.strip()

    res: dict[str, Any] = {}
    mapping = [(info.infotext, k) for k, info in shared.opts.data_labels.items() if info.infotext]

    for param_name, setting_name in mapping + INFOTEXT_TO_SETTING:
        if (value := params.get(param_name, None)) is None:
            continue

        if setting_name == "forge_additional_modules":
            res[setting_name] = literal_eval(value)
            continue

        res[setting_name] = shared.opts.cast_value(setting_name, value)

    return res


def get_override_settings(params: dict[str, Any], *, skip_fields: list[str] = None) -> list[tuple[str, str, Any]]:
    """
    Returns a list of settings overrides from the infotext parameters dictionary

    >>> {"Clip skip": "2"}
    [("Clip skip", "CLIP_stop_at_last_layers", 2)]
    """

    res: list[tuple[str, str, Any]] = []
    mapping = [(info.infotext, k) for k, info in shared.opts.data_labels.items() if info.infotext]

    for param_name, setting_name in mapping + INFOTEXT_TO_SETTING:
        if param_name in (skip_fields or []):
            continue

        if (v := params.get(param_name, None)) is None:
            continue

        if setting_name == "sd_model_checkpoint" and shared.opts.disable_weights_auto_swap:
            continue
        if setting_name == "forge_additional_modules" and shared.opts.disable_modules_auto_swap:
            continue

        v = shared.opts.cast_value(setting_name, v)
        current_value = getattr(shared.opts, setting_name, None)

        if v != current_value:
            res.append((param_name, setting_name, v))

    return res


def connect_paste(button: gr.Button, paste_fields: list[PasteField], input_comp: gr.Textbox, override_settings_component: gr.Dropdown, tabname: str):
    def paste_func(prompt: str) -> list[gr.update]:
        if not prompt and not (shared.cmd_opts.hide_ui_dir_config or shared.cmd_opts.no_prompt_history):
            try:
                filename = os.path.join(data_path, "params.txt")
                with open(filename, "r", encoding="utf8") as file:
                    prompt: str = file.read()
            except OSError:
                pass

        params = parse_generation_parameters(prompt)
        script_callbacks.infotext_pasted_callback(prompt, params)

        res: list[gr.update] = []

        for output, key in paste_fields:
            res.append(_parse_info(output, key, params))

        return res

    if override_settings_component is not None:
        _handled_fields = [key for _, key in paste_fields]

        def paste_settings(params):
            vals = get_override_settings(params, skip_fields=_handled_fields)
            vals_pairs = [f"{infotext_text}: {value}" for infotext_text, _, value in vals]
            return gr.update(value=vals_pairs, choices=vals_pairs, visible=bool(vals_pairs))

        paste_fields = paste_fields + [(override_settings_component, paste_settings)]

    button.click(
        fn=paste_func,
        inputs=[input_comp],
        outputs=[x[0] for x in paste_fields],
        show_progress=False,
    ).then(fn=None, js=f"recalculate_prompts_{tabname}")
