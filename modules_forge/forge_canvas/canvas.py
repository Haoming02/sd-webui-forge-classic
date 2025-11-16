"""
Forge Canvas
Copyright (C) 2024 lllyasviel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
"""

import gradio.component_meta

create_or_modify_pyi_org = gradio.component_meta.create_or_modify_pyi


def create_or_modify_pyi_org_patched(component_class, class_name, events):
    try:
        if component_class.__name__ == "LogicalImage":
            return
        return create_or_modify_pyi_org(component_class, class_name, events)
    except Exception:
        return


gradio.component_meta.create_or_modify_pyi = create_or_modify_pyi_org_patched


import base64
import os
import uuid
from functools import wraps
from io import BytesIO

import gradio as gr
import numpy as np
from gradio.context import Context
from PIL import Image

from modules.shared import opts

DEBUG_MODE = False
canvas_js_root_path = os.path.dirname(__file__)


def web_js(file_name):
    full_path = os.path.join(canvas_js_root_path, file_name)
    return f'<script src="file={full_path}?{os.path.getmtime(full_path)}"></script>\n'


def web_css(file_name):
    full_path = os.path.join(canvas_js_root_path, file_name)
    return f'<link rel="stylesheet" href="file={full_path}?{os.path.getmtime(full_path)}">\n'


canvas_html = open(os.path.join(canvas_js_root_path, "canvas.html"), encoding="utf-8").read()
canvas_head = "".join((web_css("canvas.css"), web_js("canvas.js")))


def image_to_base64(image_array, numpy=True):
    image = Image.fromarray(image_array) if numpy else image_array
    image = image.convert("RGBA")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def base64_to_image(base64_str, numpy=True):
    if base64_str.startswith("data:image/png;base64,"):
        base64_str = base64_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    image = image.convert("RGBA")
    image_array = np.array(image) if numpy else image
    return image_array


class LogicalImage(gr.Textbox):
    @wraps(gr.Textbox.__init__)
    def __init__(self, *args, numpy=True, **kwargs):
        self.numpy = numpy
        self.infotext = dict()

        if "value" in kwargs:
            initial_value = kwargs["value"]
            if initial_value is not None:
                kwargs["value"] = self.image_to_base64(initial_value)
            else:
                del kwargs["value"]

        super().__init__(*args, **kwargs)

    def preprocess(self, payload):
        if not isinstance(payload, str):
            return None

        if not payload.startswith("data:image/png;base64,"):
            return None

        image = base64_to_image(payload, numpy=self.numpy)
        if hasattr(image, "info"):
            image.info = self.infotext

        return image

    def postprocess(self, value):
        if value is None:
            return None

        if hasattr(value, "info"):
            self.infotext = value.info

        return image_to_base64(value, numpy=self.numpy)

    def get_block_name(self):
        return "textbox"


class ForgeCanvas:
    def __init__(self, no_upload=False, no_scribbles=False, contrast_scribbles=False, height=None, scribble_color="#000000", scribble_color_fixed=False, scribble_width=25, scribble_width_fixed=False, scribble_alpha=100, scribble_alpha_fixed=False, scribble_softness=0, scribble_softness_fixed=False, visible=True, numpy=False, initial_image=None, elem_id=None, elem_classes=None):
        self.uuid = "uuid_" + uuid.uuid4().hex

        canvas_html_uuid = canvas_html.replace("forge_mixin", self.uuid)

        if opts.forge_canvas_plain:
            canvas_html_uuid = canvas_html_uuid.replace('class="forge-image-container"', f'class="forge-image-container plain" style="background-color: {opts.forge_canvas_plain_color}"').replace('stroke="white"', "stroke=#444")
        if opts.forge_canvas_toolbar_always:
            canvas_html_uuid = canvas_html_uuid.replace('class="forge-toolbar"', 'class="forge-toolbar-static"')

        self.block = gr.HTML(canvas_html_uuid, visible=visible, elem_id=elem_id, elem_classes=elem_classes)
        self.foreground = LogicalImage(visible=DEBUG_MODE, label="foreground", numpy=numpy, elem_id=self.uuid, elem_classes=["logical_image_foreground"])
        self.background = LogicalImage(visible=DEBUG_MODE, label="background", numpy=numpy, value=initial_image, elem_id=self.uuid, elem_classes=["logical_image_background"])
        Context.root_block.load(None, js=f'async ()=>{{new ForgeCanvas("{self.uuid}", {no_upload}, {no_scribbles}, {contrast_scribbles}, {height or opts.forge_canvas_height}, ' f"'{scribble_color}', {scribble_color_fixed}, {scribble_width}, {scribble_width_fixed}, {opts.forge_canvas_consistent_brush}, " f"{scribble_alpha}, {scribble_alpha_fixed}, {scribble_softness}, {scribble_softness_fixed});}}")
