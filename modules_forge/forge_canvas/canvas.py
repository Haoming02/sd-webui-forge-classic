# Forge Canvas
# AGPL V3
# by lllyasviel
# Commercial Use is not allowed. (Contact us for commercial use.)

import gradio.component_meta


create_or_modify_pyi_org = gradio.component_meta.create_or_modify_pyi


def create_or_modify_pyi_org_patched(component_class, class_name, events):
    try:
        if component_class.__name__ == 'LogicalImage':
            return
        return create_or_modify_pyi_org(component_class, class_name, events)
    except:
        return


gradio.component_meta.create_or_modify_pyi = create_or_modify_pyi_org_patched


import os
import uuid
import base64
import time
import gradio as gr
import numpy as np
import io
from collections import OrderedDict
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import UploadFile

from PIL import Image
from io import BytesIO
from gradio.context import Context
from functools import wraps

from modules.shared import opts

canvas_js_root_path = os.path.dirname(__file__)


def web_js(file_name):
    full_path = os.path.join(canvas_js_root_path, file_name)
    return f'<script src="file={full_path}?{os.path.getmtime(full_path)}"></script>\n'


def web_css(file_name):
    full_path = os.path.join(canvas_js_root_path, file_name)
    return f'<link rel="stylesheet" href="file={full_path}?{os.path.getmtime(full_path)}">\n'


DEBUG_MODE = False

canvas_html = open(os.path.join(canvas_js_root_path, 'canvas.html'), encoding='utf-8').read()
canvas_head = ''
canvas_head += web_css('canvas.css')
canvas_head += web_js('canvas.js')


def image_to_base64(image_array, numpy=True):
    image = Image.fromarray(image_array) if numpy else image_array
    image = image.convert("RGBA")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"


def base64_to_image(base64_str, numpy=True):
    if base64_str.startswith("data:image/png;base64,"):
        base64_str = base64_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    image = image.convert("RGBA")
    image_array = np.array(image) if numpy else image
    return image_array


# Temporary in-memory store for uploaded canvas images
# Key: id (hex string) -> (bytes, content_type, timestamp)
canvas_image_store: "OrderedDict[str, tuple[bytes, str, float]]" = OrderedDict()
CANVAS_STORE_LIMIT = 256


def setup_canvas_api(app):
    """Register API endpoints for canvas image upload and retrieval."""
    app.add_api_route("/internal/forge-canvas/upload", upload_canvas_image, methods=["POST"])
    app.add_api_route("/internal/forge-canvas/{id}", get_canvas_image, methods=["GET"])


async def upload_canvas_image(file: UploadFile):
    try:
        data = await file.read()
        img_id = uuid.uuid4().hex
        canvas_image_store[img_id] = (data, file.content_type or "image/png", time.time())

        # trim store
        while len(canvas_image_store) > CANVAS_STORE_LIMIT:
            canvas_image_store.popitem(last=False)

        return JSONResponse({"id": img_id, "url": f"./internal/forge-canvas/{img_id}"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def get_canvas_image(id: str):
    entry = canvas_image_store.get(id)
    if not entry:
        from fastapi.responses import Response
        return Response(status_code=404)
    data, content_type, _ = entry
    return StreamingResponse(io.BytesIO(data), media_type=content_type)


class LogicalImage(gr.Textbox):
    @wraps(gr.Textbox.__init__)
    def __init__(self, *args, numpy=True, **kwargs):
        self.numpy = numpy
        self.infotext = dict()

        if 'value' in kwargs:
            initial_value = kwargs['value']
            if initial_value is not None:
                kwargs['value'] = self.image_to_base64(initial_value)
            else:
                del kwargs['value']

        super().__init__(*args, **kwargs)

    def preprocess(self, payload):
        if not isinstance(payload, str):
            return None
        if payload.startswith("data:image/png;base64,"):
            image = base64_to_image(payload, numpy=self.numpy)
        elif payload.startswith("forge-canvas://"):
            # payload is a reference to a previously uploaded canvas image
            img_id = payload.split('://', 1)[1]
            entry = canvas_image_store.get(img_id)
            if not entry:
                return None
            data, content_type, _ = entry
            try:
                image = Image.open(BytesIO(data))
                image = image.convert('RGBA')
                image = np.array(image) if self.numpy else image
            except Exception:
                return None
        else:
            return None
        if hasattr(image, 'info'):
            image.info = self.infotext
        
        return image

    def postprocess(self, value):
        if value is None:
            return None
            
        if hasattr(value, 'info'):
            self.infotext = value.info

        return image_to_base64(value, numpy=self.numpy)

    def get_block_name(self):
        return "textbox"


class ForgeCanvas:
    def __init__(
            self,
            no_upload=False,
            no_scribbles=False,
            contrast_scribbles=False,
            height=512,
            scribble_color='#000000',
            scribble_color_fixed=False,
            scribble_width=4,
            scribble_width_fixed=False,
            scribble_alpha=100,
            scribble_alpha_fixed=False,
            scribble_softness=0,
            scribble_softness_fixed=False,
            visible=True,
            numpy=False,
            initial_image=None,
            elem_id=None,
            elem_classes=None
    ):
        self.uuid = 'uuid_' + uuid.uuid4().hex

        canvas_html_uuid = canvas_html.replace('forge_mixin', self.uuid)

        if opts.forge_canvas_plain:
            canvas_html_uuid = canvas_html_uuid.replace('class="forge-image-container"', 'class="forge-image-container-plain"').replace('stroke="white"', 'stroke=#444')
        if opts.forge_canvas_toolbar_always:
            canvas_html_uuid = canvas_html_uuid.replace('class="forge-toolbar"', 'class="forge-toolbar-static"')
            
        self.block = gr.HTML(canvas_html_uuid, visible=visible, elem_id=elem_id, elem_classes=elem_classes)
        self.foreground = LogicalImage(visible=DEBUG_MODE, label='foreground', numpy=numpy, elem_id=self.uuid, elem_classes=['logical_image_foreground'])
        self.background = LogicalImage(visible=DEBUG_MODE, label='background', numpy=numpy, value=initial_image, elem_id=self.uuid, elem_classes=['logical_image_background'])
        
        Context.root_block.load(None, js=f'async ()=>{{new ForgeCanvas("{self.uuid}", {no_upload}, {no_scribbles}, {contrast_scribbles}, {height}, '
                                         f"'{scribble_color}', {scribble_color_fixed}, {scribble_width}, {scribble_width_fixed}, "
                                         f'{scribble_alpha}, {scribble_alpha_fixed}, {scribble_softness}, {scribble_softness_fixed});}}')
