import random
import time

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from modules.ui_components import FormColumn
from modules_forge.forge_canvas.canvas import ForgeCanvas

COLORS = ("red", "orange", "yellow", "green", "blue", "violet", "purple")


class CanvasEditor:

    def __init__(self) -> None:
        self.group = None
        self.canvas = None
        self.clear = None

    def render(self, elem_id_tabname, tabname):
        with gr.Group(elem_classes=["cnet-input-image-group"], visible=False) as self.group:
            self.canvas = ForgeCanvas(elem_id=f"{elem_id_tabname}_{tabname}_input_canvas", elem_classes=["cnet-image"], height=320, contrast_scribbles=False, scribble_color="#FF0000", scribble_color_fixed=False, scribble_alpha=255, scribble_alpha_fixed=True, scribble_softness_fixed=True, numpy=True, no_upload=True)
            with FormColumn():
                self.clear = gr.Button("Generate Canvas", elem_id=f"{elem_id_tabname}_{tabname}_clear_canvas")
                with gr.Accordion("Advanced Parameters", open=False):
                    with gr.Row():
                        self.num = gr.Slider(value=0, label="Number of Subjects", minimum=0, maximum=7, step=1, info="0 for Blank Canvas")
                        self.height = gr.Slider(value=0.4, label="Vertical Position", minimum=0.2, maximum=0.8, step=0.05)
                    with gr.Row():
                        self.randomize = gr.Slider(value=0.075, label="Randomize Bias", minimum=0.0, maximum=0.5, step=0.025)
                        self.shape = gr.Radio(choices=["Rectangle", "Oval"], value="Rectangle", label="Mask Shape")

    def register_callbacks(
        self,
        image: ForgeCanvas,
        image_group: gr.Group,
        type_filter: gr.Radio,
        width: gr.Slider,
        height: gr.Slider,
    ):
        def update_ui(type_: str) -> tuple[dict, dict]:
            vis: bool = type_ == "Region"
            return [gr.update(visible=vis), gr.update(visible=(not vis))] + [gr.update(value=None) if vis else gr.skip()] * 2

        type_filter.change(
            fn=update_ui,
            inputs=[type_filter],
            outputs=[self.group, image_group, image.background, image.foreground],
            queue=False,
        )

        self.clear.click(
            fn=self._generate_canvas,
            inputs=[width, height, self.num, self.randomize, self.height, self.shape],
            outputs=[self.canvas.background, self.canvas.foreground],
        )

        def merge(bg: np.ndarray, fg: np.ndarray):
            if fg is None:  # first stroke
                return gr.skip()
            bg_rgb = bg[..., :3].astype(np.float32)
            fg_rgb = fg[..., :3].astype(np.float32)
            alpha = fg[..., 3:4].astype(np.float32) / 255.0
            merged = fg_rgb * alpha + bg_rgb * (1.0 - alpha)
            return np.clip(merged.round(), 0, 255).astype(np.uint8)

        self.canvas.foreground.change(
            fn=merge,
            inputs=[self.canvas.background, self.canvas.foreground],
            outputs=[image.background],
        )

    @staticmethod
    def _generate_canvas(w: int, h: int, n: int, r: float, v: float, s: str) -> tuple[np.ndarray, None]:
        if n == 0:
            return gr.update(value=np.ones((h, w, 3), dtype=np.uint8) * 255), gr.update(value=None)

        image = Image.new("RGB", (w, h), "white")

        rng = random.Random(time.time())

        draw = ImageDraw.Draw(image)
        _w = w / n

        for i in range(n):
            cx = (i + 0.5 + rng.uniform(-r, r)) * _w
            cy = ((1.0 - v) + rng.uniform(-r, r)) * h

            bw = _w * 0.85 * rng.uniform(1.0 - r, 1.0 + r)
            bh = h * 0.9 * rng.uniform(1.0 - r, 1.0 + r)

            x0, x1 = int(cx - bw / 2), int(cx + bw / 2)
            y0, y1 = int(cy - bh / 2), int(cy + bh / 2)

            if s == "Rectangle":
                draw.rectangle((x0, y0, x1, y1), fill=COLORS[i])
            else:
                draw.ellipse((x0, y0, x1, y1), fill=COLORS[i])

        return gr.update(value=np.asarray(image, dtype=np.uint8)), gr.update(value=None)
