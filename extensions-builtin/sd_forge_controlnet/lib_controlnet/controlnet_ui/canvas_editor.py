import gradio as gr
import numpy as np

from modules_forge.forge_canvas.canvas import ForgeCanvas


class CanvasEditor:

    def __init__(self) -> None:
        self.group = None
        self.canvas = None
        self.clear = None

    def render(self, elem_id_tabname, tabname):
        with gr.Group(elem_classes=["cnet-input-image-group"], visible=False) as self.group:
            self.canvas = ForgeCanvas(elem_id=f"{elem_id_tabname}_{tabname}_input_canvas", elem_classes=["cnet-image"], height=320, contrast_scribbles=False, scribble_color="#FF0000", scribble_color_fixed=False, scribble_alpha=255, scribble_alpha_fixed=True, scribble_softness_fixed=True, numpy=True, no_upload=True)
            self.clear = gr.Button("Create Empty Canvas", elem_id=f"{elem_id_tabname}_{tabname}_clear_canvas")

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

        def empty(w: int, h: int):
            return np.ones((h, w, 4), dtype=np.uint8) * 255, gr.update(value=None)

        self.clear.click(
            fn=empty,
            inputs=[width, height],
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
