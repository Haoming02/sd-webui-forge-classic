from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from modules.processing import StableDiffusionProcessing

import gradio as gr
import numpy as np
import torch

from backend.args import dynamic_args
from modules import images, scripts
from modules.sd_samplers_common import approximation_indexes, images_tensor_to_samples
from modules.shared import device, opts

t2i_info = """
For <b>Flux-Kontext</b> and <b>Qwen-Image-Edit</b><br>
Use in <b>txt2img</b> to achieve the effect of empty latent with custom resolution
"""

i2i_info = """
For <b>Flux-Kontext</b> and <b>Qwen-Image-Edit</b><br>
Use in <b>img2img</b> to achieve the effect of 2 input images<br>
<b>NOTE:</b> This doesn't actually stitch the images, so use "1st/2nd" instead of "left/right" in prompts
"""


class ImageStitch(scripts.Script):
    sorting_priority = 529

    def title(self):
        return "ImageStitch Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML(i2i_info if is_img2img else t2i_info)
            img = gr.Image(
                value=None,
                type="pil",
                image_mode="RGBA",
                sources="upload",
                interactive=True,
                show_label=False,
                show_download_button=False,
                show_share_button=False,
                label="Reference Latents",
                width=384,
                height=384,
                elem_id=self.elem_id("ref_latent"),
            )

        return [img]

    def process(self, p: "StableDiffusionProcessing", reference: "Image.Image"):
        if reference is None:
            return
        if not any(dynamic_args[key] for key in ("kontext", "edit")):
            return

        image = images.flatten(reference, opts.img2img_background_color)
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.moveaxis(image, 2, 0)
        image = torch.from_numpy(image).to(device=device, dtype=torch.float32)

        ref = images_tensor_to_samples(
            image.unsqueeze(0),
            approximation_indexes.get(opts.sd_vae_encode_method),
            p.sd_model,
        )

        if dynamic_args["kontext"]:
            dynamic_args["ref_latents"] = ref
