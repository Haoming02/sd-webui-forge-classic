import gradio as gr
import numpy as np
import torch
from PIL import Image

from backend.args import dynamic_args
from modules import images, scripts, sd_models
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img
from modules.sd_samplers_common import approximation_indexes, images_tensor_to_samples
from modules.shared import device, opts
from modules.ui_components import InputAccordion

t2i_info = """
For <b>Flux-Kontext</b> and <b>Qwen-Image-Edit</b><br>
Use in <b>txt2img</b> to achieve the effect of empty latent with custom resolution<br>
<b>NOTE:</b> This doesn't actually stitch the images
"""

i2i_info = """
For <b>Flux-Kontext</b> and <b>Qwen-Image-Edit</b><br>
Use in <b>img2img</b> to achieve the effect of multiple input images<br>
<b>NOTE:</b> This doesn't actually stitch the images
"""


class ImageStitch(scripts.Script):
    sorting_priority = 529

    def __init__(self):
        self.cached_parameters: list[str, int] = None

    def title(self):
        return "ImageStitch Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(value=False, label=self.title()) as enable:
            gr.HTML(i2i_info if is_img2img else t2i_info)
            references = gr.Gallery(
                value=None,
                type="pil",
                interactive=True,
                show_label=False,
                container=False,
                show_download_button=False,
                show_share_button=False,
                label="Reference Latents",
                min_width=384,
                height=384,
                columns=3,
                rows=1,
                allow_preview=False,
                object_fit="contain",
                elem_id=self.elem_id("ref_latent"),
            )

        return [enable, references]

    def process(self, p: StableDiffusionProcessing, enable: bool, references: list[tuple[Image.Image, str]]):
        if not (enable and references and any(dynamic_args[key] for key in ("kontext", "edit"))):
            if self.cached_parameters is not None:
                self.cached_parameters = None
                p.cached_c = [None, None]
                p.cached_uc = [None, None]
            return

        cache: list[str | int] = [str(sd_models.model_data.forge_loading_parameters)]
        if isinstance(p, StableDiffusionProcessingImg2Img):
            cache.append(p.init_img_hash)
        for reference, _ in references:
            cache.append(self.hash_image(reference))

        if self.cached_parameters == cache:
            return

        self.cached_parameters = cache
        p.cached_c = [None, None]
        p.cached_uc = [None, None]

        for reference, _ in references:
            image = images.flatten(reference, opts.img2img_background_color)
            image = np.array(image, dtype=np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)
            image = torch.from_numpy(image).to(device=device, dtype=torch.float32)

            images_tensor_to_samples(
                image.unsqueeze(0),
                approximation_indexes.get(opts.sd_vae_encode_method),
                p.sd_model,
            )

    @staticmethod
    def hash_image(img: Image.Image) -> int:
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        img = img.convert("L")
        return hash(str(list(img.getdata())))
