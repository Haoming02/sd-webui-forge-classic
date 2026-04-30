import gradio as gr
import numpy as np
import torch
from PIL import Image

from backend.args import dynamic_args
from modules import images, scripts, sd_models
from modules.api import api
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, logger
from modules.sd_samplers_common import images_tensor_to_samples
from modules.shared import device, opts
from modules.ui_components import FormRow, InputAccordion

t2i_info = """
For <b>Flux.1-Kontext</b> / <b>Flux.2-Klein</b> / <b>Qwen-Image-Edit</b>: Use in <b>txt2img</b> to achieve the effect of empty latent with custom resolution<br>
For <b>Wan 2.2 I2V</b>: Use in <b>txt2img</b> to set as the Last Frame to achieve LastFrameToVideo<br>
<b>Note:</b> This doesn't actually stitch the images ; <b>Tip:</b> Use the "Image to Upload" to paste images
"""

i2i_info = """
For <b>Flux.1-Kontext</b> / <b>Flux.2-Klein</b> / <b>Qwen-Image-Edit</b>: Use in <b>img2img</b> to achieve the effect of multiple input images<br>
For <b>Wan 2.2 I2V</b>: Use in <b>img2img</b> to set as the Last Frame to achieve FirstLastFrameToVideo<br>
<b>Note:</b> This doesn't actually stitch the images ; <b>Tip:</b> Use the "Image to Upload" to paste images
"""


class ImageStitch(scripts.Script):
    sorting_priority = 529
    cached_parameters: list[int] = None

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
                label="Reference Image(s)",
                min_width=512,
                height=384,
                columns=3,
                rows=1,
                allow_preview=False,
                object_fit="contain",
                elem_id=self.elem_id("ref_latent"),
            )

            select_index = gr.State(-1)

            def on_select(evt: gr.SelectData) -> int:
                return evt.index

            references.select(
                fn=on_select,
                outputs=[select_index],
                queue=False,
                show_progress=False,
            )

            with FormRow():
                upload = gr.Image(
                    height=225,
                    width=225,
                    sources="upload",
                    type="pil",
                    label="Image to Upload",
                    show_download_button=False,
                    show_share_button=False,
                )
                with gr.Column():
                    btn_upload = gr.Button("Append Pasted Image")
                    btn_replace = gr.Button("Replace Selected Image")
                    btn_delete = gr.Button("Delete Selected Image", variant="stop")
                    btn_clear = gr.Button("Clear All References", variant="stop")

            max_dim = gr.Slider(
                minimum=0,
                maximum=2048,
                value=1024,
                step=256,
                label="Maximum Side Length",
                info="reduce VRAM usage during encoding ; apply to all reference images ; set to 0 for no limit",
            )

        def _upload(gallery: list[tuple[Image.Image, str]], image: Image.Image):
            if not image:
                return [gr.skip(), gr.skip()]
            elif not gallery:
                gallery = [(image, None)]
            else:
                gallery.append((image, None))
            return [gr.update(value=gallery), gr.update(value=None)]

        def _replace(index: int, gallery: list[tuple[Image.Image, str]], image: Image.Image):
            if not image or not gallery or index < 0 or index >= len(gallery):
                return [-1, gr.skip(), gr.skip()]
            gallery[index] = (image, None)
            return [-1, gr.update(value=gallery), gr.update(value=None)]

        def _delete(index: int, gallery: list[tuple[Image.Image, str]]):
            if not gallery or index < 0 or index >= len(gallery):
                return [-1, gr.skip()]
            gallery.pop(index)
            return [-1, gr.update(value=gallery)]

        btn_upload.click(
            fn=_upload,
            inputs=[references, upload],
            outputs=[references, upload],
            queue=False,
            show_progress=False,
        )

        btn_replace.click(
            fn=_replace,
            inputs=[select_index, references, upload],
            outputs=[select_index, references, upload],
            queue=False,
            show_progress=False,
        )

        btn_delete.click(
            fn=_delete,
            inputs=[select_index, references],
            outputs=[select_index, references],
            queue=False,
            show_progress=False,
        )

        btn_clear.click(
            fn=lambda: [-1, gr.update(value=[])],
            outputs=[select_index, references],
            queue=False,
            show_progress=False,
        )

        return [enable, references, max_dim]

    @staticmethod
    def reset_references(p: StableDiffusionProcessing):
        # re-encode conditioning
        p.clear_prompt_cache()
        p.sd_model.clear_references()

    def process(self, p: StableDiffusionProcessing, enable: bool, references: list[str | tuple[Image.Image, str]], max_dim: int):
        if not (enable and references and any(getattr(dynamic_args, key) for key in ("kontext", "edit", "klein", "wan"))):
            if ImageStitch.cached_parameters is None:
                return

            # if previously enabled, clear out the ref_latents
            ImageStitch.cached_parameters = None
            self.reset_references(p)
            return

        references = self.extract_images(references)

        # cache is based on reference inputs & model
        cache: list[str | int | bool] = [str(sd_models.model_data.forge_loading_parameters), *(self.hash_image(ref) for ref in references), (dynamic_args.wan and isinstance(p, StableDiffusionProcessingTxt2Img))]
        if ImageStitch.cached_parameters == cache:
            return

        ImageStitch.cached_parameters = cache
        self.reset_references(p)

        _batch_size: int = None

        if dynamic_args.wan:
            if isinstance(p, StableDiffusionProcessingTxt2Img):
                _batch_size = p.batch_size
                if _batch_size == 1:
                    logger.error("Wan 2.2 requires more than one frame...")
                    return
            if len(references) > 1:
                logger.warning("Wan 2.2 only uses the first reference image...")
                references = [references[0]]

        dynamic_args.is_referencing = True

        for reference in references:
            reference = self.preprocess(reference, max_dim)
            if _batch_size:
                reference = images.resize_image(1, reference, p.width, p.height)
            image = images.flatten(reference, opts.img2img_background_color)
            image = np.array(image, dtype=np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)
            image = torch.from_numpy(image).to(device=device).unsqueeze(0)

            if _batch_size:
                dim = [_batch_size - 1] + list(image.shape)[1:]
                empty = torch.empty(dim, dtype=torch.float32, device=device)
                image = torch.cat([image, empty], dim=0)

            images_tensor_to_samples(image, 0, p.sd_model)  # calls encode_first_stage

        dynamic_args.is_referencing = False

    @staticmethod
    def extract_images(gallery: list[str | tuple[Image.Image, str]]) -> list[Image.Image]:
        if isinstance(gallery[0], str):
            return [api.decode_base64_to_image(img) for img in gallery]
        return [img for (img, _) in gallery]

    @staticmethod
    def preprocess(img: Image.Image, limit: int) -> Image.Image:
        w, h = img.size

        if limit > 0 and max(w, h) > limit:
            ratio = limit / max(w, h)
            _w, _h = int(w * ratio), int(h * ratio)
        else:
            _w, _h = w, h

        if _w % 64 != 0 or _h % 64 != 0:
            _w = round(_w / 64) * 64
            _h = round(_h / 64) * 64

        if w != _w or h != _h:
            return images.resize_image(1, img, _w, _h)
        else:
            return img

    @staticmethod
    def hash_image(img: Image.Image) -> int:
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        img = img.convert("L")
        return hash(str(list(img.getdata())))
