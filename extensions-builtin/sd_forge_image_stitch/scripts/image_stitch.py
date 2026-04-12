import gradio as gr
import numpy as np
import torch
from PIL import Image

from backend.args import dynamic_args
from modules import images, scripts, sd_models
from modules.api import api
from modules.processing import StableDiffusionProcessing
from modules.sd_samplers_common import images_tensor_to_samples
from modules.shared import device, opts
from modules.ui_components import InputAccordion

t2i_info = """
For <b>Flux-Kontext</b> / <b>Flux.2-Klein</b> / <b>Qwen-Image-Edit</b><br>
Use in <b>txt2img</b> to achieve the effect of empty latent with custom resolution<br>
<b>NOTE:</b> This doesn't actually stitch the images
"""

i2i_info = """
For <b>Flux-Kontext</b> / <b>Flux.2-Klein</b> / <b>Qwen-Image-Edit</b><br>
Use in <b>img2img</b> to achieve the effect of multiple input images<br>
<b>NOTE:</b> This doesn't actually stitch the images
"""


class ImageStitch(scripts.Script):
    sorting_priority = 529

    def __init__(self):
        self.cached_parameters: list[int] = None

    def title(self):
        return "ImageStitch Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(value=False, label=self.title()) as enable:
            gr.HTML(i2i_info if is_img2img else t2i_info)
            css = """
            <style>
            #image_stitch_grid {
                display: flex !important;
                flex-wrap: wrap !important;
                flex-direction: row !important;
                gap: 12px !important;
                margin-top: 15px;
            }
            /* Lock the parent boundary aggressively to 120x120px */
            .img-stitch-box {
                position: relative !important;
                flex: 0 0 120px !important;
                width: 120px !important;
                height: 120px !important;
                min-width: 120px !important;
                min-height: 120px !important;
                max-width: 120px !important;
                max-height: 120px !important;
                order: 2;
                border-radius: 8px !important;
                overflow: hidden !important;
                padding: 0 !important;
                margin: 0 !important;
                border: none !important;
                background: transparent !important;
            }
            .img-stitch-box:not(:has(img)) { display: none !important; }
            .img-stitch-box:has(img) { display: block !important; order: 1; }
            
            /* Destroy all intermediate relative anchors from Svelte */
            .img-stitch-box * {
                position: static !important;
                border: none !important;
                box-shadow: none !important;
                transform: none !important;
            }
            
            /* Tarp the entire 120px area with the image */
            .img-stitch-box img {
                position: absolute !important;
                top: 0 !important;
                left: 0 !important;
                width: 100% !important;
                height: 100% !important;
                object-fit: cover !important;
                z-index: 10 !important;
                visibility: visible !important;
                background: transparent !important;
            }

            /* Identify the true X button and anchor it precisely to the 120px boundary */
            .img-stitch-box button.clear-button,
            .img-stitch-box button[aria-label*="Remove"],
            .img-stitch-box button[aria-label*="Clear"],
            .img-stitch-box button[aria-label*="Eliminar"],
            .img-stitch-box button[aria-label*="Quitar"],
            .img-stitch-box button[title*="Remove"],
            .img-stitch-box button[title*="Clear"],
            .img-stitch-box button[title*="Eliminar"],
            .img-stitch-box button[title*="Quitar"] {
                position: absolute !important;
                top: 4px !important;
                right: 4px !important;
                z-index: 50 !important;
                display: flex !important;
                width: 24px !important;
                height: 24px !important;
                background: rgba(10,10,10,0.7) !important;
                color: white !important;
                border-radius: 6px !important;
                padding: 2px !important;
                align-items: center !important;
                justify-content: center !important;
                border: 1px solid rgba(255,255,255,0.2) !important;
                opacity: 1 !important;
                visibility: visible !important;
                pointer-events: auto !important;
            }
            
            /* Destroy any button whose parent div DOES NOT have the image (The Bottom Toolbar) */
            .img-stitch-box div:not(:has(img)) > button {
                display: none !important;
                opacity: 0 !important;
                z-index: -100 !important;
                pointer-events: none !important;
            }
            
            /* Hide Clear All button unless there's at least one image loaded */
            #img_stitch_clear_btn {
                display: none !important;
            }
            #image_stitch_container:has(img) #img_stitch_clear_btn {
                display: block !important;
            }
            </style>
            """
            gr.HTML(css)
            
            html_dropzone = """
            <div id="custom_stitch_dropzone" style="border: 2px dashed #777; border-radius: 8px; padding: 15px; text-align: center; cursor: pointer; transition: 0.2s; background: rgba(0,0,0,0.2);">
                <svg style="width: 24px; margin: 0 auto 5px; fill: #aaa;" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>
                <div style="font-size: 14px; color: #ccc;">Click here, drag or paste images to upload</div>
                <input type="file" id="custom_stitch_file_input" multiple style="display: none;" accept="image/*">
            </div>
            """
            
            with gr.Column(elem_id="image_stitch_container"):
                gr.HTML(html_dropzone)
                clear_btn = gr.Button("🗑️ Clear All", elem_id="img_stitch_clear_btn", size="sm", variant="secondary")
                
                with gr.Column(elem_id="image_stitch_grid"):
                    refs = []
                    for i in range(50):
                        refs.append(gr.Image(type="pil", show_label=False, show_download_button=False, show_share_button=False, elem_id=f"img_stitch_ref{i+1}", elem_classes="img-stitch-box"))

            clear_btn.click(
                fn=lambda: [None] * 50,
                inputs=[],
                outputs=refs
            )

        return [enable] + refs

    @staticmethod
    def reset_references(p: StableDiffusionProcessing):
        # re-encode conditioning
        p.clear_prompt_cache()
        p.sd_model.clear_references()

    def process(self, p: StableDiffusionProcessing, enable: bool, *refs):
        raw_references = [r for r in refs[:50] if r is not None]

        if not (enable and raw_references and any(getattr(dynamic_args, key) for key in ("kontext", "edit", "klein"))):
            if self.cached_parameters is None:
                return

            # if previously enabled, clear out the ref_latents
            self.cached_parameters = None
            self.reset_references(p)
            return

        references = self.extract_images(raw_references)

        # cache is based on reference inputs & model
        cache: list[str | int] = [str(sd_models.model_data.forge_loading_parameters), *(self.hash_image(ref) for ref in references)]
        if self.cached_parameters == cache:
            return

        self.cached_parameters = cache
        self.reset_references(p)

        dynamic_args.is_referencing = True

        for reference in references:
            reference = self.preprocess(reference)
            image = images.flatten(reference, opts.img2img_background_color)
            image = np.array(image, dtype=np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)
            image = torch.from_numpy(image).to(device=device, dtype=torch.float32)

            images_tensor_to_samples(image.unsqueeze(0), 0, p.sd_model)  # calls encode_first_stage

        dynamic_args.is_referencing = False

    @staticmethod
    def extract_images(gallery) -> list[Image.Image]:
        import os
        res = []
        for x in gallery:
            if isinstance(x, Image.Image):
                res.append(x)
            elif isinstance(x, str):
                if os.path.isfile(x):
                    res.append(Image.open(x))
                else:
                    res.append(api.decode_base64_to_image(x))
            elif isinstance(x, tuple):
                res.append(x[0])
            elif hasattr(x, 'name') and os.path.isfile(x.name):
                res.append(Image.open(x.name))
        return res

    @staticmethod
    def preprocess(img: Image.Image) -> Image.Image:
        w, h = img.size
        if w % 64 == 0 and h % 64 == 0:
            return img

        return images.resize_image(1, img, round(w / 64) * 64, round(h / 64) * 64)

    @staticmethod
    def hash_image(img: Image.Image) -> int:
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        img = img.convert("L")
        return hash(str(list(img.getdata())))
