import gradio as gr
from lib_multidiffusion.tiled_diffusion import TiledDiffusion

from modules import scripts
from modules.ui_components import InputAccordion


class MultiDiffusionForForge(scripts.Script):
    sorting_priority = 16

    def title(self):
        return "MultiDiffusion Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else None

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:
            method = gr.Radio(label="Method", choices=("MultiDiffusion", "Mixture of Diffusers"), value="Mixture of Diffusers")
            with gr.Row():
                tile_width = gr.Slider(label="Tile Width", minimum=256, maximum=2048, step=64, value=768)
                tile_height = gr.Slider(label="Tile Height", minimum=256, maximum=2048, step=64, value=768)
            with gr.Row():
                tile_overlap = gr.Slider(label="Tile Overlap", minimum=0, maximum=1024, step=16, value=64)
                tile_batch_size = gr.Slider(label="Tile Batch Size", minimum=1, maximum=8, step=1, value=1)

        return enabled, method, tile_width, tile_height, tile_overlap, tile_batch_size

    def process_before_every_sampling(self, p, enabled: bool, method: str, tile_width: int, tile_height: int, tile_overlap: int, tile_batch_size: int, **kwargs):
        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet
        unet = TiledDiffusion.apply(unet, method, tile_width, tile_height, tile_overlap, tile_batch_size)
        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(
            {
                "multidiffusion_enabled": enabled,
                "multidiffusion_method": method,
                "multidiffusion_tile_width": tile_width,
                "multidiffusion_tile_height": tile_height,
                "multidiffusion_tile_overlap": tile_overlap,
                "multidiffusion_tile_batch_size": tile_batch_size,
            }
        )
