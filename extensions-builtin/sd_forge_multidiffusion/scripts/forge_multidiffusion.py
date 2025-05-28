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
            with gr.Row():
                method = gr.Radio(label="Method", choices=("MultiDiffusion", "Mixture of Diffusers", "SpotDiffusion"), value="Mixture of Diffusers")
                shift_method = gr.Radio(label="Shift Method", choices=("random", "sorted", "fibonacci"), value="random", visible=False)
            with gr.Row():
                tile_width = gr.Slider(label="Tile Width", minimum=256, maximum=2048, step=64, value=768)
                tile_height = gr.Slider(label="Tile Height", minimum=256, maximum=2048, step=64, value=768)
            with gr.Row():
                tile_overlap = gr.Slider(label="Tile Overlap", minimum=0, maximum=1024, step=16, value=64)
                tile_batch_size = gr.Slider(label="Tile Batch Size", minimum=1, maximum=8, step=1, value=1)

            method.change(fn=lambda m: gr.update(visible=(m == "SpotDiffusion")), inputs=[method], outputs=[shift_method], show_progress="hidden", queue=False)

        return enabled, method, tile_width, tile_height, tile_overlap, tile_batch_size, shift_method

    def process_before_every_sampling(self, p, enabled: bool, method: str, tile_width: int, tile_height: int, tile_overlap: int, tile_batch_size: int, shift_method: str, **kwargs):
        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet

        unet.model_options["tiled_diffusion_shift_method"] = shift_method
        unet.model_options["tiled_diffusion_seed"] = getattr(p, "seed", 0)

        unet = TiledDiffusion.apply(unet, method, tile_width, tile_height, tile_overlap, tile_batch_size)

        p.sd_model.forge_objects.unet = unet
        params = {
            "multidiffusion_enabled": enabled,
            "multidiffusion_method": method,
            "multidiffusion_tile_width": tile_width,
            "multidiffusion_tile_height": tile_height,
            "multidiffusion_tile_overlap": tile_overlap,
            "multidiffusion_tile_batch_size": tile_batch_size,
            "multidiffusion_shift_method": shift_method,
        }

        if method == "SpotDiffusion":
            params["multidiffusion_shift_method"] = shift_method

        p.extra_generation_params = params
