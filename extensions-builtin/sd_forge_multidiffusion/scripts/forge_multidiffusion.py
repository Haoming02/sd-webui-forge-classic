import gradio as gr
from lib_multidiffusion.tiled_diffusion import TiledDiffusion

from modules import scripts

opTiledDiffusion = TiledDiffusion().apply


class MultiDiffusionForForge(scripts.Script):
    sorting_priority = 16

    def title(self):
        return "MultiDiffusion Integrated"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label="Enabled", value=False)
            method = gr.Radio(label="Method", choices=["MultiDiffusion", "Mixture of Diffusers", "SpotDiffusion"], value="Mixture of Diffusers")
            with gr.Row():
                tile_width = gr.Slider(label="Tile Width", minimum=16, maximum=8192, step=16, value=768)
                tile_height = gr.Slider(label="Tile Height", minimum=16, maximum=8192, step=16, value=768)
            with gr.Row():
                tile_overlap = gr.Slider(label="Tile Overlap", minimum=0, maximum=2048, step=32, value=64)
                tile_batch_size = gr.Slider(label="Tile Batch Size", minimum=1, maximum=8192, step=1, value=4)
            with gr.Row():
                shift_method = gr.Radio(label="Shift Method (SpotDiffusion)", choices=["random", "sorted", "fibonacci"], value="random", visible=True)

        return enabled, method, tile_width, tile_height, tile_overlap, tile_batch_size, shift_method

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        enabled, method, tile_width, tile_height, tile_overlap, tile_batch_size, shift_method = script_args

        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet

        # Store additional options for SpotDiffusion
        unet.model_options["tiled_diffusion_shift_method"] = shift_method
        unet.model_options["tiled_diffusion_seed"] = getattr(p, "seed", 0)

        unet = opTiledDiffusion(unet, method, tile_width, tile_height, tile_overlap, tile_batch_size)[0]

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(multidiffusion_enabled=enabled, multidiffusion_method=method, multidiffusion_tile_width=tile_width, multidiffusion_tile_height=tile_height, multidiffusion_tile_overlap=tile_overlap, multidiffusion_tile_batch_size=tile_batch_size, multidiffusion_shift_method=shift_method if method == "SpotDiffusion" else None))

        return
