import gradio as gr

from backend import memory_management
from backend.patcher.unet import UnetPatcher
from modules import scripts


class NeverOOMForForge(scripts.Script):
    sorting_priority = 18

    def __init__(self):
        self.previous_unet_enabled: bool = False

    def title(self):
        return "Never OOM Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            unet_enabled = gr.Checkbox(False, label="Enabled for UNet (always offload)")
            vae_enabled = gr.Checkbox(False, label="Enabled for VAE (always tiled)")

        return [unet_enabled, vae_enabled]

    def setup(self, p, unet_enabled: bool, vae_enabled: bool):

        if unet_enabled:
            memory_management.logger.info("[NeverOOM] Enabled for UNet (always offload)")
        if vae_enabled:
            memory_management.logger.info("[NeverOOM] Enabled for VAE (always tiled)")

        memory_management.UNET_ALWAYS_OFFLOAD = unet_enabled
        memory_management.VAE_ALWAYS_TILED = vae_enabled

        if self.previous_unet_enabled != unet_enabled:
            idx = None

            for i, loaded_models in enumerate(memory_management.current_loaded_models):
                if isinstance(loaded_models.model, UnetPatcher):
                    idx = i
                    break

            if idx is not None:
                mdl: memory_management.LoadedModel = memory_management.current_loaded_models.pop(idx)
                mdl.model_unload()
                del mdl

            self.previous_unet_enabled = unet_enabled
