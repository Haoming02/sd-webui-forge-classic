import gradio as gr

from modules import scripts
from backend import memory_management


class NeverOOMForForge(scripts.Script):
    sorting_priority = 18

    def __init__(self):
        self.previous_unet_enabled: bool = False
        self.original_vram_state: memory_management.VRAMState = memory_management.vram_state

    def title(self):
        return "Never OOM Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            unet_enabled = gr.Checkbox(False, label="Enabled for UNet (always offload)")
            vae_enabled = gr.Checkbox(False, label="Enabled for VAE (always tiled)")

        return [unet_enabled, vae_enabled]

    def process(self, p, unet_enabled: bool, vae_enabled: bool):

        if unet_enabled:
            memory_management.logger.info("[NeverOOM] Enabled for UNet (always offload)")
        if vae_enabled:
            memory_management.logger.info("[NeverOOM] Enabled for VAE (always tiled)")

        memory_management.VAE_ALWAYS_TILED = vae_enabled

        if self.previous_unet_enabled != unet_enabled:
            memory_management.unload_all_models()

            if unet_enabled:
                self.original_vram_state = memory_management.vram_state
                memory_management.vram_state = memory_management.VRAMState.NO_VRAM
            else:
                memory_management.vram_state = self.original_vram_state

            memory_management.logger.info(f"Changed VRAM State to {memory_management.vram_state.name}")
            self.previous_unet_enabled = unet_enabled
