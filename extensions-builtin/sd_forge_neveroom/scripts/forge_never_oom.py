import gradio as gr

from modules import scripts, shared, script_callbacks
from backend import memory_management


class NeverOOMForForge(scripts.Script):
    sorting_priority = 18

    def __init__(self):
        self.previous_unet_enabled = False
        self.original_vram_state = memory_management.vram_state

    def title(self):
        return "Never OOM Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        return []

    def process(self, p, *script_args, **kwargs):
        unet_enabled = getattr(shared.opts, "forge_never_oom_unet", False)
        vae_enabled = getattr(shared.opts, "forge_never_oom_vae", False)

        if unet_enabled:
            print('NeverOOM Enabled for UNet (always maximize offload)')

        if vae_enabled:
            print('NeverOOM Enabled for VAE (always tiled)')

        memory_management.VAE_ALWAYS_TILED = vae_enabled

        if self.previous_unet_enabled != unet_enabled:
            memory_management.unload_all_models()
            if unet_enabled:
                self.original_vram_state = memory_management.vram_state
                memory_management.vram_state = memory_management.VRAMState.NO_VRAM
            else:
                memory_management.vram_state = self.original_vram_state
            print(f'VARM State Changed To {memory_management.vram_state.name}')
            self.previous_unet_enabled = unet_enabled

        return

def on_ui_settings():
    section = ('never_oom', "Never OOM")
    shared.opts.add_option(
        "forge_never_oom_unet",
        shared.OptionInfo(False, "Enabled for UNet (always maximize offload)", section=section)
    )
    shared.opts.add_option(
        "forge_never_oom_vae",
        shared.OptionInfo(False, "Enabled for VAE (always tiled)", section=section)
    )

script_callbacks.on_ui_settings(on_ui_settings)