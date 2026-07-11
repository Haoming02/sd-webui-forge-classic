# https://github.com/chengzeyi/ParaAttention (First Block Cache)

import logging

import gradio as gr

from backend.logging import setup_logger
from modules import scripts
from modules.ui_components import InputAccordion

logger = logging.getLogger("fbcache")
setup_logger(logger)

SUPPORTED_MODELS = ("Anima",)


class FirstBlockCacheForForge(scripts.ScriptBuiltinUI):
    sorting_priority = 18138

    def title(self):
        return "First Block Cache Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enable:
            gr.Markdown("Skips the DiT blocks on steps where the first block's output barely changed, reusing the previous result ; **Supported Models:** Anima")
            with gr.Row():
                threshold = gr.Slider(minimum=0.0, maximum=0.5, value=0.1, step=0.005, label="Residual Diff Threshold", info="Higher skips more steps = Faster ; Lower is more Accurate")
                max_consecutive = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Max Consecutive Cached Steps", info="Force a full computation after this many skipped steps in a row")
            with gr.Row():
                start_percent = gr.Slider(minimum=0, maximum=90, value=20, step=1, label="Warmup (% of Steps)", info="Never skip during the first steps ; the final step is always computed")

        for comp in (comps := (enable, threshold, max_consecutive, start_percent)):
            comp.do_not_save_to_config = True

        return comps

    def process_before_every_sampling(self, p, enable: bool, threshold: float, max_consecutive: float, start_percent: float, **kwargs):
        unet = p.sd_model.forge_objects.unet
        diffusion_model = unet.get_model_object("diffusion_model")
        supported = diffusion_model.__class__.__name__ in SUPPORTED_MODELS and hasattr(diffusion_model, "fbc_reset")

        if not enable:
            if supported:
                diffusion_model.fbc_reset()
            return

        if not supported:
            logger.warning(f'First Block Cache does not support "{diffusion_model.__class__.__name__}" ; skipping...')
            return

        unet = unet.clone()
        unet.model_options["transformer_options"]["fbcache"] = {
            "threshold": float(threshold),
            "max_consecutive": int(max_consecutive),
            "start_percent": float(start_percent) / 100.0,
        }

        diffusion_model.fbc_reset()
        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params["fbcache"] = f"{threshold}/{int(max_consecutive)}/{int(start_percent)}"

    def postprocess(self, p, processed, enable: bool = False, *args):
        if not enable:
            return
        try:
            diffusion_model = p.sd_model.forge_objects.unet.get_model_object("diffusion_model")
            if hasattr(diffusion_model, "fbc_reset"):
                diffusion_model.fbc_reset()
        except Exception:
            pass
