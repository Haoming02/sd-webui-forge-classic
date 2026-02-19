from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.modules.k_model import KModel

import gradio as gr
import torch

from backend.utils import get_attr, set_attr_raw
from modules import scripts


class TorchCompileForForge(scripts.Script):
    sorting_priority = 67

    def __init__(self):
        torch._dynamo.config.cache_size_limit = 256

    def title(self):
        return "Torch Compile Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            with gr.Row():
                behavior = gr.Radio(
                    label="Behavior",
                    value="auto",
                    choices=["enable", "disable", "auto"],
                    info='"auto" maintains the current status',
                    scale=2,
                )
                gr.Label(
                    value="Only enable this if you know what you are doing",
                    show_label=False,
                    scale=5,
                )
            with gr.Row():
                backend = gr.Radio(
                    label="Backend",
                    value="inductor",
                    choices=["inductor", "cudagraphs"],
                    scale=2,
                )
                mode = gr.Radio(
                    label="Mode",
                    value="default",
                    choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                    scale=5,
                )

        backend.change(lambda v: gr.update(visible=(v == "inductor")), inputs=[backend], outputs=[mode], queue=False, show_progress=False)
        return [behavior, backend, mode]

    @staticmethod
    def restore(kmodel: "KModel"):
        model = get_attr(kmodel, "_model_backup")
        set_attr_raw(kmodel, "diffusion_model", model)
        del kmodel._compile_config
        del kmodel._model_backup

    def process_batch(self, p, behavior: str, backend: str, mode: str, **kwargs):
        kmodel: "KModel" = p.sd_model.forge_objects.unet.model
        enable: bool = hasattr(kmodel, "_compile_config") if behavior == "auto" else (behavior == "enable")

        if not enable:
            if hasattr(kmodel, "_compile_config"):
                self.restore(kmodel)
            return

        if backend == "inductor":
            config = dict(backend=backend, mode=mode, dynamic=True, fullgraph=False)
        else:
            config = dict(backend=backend, dynamic=True, fullgraph=True)

        _cache: str = str(config) + p.sd_model.current_lora_hash

        if (_config := getattr(kmodel, "_compile_config", None)) is not None:
            if _config != _cache:
                self.restore(kmodel)
            else:
                return

        model = get_attr(kmodel, "diffusion_model")
        set_attr_raw(kmodel, "_model_backup", model)

        set_attr_raw(
            kmodel,
            "diffusion_model",
            torch.compile(model, **config),
        )

        kmodel._compile_config = _cache
