from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.modules.k_model import KModel

import gradio as gr
import torch

from backend.utils import get_attr, set_attr_raw
from modules import scripts


def skip_torch_compile_dict(guard_entries):
    # https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_torch_compile.py#L5
    return [("transformer_options" not in entry.name) for entry in guard_entries]


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
            gr.Markdown(
                """
**torch.compile** speeds up the Inference by compiling the model ahead of time
- **guard_filter_fn:** Compile the Fastest ; Require recompilation if Resolution / Batch Size is changed
- **dynamic:** Longer to Compile ; Support any Resolution / Batch Size
- **cudagraphs:** Not Recommended
                """
            )
            preset = gr.Dropdown(
                label="Preset",
                value="Automatic",
                choices=["Automatic", "Disable", "guard_filter_fn", "dynamic", "cudagraphs"],
                info='"Automatic" maintains the current compile status',
            )

        return [preset]

    @staticmethod
    def restore(kmodel: "KModel"):
        model = get_attr(kmodel, "_model_backup")
        set_attr_raw(kmodel, "diffusion_model", model)
        del kmodel._compile_config
        del kmodel._compiled_backup
        del kmodel._model_backup

    def before_process_batch(self, p, *args, **kwargs):
        kmodel: "KModel" = p.sd_model.forge_objects.unet.model
        if not hasattr(kmodel, "_compile_config"):
            return

        c_model = get_attr(kmodel, "diffusion_model")
        set_attr_raw(kmodel, "_compiled_backup", c_model)
        # temporarily restores the original model so LoRA can apply
        model = get_attr(kmodel, "_model_backup")
        set_attr_raw(kmodel, "diffusion_model", model)

    def process_batch(self, p, preset: str, **kwargs):
        kmodel: "KModel" = p.sd_model.forge_objects.unet.model
        compiled: bool = hasattr(kmodel, "_compile_config")
        enable: bool = compiled if preset == "Automatic" else (preset != "Disable")

        if not enable:
            if compiled:
                self.restore(kmodel)
            return

        match preset:
            case "guard_filter_fn":
                config = dict(backend="inductor", dynamic=False, fullgraph=False, options={"guard_filter_fn": skip_torch_compile_dict})
            case "dynamic":
                config = dict(backend="inductor", dynamic=True, fullgraph=False)
            case "cudagraphs":
                config = dict(backend="cudagraphs", dynamic=True, fullgraph=True)
            case _:
                config: dict = kmodel._compile_config

        if compiled:
            if kmodel._compile_config == config:
                c_model = get_attr(kmodel, "_compiled_backup")
                set_attr_raw(kmodel, "diffusion_model", c_model)
                del kmodel._compiled_backup
                return

            self.restore(kmodel)

        model = get_attr(kmodel, "diffusion_model")
        set_attr_raw(kmodel, "_model_backup", model)

        set_attr_raw(
            kmodel,
            "diffusion_model",
            torch.compile(model, **config),
        )

        kmodel._compile_config = config
