# https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_torch_compile.py

import logging
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.modules.k_model import KModel

import gradio as gr
import torch

from backend.args import args as cmd_args
from backend.logging import setup_logger
from backend.utils import get_attr, set_attr_raw
from modules import scripts

try:
    import triton
except ImportError:
    TRITON_AVAILABLE = False
else:
    TRITON_AVAILABLE = True

_COMPILE_CONFIG_KEY = "_torch_compile_config"
_ORIG_APPLY_KEY = "_orig_apply_model"

logger = logging.getLogger("compile")
setup_logger(logger)


def skip_torch_compile_dict(guard_entries):
    return [("transformer_options" not in entry.name) for entry in guard_entries]


class TorchCompileForForge(scripts.Script):
    sorting_priority = 99999

    def __init__(self):
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.suppress_errors = True

    def title(self):
        return "Torch Compile Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if TRITON_AVAILABLE else None

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            preset = gr.Dropdown(
                label="Preset",
                value="Automatic",
                choices=[
                    "Automatic",
                    "Disable",
                    "guard_filter_fn",
                    "dynamic",
                    "max-autotune",
                    "max-autotune-no-cudagraphs",
                    "reduce-overhead",
                ],
                info='"Automatic" maintains the current compile status',
            )

            _dynamic = "Support any Resolution / Batch Size"
            _indynamic = "Require recompilation if Resolution / Batch Size is changed"
            _no_malloc = "Does not work with --cuda-malloc"

            gr.Markdown(rf"""
**torch.compile** speeds up the inference by compiling the model ahead of time
- **guard_filter_fn:** Compile the Fastest ; {_indynamic}
- **dynamic:** {_dynamic} ; Slower to Compile
- **max-autotune:** Best Runtime Speed ; {_indynamic} ; {_no_malloc}
- **max-autotune-no-cudagraphs:** {_dynamic} ; Faster than **dynamic** ; Even Slower to Compile
- **reduce-overhead:** Similar to **max-autotune** ; {_indynamic} ; {_no_malloc}
            """)

        return [preset]

    def process_batch(self, p, preset: str, **kwargs):
        kmodel: "KModel" = p.sd_model.forge_objects.unet.model
        prev_config: tuple[str] = getattr(kmodel, _COMPILE_CONFIG_KEY, None)
        enable: bool = (prev_config is not None) if preset == "Automatic" else (preset != "Disable")

        if not enable:
            self._remove_compile_wrapper(kmodel)
            return

        if preset in ("max-autotune", "reduce-overhead") and cmd_args.cuda_malloc:
            logger.error(f"{preset} does not support --cuda-malloc\nModel is not compiled...")
            return

        _config: tuple[str] = (preset,)
        if _config == prev_config:
            return

        setattr(kmodel, _COMPILE_CONFIG_KEY, _config)

        if prev_config is not None:
            self._remove_compile_wrapper(kmodel)

        match preset:
            case "guard_filter_fn":
                config = dict(backend="inductor", dynamic=False, fullgraph=False, options={"guard_filter_fn": skip_torch_compile_dict})
            case "dynamic":
                config = dict(backend="inductor", dynamic=True, fullgraph=False)
            case "max-autotune":
                config = dict(backend="inductor", dynamic=False, fullgraph=False, options={"coordinate_descent_tuning": True, "max_autotune": True, "triton.cudagraphs": True})
            case "max-autotune-no-cudagraphs":
                config = dict(backend="inductor", dynamic=True, fullgraph=False, options={"coordinate_descent_tuning": True, "max_autotune": True})
            case "reduce-overhead":
                config = dict(backend="inductor", mode="reduce-overhead", dynamic=False, fullgraph=False)

        self._wrap_apply_model(kmodel, config)

        logger.info(f"Model Compiled ({preset})")

    @staticmethod
    def _wrap_apply_model(kmodel: "KModel", compile_config: dict):
        setattr(kmodel, _ORIG_APPLY_KEY, kmodel.apply_model)

        @wraps(kmodel._orig_apply_model)
        def apply_model_with_compile(*args, **kwargs):
            orig_model = get_attr(kmodel, "diffusion_model")
            compiled = torch.compile(orig_model, **compile_config)
            set_attr_raw(kmodel, "diffusion_model", compiled)
            try:
                return kmodel._orig_apply_model(*args, **kwargs)
            finally:
                set_attr_raw(kmodel, "diffusion_model", orig_model)

        kmodel.apply_model = apply_model_with_compile

    @staticmethod
    def _remove_compile_wrapper(kmodel: "KModel"):
        if (orig := getattr(kmodel, _ORIG_APPLY_KEY, None)) is not None:
            kmodel.apply_model = orig
            delattr(kmodel, _ORIG_APPLY_KEY)
            delattr(kmodel, _COMPILE_CONFIG_KEY)
            logger.info("Model Decompiled")
