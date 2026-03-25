from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.modules.k_model import KModel

import gradio as gr
import torch

from backend.utils import get_attr, set_attr_raw
from modules import scripts

# Base options for inductor modes, to then add advanced options alongside these. These come directly from torch._inductor.config.
_MODE_BASE_OPTIONS: dict[str, dict] = {
    "max-autotune": {
        "coordinate_descent_tuning": True,
        "max_autotune": True,
        "triton.cudagraphs": True,
    },
    "max-autotune-no-cudagraphs": {
        "coordinate_descent_tuning": True,
        "max_autotune": True,
    },
    "reduce-overhead": {
        "triton.cudagraphs": True,
    },
}


def skip_torch_compile_dict(guard_entries):
    # https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_torch_compile.py#L5
    return [("transformer_options" not in entry.name) for entry in guard_entries]


class TorchCompileForForge(scripts.Script):
    sorting_priority = 67

    def __init__(self):
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.suppress_errors = True

    def title(self):
        return "Torch Compile Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.Markdown(
                """
**torch.compile** speeds up the Inference by compiling the model ahead of time.
- **guard_filter_fn:** Compile the Fastest ; Require recompilation if Resolution / Batch Size is changed.
- **dynamic:** Longer to Compile ; Support any Resolution / Batch Size.
- **max-autotune:** CUDA graphs + coordinate_descent_tuning — best runtime speed, doesn't work with CUDA Malloc. Require recompilation if Resolution / Batch Size is changed.
- **max-autotune-no-cudagraphs:** Triton autotuned kernels + coordinate descent tuning, no CUDA graphs. Faster than dynamic but takes longer to compile. Support any Resolution / Batch Size.
- **reduce-overhead:** CUDA graphs via inductor, lower compile cost than max-autotune. Doesn't work with CUDA Malloc. Require recompilation if Resolution / Batch Size is changed.
- **cudagraphs:** Not Recommended. Doesn't work with CUDA Malloc.
                """
            )
            preset = gr.Dropdown(
                label="Preset",
                value="Automatic",
                choices=[
                    "Automatic",
                    "Disable",
                    "guard_filter_fn",
                    "dynamic",
                    "max-autotune-no-cudagraphs",
                    "max-autotune",
                    "reduce-overhead",
                    "cudagraphs",
                ],
                info='"Automatic" maintains the current compile status',
            )

            with gr.Accordion(open=False, label="Advanced Options"):
                gr.Markdown(
                    """
These options are layered on top of the selected preset (inductor backend only, so it doesn't apply on cudagraphs).
When combined with a mode-based preset (`max-autotune`, `max-autotune-no-cudagraphs`, `reduce-overhead`),
the mode is first expanded to its exact base options and then your selections are merged on top —
so no mode options are lost.

- **dynamic shapes:** Override the preset's dynamic setting — allows any resolution/batch size at the cost of longer compile. Note: incompatible with `triton.cudagraphs` and CUDA-graph-based presets
- **max_autotune_pointwise:** Autotune pointwise/reduction kernels independently (subset of `max_autotune`)
- **max_autotune_gemm:** Autotune GEMM kernels independently (subset of `max_autotune`)
- **epilogue_fusion:** Fuse pointwise ops into the preceding kernel — requires `max_autotune` or `max_autotune_gemm`
- **shape_padding:** Pad tensor shapes to powers of 2 for better kernel alignment
- **fallback_random:** Allow graph fallback for random ops (improves capture rate)
- **triton.cudagraphs:** Enable CUDA graphs within inductor (distinct from the `cudagraphs` backend preset)
- **coordinate_descent_tuning:** Tune tile configs via coordinate descent — included in both max-autotune modes, expose here for other presets
                    """
                )
                dynamic                   = gr.Checkbox(label="dynamic shapes — support any resolution/batch size (overrides preset default)", value=False)
                epilogue_fusion           = gr.Checkbox(label="epilogue_fusion",           value=False)
                shape_padding             = gr.Checkbox(label="shape_padding",             value=False)
                fallback_random           = gr.Checkbox(label="fallback_random",           value=False)
                triton_cudagraphs         = gr.Checkbox(label="triton.cudagraphs",         value=False)
                coordinate_descent_tuning = gr.Checkbox(label="coordinate_descent_tuning", value=False)
                max_autotune_pointwise    = gr.Checkbox(label="max_autotune_pointwise",    value=False)
                max_autotune_gemm         = gr.Checkbox(label="max_autotune_gemm",         value=False)

        return [
            preset,
            dynamic,
            epilogue_fusion,
            shape_padding,
            fallback_random,
            triton_cudagraphs,
            coordinate_descent_tuning,
            max_autotune_pointwise,
            max_autotune_gemm,
        ]

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

    def process_batch(
        self,
        p,
        preset: str,
        dynamic: bool,
        epilogue_fusion: bool,
        shape_padding: bool,
        fallback_random: bool,
        triton_cudagraphs: bool,
        coordinate_descent_tuning: bool,
        max_autotune_pointwise: bool,
        max_autotune_gemm: bool,
        **kwargs,
    ):
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
            case "max-autotune-no-cudagraphs":
                config = dict(backend="inductor", mode="max-autotune-no-cudagraphs", dynamic=True, fullgraph=False)
            case "max-autotune":
                config = dict(backend="inductor", mode="max-autotune", dynamic=False, fullgraph=False)
            case "reduce-overhead":
                config = dict(backend="inductor", mode="reduce-overhead", dynamic=False, fullgraph=False)
            case "cudagraphs":
                config = dict(backend="cudagraphs", dynamic=True, fullgraph=True)
            case _:
                config: dict = kmodel._compile_config
 
        if dynamic and config.get("backend") == "inductor":
            config["dynamic"] = True

        if config.get("backend") == "inductor":
            extra_options = {}
            if epilogue_fusion:            extra_options["epilogue_fusion"]           = True
            if shape_padding:              extra_options["shape_padding"]             = True
            if fallback_random:            extra_options["fallback_random"]           = True
            if triton_cudagraphs:          extra_options["triton.cudagraphs"]         = True
            if coordinate_descent_tuning:  extra_options["coordinate_descent_tuning"] = True
            if max_autotune_pointwise:     extra_options["max_autotune_pointwise"]    = True
            if max_autotune_gemm:          extra_options["max_autotune_gemm"]         = True

            if extra_options:
                # torch.compile forbids mode + options together.
                # Pop the mode key and seed base with its exact known expansion
                # then merge the user's checkboxes on top — user values win on conflict.
                mode = config.pop("mode", None)
                base = _MODE_BASE_OPTIONS.get(mode, {}).copy() if mode else {}
                base.update(extra_options)
                config.setdefault("options", {}).update(base)

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
