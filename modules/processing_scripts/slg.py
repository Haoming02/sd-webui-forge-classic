# https://github.com/Comfy-Org/ComfyUI/blob/v0.42.0/comfy_extras/nodes_slg.py

import re
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

from backend.args import dynamic_args
from backend.patcher.base import set_model_options_patch_replace
from backend.sampling import sampling_function
from modules import scripts
from modules.shared import opts

_calc_cond_uncond_batch = sampling_function.calc_cond_uncond_batch


@wraps(_calc_cond_uncond_batch)
def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options):
    if (calc_batch_function := model_options.get("sampler_calc_cond_batch_function", None)) is None:
        return _calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options)
    return calc_batch_function(model, cond, uncond, x_in, timestep, model_options)


sampling_function.calc_cond_uncond_batch = calc_cond_uncond_batch


def parse_layers(layers: str) -> list[int]:
    return [int(layer) for layer in re.findall(r"\d+", layers)]


class SkipLayerGuidance(scripts.Script):
    def title(self):
        return "Skip Layer Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process_before_every_sampling(self, p: "StableDiffusionProcessing", *args, **kwargs):
        if not (dynamic_args.moss and opts.slg_enable):
            return

        double_layers = parse_layers("")  # Moss-Image has no double_block
        single_layers = parse_layers(opts.slg_single_layers)

        if len(double_layers) == 0 and len(single_layers) == 0:
            return

        p_start = max(min(opts.slg_start_percent, 1.0), 0.0)
        p_end = max(min(opts.slg_end_percent, 1.0), 0.0)

        unet = p.sd_model.forge_objects.unet
        predictor = unet.model.predictor
        sigma_start = predictor.percent_to_sigma(p_start)
        sigma_end = predictor.percent_to_sigma(p_end)

        calc_batch_function = self.make_calc_cond_batch_function(double_layers, single_layers, sigma_start, sigma_end)

        m = unet.clone()
        m.set_model_sampler_calc_cond_batch_function(calc_batch_function)
        p.sd_model.forge_objects.unet = m

        p.extra_generation_params.update(
            {
                "Skip Layer Guidance": True,
                "SLG Single Layers": ", ".join(map(str, single_layers)) if single_layers else None,
                "SLG Start Percent": p_start,
                "SLG End Percent": p_end,
            }
        )

    @staticmethod
    def make_calc_cond_batch_function(double_layers: list[int], single_layers: list[int], sigma_start: float, sigma_end: float):

        def skip(args, extra_args):
            return args

        def calc_batch_function(model, cond, uncond, x, sigma, model_options):
            slg_model_options = model_options.copy()

            for layer in double_layers:
                slg_model_options = set_model_options_patch_replace(slg_model_options, skip, "dit", "double_block", layer)

            for layer in single_layers:
                slg_model_options = set_model_options_patch_replace(slg_model_options, skip, "dit", "single_block", layer)

            sigma_ = sigma[0].item()

            if sigma_ >= sigma_end and sigma_ <= sigma_start and uncond is not None:
                cond_out, _ = _calc_cond_uncond_batch(model, cond, None, x, sigma, model_options)
                _, uncond_out = _calc_cond_uncond_batch(model, [], uncond, x, sigma, slg_model_options)
                return [cond_out, uncond_out]

            return _calc_cond_uncond_batch(model, cond, uncond, x, sigma, model_options)

        return calc_batch_function
