# Reference:  https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_eps.py
# Credit:     https://arxiv.org/abs/2308.15321v6

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.patcher.unet import UnetPatcher


class EpsilonScaling:
    """
    Implements the Epsilon Scaling method from 'Elucidating the Exposure Bias in Diffusion Models'

    This method mitigates exposure bias by scaling the predicted noise during sampling,
    which can significantly improve sample quality. This implementation uses the "uniform schedule"
    recommended by the paper for its practicality and effectiveness.
    """

    @staticmethod
    def patch(model: "UnetPatcher", scaling_factor: float):

        def epsilon_scaling_function(args):
            """
            This function is applied after the CFG guidance has been calculated.
            It recalculates the denoised latent by scaling the predicted noise.
            """
            denoised = args["denoised"]
            x = args["input"]

            noise_pred = x - denoised

            scaled_noise_pred = noise_pred / scaling_factor

            new_denoised = x - scaled_noise_pred

            return new_denoised

        m = model.clone()
        m.set_model_sampler_post_cfg_function(epsilon_scaling_function)
        return m
