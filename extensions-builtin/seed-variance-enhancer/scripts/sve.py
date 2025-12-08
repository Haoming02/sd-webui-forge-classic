# https://github.com/ChangeTheConstants/SeedVarianceEnhancer

import gradio as gr
import torch

from modules import scripts
from modules.infotext_utils import PasteField
from modules.processing import StableDiffusionProcessingTxt2Img
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser
from modules.ui_components import InputAccordion


info = """
Improve seed-to-seed image variance for distilled models (<b>i.e.</b> CFG = <code>1.0</code>)<br>
(default parameters are tuned for <b>Z-Image</b> ; lower the values for <b>Qwen-Image</b>)
"""


class SeedVarianceEnhancer(scripts.Script):
    sorting_priority = 1125

    enable: bool
    steps: int
    percentage: float
    strength: float
    seed: int

    def title(self):
        return "SeedVarianceEnhancer Integrated"

    def show(self, is_img2img):
        return None if is_img2img else scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(value=False, label=self.title()) as enable:
            gr.HTML(info)
            with gr.Row():
                steps = gr.Slider(value=3, minimum=0, maximum=8, step=1, label="Steps", info="the number of steps to inject random noise")
                percentage = gr.Slider(value=0.6, minimum=0.0, maximum=1.0, step=0.05, label="Percentage", info="the percentage of conditioning to inject random noise")
                strength = gr.Slider(value=32, minimum=0, maximum=64, step=1, label="Strength", info="the strength of the random noise")

        self.infotext_fields = [
            PasteField(steps, "SVE Steps"),
            PasteField(percentage, "SVE Percentage"),
            PasteField(strength, "SVE Strength"),
        ]

        return [enable, steps, percentage, strength]

    def before_process_batch(self, p: StableDiffusionProcessingTxt2Img, enable: bool, steps: int, percentage: float, strength: int, **kwargs):
        SeedVarianceEnhancer.enable = enable and isinstance(p, StableDiffusionProcessingTxt2Img)
        if not SeedVarianceEnhancer.enable:
            return

        SeedVarianceEnhancer.steps = steps
        SeedVarianceEnhancer.percentage = percentage
        SeedVarianceEnhancer.strength = strength
        SeedVarianceEnhancer.seed = kwargs["seeds"][0]

        p.extra_generation_params.update(
            {
                "SVE Steps": steps,
                "SVE Percentage": percentage,
                "SVE Strength": strength,
            }
        )

    @classmethod
    @torch.inference_mode()
    def on_cfg(cls, params: CFGDenoiserParams):
        if not cls.enable:
            return
        if cls.steps < params.sampling_step:
            return

        cond: torch.Tensor = params.text_cond
        torch.manual_seed(cls.seed)

        noise = torch.rand_like(cond) * 2.0 * cls.strength - cls.strength
        noise_mask = torch.bernoulli(torch.ones_like(cond) * cls.percentage).bool()

        modified_noise = noise * noise_mask
        params.text_cond = cond + modified_noise


on_cfg_denoiser(SeedVarianceEnhancer.on_cfg)
