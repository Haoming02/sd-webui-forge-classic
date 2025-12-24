import gradio as gr
import torch
from lib_sve import DecayMethod
from lib_sve.xyz_sve import xyz_support

from modules import scripts
from modules.infotext_utils import PasteField
from modules.processing import StableDiffusionProcessingTxt2Img
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser
from modules.ui_components import InputAccordion


class SeedVarianceEnhancer(scripts.Script):
    sorting_priority = 1125

    enable: bool = False
    seed: int = -1
    XYZ_CACHE: dict[str, str | float] = {}

    steps: int = -1
    percentage: float = 0.0
    strength: int = 0
    decay: str = None
    clamping: float = 1.0

    def __init__(self):
        xyz_support(self.XYZ_CACHE)

    def title(self):
        return "SeedVarianceEnhancer Integrated"

    def show(self, is_img2img):
        return None if is_img2img else scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(value=False, label=self.title()) as enable:
            gr.HTML("Improve seed-to-seed image variance for distilled models <b>(i.e. CFG = 1.0)</b>")
            with gr.Row():
                steps = gr.Slider(value=2, minimum=1, maximum=150, step=1, label="Steps", info="the number of steps to inject random noise")
                percentage = gr.Slider(value=0.6, minimum=0.0, maximum=1.0, step=0.05, label="Percentage", info="the percentage of conditioning to inject random noise")
            with gr.Row():
                strength = gr.Slider(value=24, minimum=0, maximum=64, step=1, label="Strength", info="the strength of the random noise")
                clamping = gr.Slider(value=1.0, minimum=0.0, maximum=1.0, step=0.05, label="Clamping", info="reduce effect strength by clamping the initial noise")
            decay = gr.Dropdown(
                value="No Decay",
                choices=DecayMethod.choices(),
                label="Decay",
                info="apply scaling to the strength based on steps",
            )

        self.infotext_fields = [
            PasteField(enable, "SVE Enable"),
            PasteField(steps, "SVE Steps"),
            PasteField(percentage, "SVE Percentage"),
            PasteField(strength, "SVE Strength"),
            PasteField(decay, "SVE Decay"),
            PasteField(clamping, "SVE Clamping"),
        ]

        return [enable, steps, percentage, strength, decay, clamping]

    def before_process_batch(self, p: StableDiffusionProcessingTxt2Img, enable: bool, steps: int, percentage: float, strength: int, decay: str, clamping: float, **kwargs):
        enable = bool(self.XYZ_CACHE.get("enable", enable))
        SeedVarianceEnhancer.enable = enable
        if not enable:
            return

        SeedVarianceEnhancer.steps = int(self.XYZ_CACHE.get("steps", steps))
        SeedVarianceEnhancer.percentage = float(self.XYZ_CACHE.get("percentage", percentage))
        SeedVarianceEnhancer.strength = int(self.XYZ_CACHE.get("strength", strength))
        SeedVarianceEnhancer.decay = str(self.XYZ_CACHE.get("decay", decay))
        SeedVarianceEnhancer.clamping = float(self.XYZ_CACHE.get("clamping", clamping))
        SeedVarianceEnhancer.seed = kwargs["seeds"][0]

        p.extra_generation_params.update(
            {
                "SVE Enable": enable,
                "SVE Steps": SeedVarianceEnhancer.steps,
                "SVE Percentage": SeedVarianceEnhancer.percentage,
                "SVE Strength": SeedVarianceEnhancer.strength,
                "SVE Decay": SeedVarianceEnhancer.decay,
                "SVE Clamping": SeedVarianceEnhancer.clamping,
            }
        )

        self.XYZ_CACHE.clear()

    @classmethod
    def apply_decay(cls, current_step, total_steps, strength):
        function = DecayMethod.decay_function(cls.decay)
        return function(current_step, total_steps, strength)

    @classmethod
    @torch.inference_mode()
    def on_cfg(cls, params: CFGDenoiserParams):
        if not isinstance(params.denoiser.p, StableDiffusionProcessingTxt2Img) or not cls.enable:
            return
        if params.text_cond is None:
            return
        all_steps: int = min(cls.steps, params.total_sampling_steps)
        if all_steps < params.sampling_step:
            return

        cond: torch.Tensor = params.text_cond
        torch.manual_seed(cls.seed)

        noise_start = torch.clamp(torch.rand_like(cond), min=-cls.clamping, max=cls.clamping)
        strength = cls.apply_decay(params.sampling_step, all_steps, cls.strength)
        noise = noise_start * 2.0 * strength - strength
        noise_mask = torch.bernoulli(noise_start * cls.percentage).bool()

        modified_noise = noise * noise_mask
        params.text_cond = cond + modified_noise


on_cfg_denoiser(SeedVarianceEnhancer.on_cfg)
