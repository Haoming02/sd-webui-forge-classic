# https://github.com/ChangeTheConstants/SeedVarianceEnhancer

import gradio as gr
import torch
import math


from modules import scripts
from modules.infotext_utils import PasteField
from modules.processing import StableDiffusionProcessingTxt2Img
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser
from modules.ui_components import InputAccordion

from lib_sve.xyz_sve import xyz_support


info = """
Improve seed-to-seed image variance for distilled models (<b>i.e.</b> CFG = <code>1.0</code>)<br>
(default parameters are tuned for <b>Z-Image</b> ; lower the values for <b>Qwen-Image</b>)
"""


class SeedVarianceEnhancer(scripts.Script):
    sorting_priority = 1125

    enable: bool = False
    steps: int = -1
    percentage: float = 0.0
    strength: float = 0.0
    preset_checkbox: bool = False
    seed: int = -1
    early_decay: str = "No decay"
    md_threshold1: float = 0.3
    mid_decay: str = "No decay"
    threshold2: float = 0.5
    late_decay: str = "No decay"
    clamping: float = 1.0

    DECAY_FUNCTIONS = {
        "No decay": lambda current_step, total_steps, strength: strength,
        "Linear": lambda current_step, total_steps, strength: 
            strength - (strength - 1) * (current_step / total_steps),
        "Cosine": lambda current_step, total_steps, strength: 
            strength - (strength - 1) * ((1 - math.cos((current_step / total_steps) * math.pi)) / 2),
        "Exponential": lambda current_step, total_steps, strength: 
            strength * ((1 / strength) ** (1 / (total_steps - 1))) ** current_step,
        "Quadratic": lambda current_step, total_steps, strength: 
            1 + (strength - 1) * (1 - current_step / (total_steps - 1)) ** 2,
    }

    def __init__(self):
        self.XYZ_CACHE = {}
        xyz_support(self.XYZ_CACHE)

    def title(self):
        return "SeedVarianceEnhancer Integrated"

    def show(self, is_img2img):
        return None if is_img2img else scripts.AlwaysVisible

    def ui(self, is_img2img):
        components = []
        
        with InputAccordion(value=False, label=self.title()) as enable:
            gr.HTML(info)
            with gr.Row():
                steps = gr.Slider(value=2, minimum=1, maximum=150, step=1, label="Steps", info="the number of steps to inject random noise") # max because decay allows it, minimub because makes no sense
                percentage = gr.Slider(value=0.6, minimum=0.0, maximum=1.0, step=0.05, label="Percentage", info="the percentage of conditioning to inject random noise")
                strength = gr.Slider(value=24, minimum=0, maximum=64, step=1, label="Strength", info="the strength of the random noise")
                preset_checkbox = gr.Checkbox(value=False, label="Recommended", info="Preset for Z-Image Turbo with DPM++ 2s a RF / Beta 8-10 steps")
            with gr.Row():
                early_decay = gr.Dropdown(choices=["No decay", "Linear", "Cosine", "Exponential", "Quadratic"], value="No decay", label="Early Decay", info="Apply decaying function to strength on first third.", )
                md_threshold1 = gr.Slider(value=0.3, minimum=0.0, maximum=1.0, step=0.05, label="Percentage for mid decay", info="Percentage threshold 1 to switch to second type of decay")#idk, but threshold1 and 2 breaks gradio paste
                mid_decay = gr.Dropdown(choices=["No decay", "Linear", "Cosine", "Exponential", "Quadratic"], value="No decay", label="Mid Decay", info="Apply decaying function to strength on second third.")
                threshold2 = gr.Slider(value=0.5, minimum=0.0, maximum=1.0, step=0.05, label="Percentage for late decay", info="Percentage threshold 2 to switch to third type of decay")
                late_decay = gr.Dropdown(choices=["No decay", "Linear", "Cosine", "Exponential", "Quadratic"], value="No decay", label="Late Decay", info="Apply decaying function to strength on last part.")
                clamping = gr.Slider(value=1.0, minimum=0.0, maximum=1.0, step=0.05, label="Clamping for noise", info="Reduce outliers by clamping initial noise. Reduces overall stength")

        
        components = [enable, steps, percentage, strength, preset_checkbox, early_decay, md_threshold1, mid_decay, threshold2, late_decay, clamping]
        
        def validate_thresholds(th1, th2):
            return max(th1, th2)
        
        md_threshold1.change(fn=validate_thresholds, inputs=[md_threshold1, threshold2], outputs=[threshold2])
        threshold2.change(fn=validate_thresholds, inputs=[md_threshold1, threshold2], outputs=[threshold2])
        
        # Function to handle checkbox change for components 
        def handle_lock_change(lock_state, steps_val, percentage_val,
                            dt1_val, th1_val, dt2_val, th2_val, dt3_val, clmp_val):
            if lock_state:
                # Lock ALL components from both rows
                return [
                    # First row components
                    gr.update(interactive=False, value=steps_val),
                    gr.update(interactive=False, value=1.0),
                    # Second row components
                    gr.update(interactive=False, value="Quadratic"),
                    gr.update(interactive=False, value=0.25),
                    gr.update(interactive=False, value="Exponential"),
                    gr.update(interactive=False, value=0.35),
                    gr.update(interactive=False, value="Linear"),
                    gr.update(interactive=False, value=0.75),
                ]
            else:
                # Unlock components from both rows
                return [
                    # First row components
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    # Second row components
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                ]
        
        # Connect checkbox change to components // There may be better way to implement this
        preset_checkbox.change(
            fn=handle_lock_change,
            inputs=[
                preset_checkbox,
                # First row inputs
                steps, percentage, 
                # Second row inputs
                early_decay, md_threshold1, mid_decay, threshold2, late_decay, clamping,
            ],
            outputs=[
                # First row outputs
                steps, percentage, 
                # Second row outputs
                early_decay, md_threshold1, mid_decay, threshold2, late_decay, clamping,
            ]
        )

        self.infotext_fields = [
            PasteField(enable, "SVE Enable"),
            PasteField(steps, "SVE Steps"),
            PasteField(percentage, "SVE Percentage"),
            PasteField(strength, "SVE Strength"),
            PasteField(early_decay, "SVE Early Decay"),
            PasteField(md_threshold1, "SVE Mid Threshold"),
            PasteField(mid_decay, "SVE Mid Decay"),
            PasteField(threshold2, "SVE Late Threshold"),
            PasteField(late_decay, "SVE Late Decay"),
            PasteField(clamping, "SVE Clamping"),
        ]

        return components

    def process(self, p, enable: bool, steps: int, percentage: float, strength: float, preset_checkbox:bool, early_decay: str, md_threshold1: float, mid_decay: str, threshold2: float, late_decay: str, clamping: float, *args, **kwargs):
        # Apply overrides from  XYZ_CACHE
        steps = int(self.XYZ_CACHE.pop("steps", steps))
        percentage = float(self.XYZ_CACHE.pop("percentage", percentage))
        strength = int(self.XYZ_CACHE.pop("strength", strength))
        early_decay = str(self.XYZ_CACHE.pop("early_decay", early_decay))
        md_threshold1 = float(self.XYZ_CACHE.pop("md_threshold1", md_threshold1))
        mid_decay = str(self.XYZ_CACHE.pop("mid_decay", mid_decay))
        threshold2 = float(self.XYZ_CACHE.pop("threshold2", threshold2))
        late_decay = str(self.XYZ_CACHE.pop("late_decay", late_decay))
        clamping = float(self.XYZ_CACHE.pop("clamping", clamping))
        
        self.cached_steps = steps
        self.cached_percentage = percentage
        self.cached_strength = strength
        self.cached_early_decay = early_decay
        self.cached_md_threshold1 = md_threshold1
        self.cached_mid_decay = mid_decay
        self.cached_threshold2 = threshold2
        self.cached_late_decay = late_decay
        self.cached_clamping = clamping
        
        return p


    def before_process_batch(self, p: StableDiffusionProcessingTxt2Img, enable: bool, steps: int, percentage: float, strength: float, preset_checkbox: bool, early_decay: str, md_threshold1: float, mid_decay: str, threshold2: float, late_decay: str, clamping: float, **kwargs):
        SeedVarianceEnhancer.enable = enable
        if not enable:
            return
        
        if hasattr(self, 'cached_steps'):
            steps = self.cached_steps
        
        if hasattr(self, 'cached_percentage'):
            percentage = self.cached_percentage
        
        if hasattr(self, 'cached_strength'):
            strength = self.cached_strength
        
        if hasattr(self, 'cached_early_decay'):
            early_decay = self.cached_early_decay
        
        if hasattr(self, 'cached_md_threshold1'):
            md_threshold1 = self.cached_md_threshold1
        
        if hasattr(self, 'cached_mid_decay'):
            mid_decay = self.cached_mid_decay
        
        if hasattr(self, 'cached_threshold2'):
            if self.cached_threshold2 < self.cached_md_threshold1:#guard for x/y/z
                threshold2 = self.cached_md_threshold1
                print("\n Threshold 2 cannot be < Threshold 1 \n")
            else:
                threshold2 = self.cached_threshold2
        
        if hasattr(self, 'cached_late_decay'):
            late_decay = self.cached_late_decay
        
        if hasattr(self, 'cached_clamping'):
            clamping = self.cached_clamping
        
        SeedVarianceEnhancer.steps = steps
        SeedVarianceEnhancer.percentage = percentage
        SeedVarianceEnhancer.strength = strength
        SeedVarianceEnhancer.early_decay = early_decay
        SeedVarianceEnhancer.md_threshold1 = md_threshold1
        SeedVarianceEnhancer.mid_decay = mid_decay
        SeedVarianceEnhancer.threshold2 = threshold2
        SeedVarianceEnhancer.late_decay = late_decay
        SeedVarianceEnhancer.clamping = clamping
        SeedVarianceEnhancer.seed = kwargs["seeds"][0]

        p.extra_generation_params.update(
            {
                "SVE Enable": enable,
                "SVE Steps": steps,
                "SVE Percentage": percentage,
                "SVE Strength": strength,
                "SVE Early Decay": early_decay,
                "SVE Mid Threshold": md_threshold1,
                "SVE Mid Decay": mid_decay,
                "SVE Late Threshold": threshold2,
                "SVE Late Decay": late_decay,
                "SVE Clamping": clamping,
            }
        )

        if hasattr(self, 'cached_steps'):
            del self.cached_steps
        
        if hasattr(self, 'cached_percentage'):
            del self.cached_percentage
        
        if hasattr(self, 'cached_strength'):
            del self.cached_strength
        
        if hasattr(self, 'cached_early_decay'):
            del self.cached_early_decay
        
        if hasattr(self, 'cached_md_threshold1'):
            del self.cached_md_threshold1
        
        if hasattr(self, 'cached_mid_decay'):
            del self.cached_mid_decay
        
        if hasattr(self, 'cached_threshold2'):
            del self.cached_threshold2
        
        if hasattr(self, 'cached_late_decay'):
            del self.cached_late_decay
        
        if hasattr(self, 'cached_clamping'):
            del self.cached_clamping
        
        self.XYZ_CACHE.clear()

    def apply_decay(decay_type, current_step, total_steps, strength):
        """Apply the specified decay function."""
        if decay_type in SeedVarianceEnhancer.DECAY_FUNCTIONS:
            return SeedVarianceEnhancer.DECAY_FUNCTIONS[decay_type](current_step, total_steps, strength)
        return strength  # Default to no decay
    
    @classmethod
    @torch.inference_mode()
    def on_cfg(cls, params: CFGDenoiserParams):
        if not isinstance(params.denoiser.p, StableDiffusionProcessingTxt2Img) or not cls.enable:
            return
        if params.text_cond is None:
            return
        all_steps = cls.steps
        if cls.preset_checkbox == True:
            all_steps = params.total_sampling_steps
        if all_steps < params.sampling_step: # params.sampling_step starts at 0
            return

        # Apply decay logic to strength
        current_strength = cls.strength
        # Normalise steps because internally they start from 0 to steps - 1
        current_step = params.sampling_step
        
        if  cls.strength != 0: # = disabled
            end_step1 = max(1, int(cls.md_threshold1 * all_steps))
            end_step2 = max(1, int(cls.threshold2 * all_steps))
            #Calculate transition strength 1
            thresh1_strength = cls.apply_decay(
                    cls.early_decay, end_step1, all_steps, cls.strength
                )
            thresh2_strength = cls.apply_decay(
                    cls.mid_decay, end_step2, all_steps - end_step1, thresh1_strength
                )
            if current_step <= end_step1:
                current_strength = cls.apply_decay(
                    cls.early_decay, current_step, all_steps, cls.strength
                )
                #print("\n", cls.early_decay," ", current_strength, "\n")
            elif current_step <= end_step2:
                current_strength = cls.apply_decay(
                    cls.mid_decay, current_step - end_step1, all_steps - end_step1, thresh1_strength
                )
                #print("\n", cls.mid_decay," ", current_strength, "\n")
            else:
                current_strength = cls.apply_decay(
                    cls.late_decay, current_step - end_step2, all_steps - end_step2, thresh2_strength
                )
                #print("\n", cls.late_decay," ", current_strength, "\n")

        cond: torch.Tensor = params.text_cond
        torch.manual_seed(cls.seed)
        
        noise_start = torch.clamp(torch.rand_like(cond), min=-cls.clamping, max=cls.clamping)
        noise = noise_start * 2.0 * current_strength - current_strength
        noise_mask = torch.bernoulli(noise_start * cls.percentage).bool()
        
        modified_noise = noise * noise_mask
        params.text_cond = cond + modified_noise
        #print(f"Min value: {noise_start.min():.6f}")
        #print(f"Max value: {noise_start.max():.6f}")
        #print("\n Strength:", current_strength," Step ",current_step, "\n")


on_cfg_denoiser(SeedVarianceEnhancer.on_cfg)
