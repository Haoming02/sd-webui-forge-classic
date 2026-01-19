import gradio as gr

from modules import scripts, sd_samplers, sd_schedulers
from modules.infotext_utils import PasteField
from modules.ui_components import FormRow


class ScriptSampler(scripts.ScriptBuiltinUI):
    section = "sampler"

    def __init__(self):
        self.steps = None
        self.sampler_name = None
        self.scheduler = None

    def title(self):
        return "Sampler"

    def ui(self, is_img2img):
        sampler_names = [x.name for x in sd_samplers.visible_samplers()]
        scheduler_names = [x.label for x in sd_schedulers.schedulers]

        with FormRow(elem_id=f"sampler_selection_{self.tabname}"):
            self.sampler_name = gr.Dropdown(label="Sampling Method", elem_id=f"{self.tabname}_sampling", choices=sampler_names, value=sampler_names[0])
            self.scheduler = gr.Dropdown(label="Schedule Type", elem_id=f"{self.tabname}_scheduler", choices=scheduler_names, value=scheduler_names[0])
            self.steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{self.tabname}_steps", label="Sampling Steps", value=20)

        self.infotext_fields = [
            PasteField(self.steps, "Steps", api="steps"),
            PasteField(self.sampler_name, sd_samplers.get_sampler_from_infotext, api="sampler_name"),
            PasteField(self.scheduler, sd_samplers.get_scheduler_from_infotext, api="scheduler"),
        ]

        return self.steps, self.sampler_name, self.scheduler

    def setup(self, p, steps, sampler_name, scheduler):
        p.steps = steps
        p.sampler_name = sampler_name
        p.scheduler = scheduler
