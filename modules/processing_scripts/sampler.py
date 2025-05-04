import gradio as gr

from modules import scripts, sd_samplers, sd_schedulers, shared
from modules.infotext_utils import PasteField
from modules.ui_components import FormRow


class ScriptSampler(scripts.ScriptBuiltinUI):
    create_group = False
    section = "sampler"

    def __init__(self):
        self.steps = None
        self.sampler_name = None
        self.scheduler = None

    def title(self):
        return "Sampler"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        sampler_names: list[str] = sd_samplers.visible_sampler_names()
        scheduler_names: list[str] = [x.label for x in sd_schedulers.schedulers]

        with FormRow(elem_id=f"sampler_selection_{self.tabname}"):
            self.sampler_name = gr.Dropdown(
                label="Sampling method",
                elem_id=f"{self.tabname}_sampling",
                choices=sampler_names,
                value=sampler_names[0],
            )
            if shared.opts.show_scheduler:
                self.scheduler = gr.Dropdown(
                    label="Schedule type",
                    elem_id=f"{self.tabname}_scheduler",
                    choices=scheduler_names,
                    value=scheduler_names[0],
                )
            else:
                self.scheduler = gr.State(value="Automatic")
                self.scheduler.do_not_save_to_config = True
            self.steps = gr.Slider(
                minimum=1,
                maximum=150,
                step=1,
                elem_id=f"{self.tabname}_steps",
                label="Sampling steps",
                value=20,
            )

        self.infotext_fields = [
            PasteField(self.steps, "Steps", api="steps"),
            PasteField(self.sampler_name, sd_samplers.get_sampler_from_infotext, api="sampler_name"),
        ]

        if shared.opts.show_scheduler:
            self.infotext_fields.append(PasteField(self.scheduler, sd_samplers.get_scheduler_from_infotext, api="scheduler"))

        return self.steps, self.sampler_name, self.scheduler

    def setup(self, p, steps, sampler_name, scheduler):
        p.steps = steps
        p.sampler_name = sampler_name
        p.scheduler = scheduler
