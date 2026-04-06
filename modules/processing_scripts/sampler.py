import json

import gradio as gr

from modules import scripts, sd_samplers, sd_schedulers, shared
from modules.infotext_utils import PasteField
from modules.ui_components import FormRow


def _get_saved_sampler_ui_default(tabname, label, default_value):
    key = f"customscript/sampler.py/{tabname}/{label}/value"
    ui_config_file = getattr(shared.cmd_opts, "ui_config_file", None)

    if not ui_config_file:
        return default_value

    try:
        with open(ui_config_file, "r", encoding="utf8") as file:
            ui_settings = json.load(file)
    except Exception:
        return default_value

    return ui_settings.get(key, default_value)


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

        default_sampler = _get_saved_sampler_ui_default(self.tabname, "Sampling Method", sampler_names[0])
        if default_sampler not in sampler_names:
            default_sampler = sampler_names[0]

        default_scheduler = _get_saved_sampler_ui_default(self.tabname, "Schedule Type", scheduler_names[0])
        if default_scheduler not in scheduler_names:
            default_scheduler = scheduler_names[0]

        default_steps = _get_saved_sampler_ui_default(self.tabname, "Sampling Steps", 20)
        try:
            default_steps = int(default_steps)
        except Exception:
            default_steps = 20

        default_steps = max(1, min(default_steps, 150))

        with FormRow(elem_id=f"sampler_selection_{self.tabname}"):
            self.sampler_name = gr.Dropdown(label="Sampling Method", elem_id=f"{self.tabname}_sampling", choices=sampler_names, value=default_sampler)
            self.scheduler = gr.Dropdown(label="Schedule Type", elem_id=f"{self.tabname}_scheduler", choices=scheduler_names, value=default_scheduler)
            self.steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{self.tabname}_steps", label="Sampling Steps", value=default_steps)

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
