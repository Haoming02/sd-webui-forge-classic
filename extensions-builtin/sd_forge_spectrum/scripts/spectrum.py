import gradio as gr
from lib_spectrum.forecaster import SpectrumNode
from lib_spectrum.presets import PresetManager

from modules import scripts, shared
from modules.infotext_utils import PasteField
from modules.ui_components import InputAccordion

PresetManager.load_presets()


class SpectrumForForge(scripts.Script):
    sorting_priority = 2026

    def title(self):
        return "Spectrum Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enable:
            with gr.Row():
                w = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.25,
                    step=0.05,
                    label="Prediction Weighting",
                    info="higher = long-term trend ; lower = short-term changes",
                )
                m = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=6,
                    step=1,
                    label="Polynomial Degree",
                    info="higher = complex & subtle patterns ; lower = stable & faster",
                )
            with gr.Row():
                lam = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.5,
                    step=0.05,
                    label="Regularization",
                    info="higher = reduce overfitting ; lower = fit more data",
                )
                window_size = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=2,
                    step=1,
                    label="Cache Window",
                    info="higher = skip more steps ; lower = slower but more accurate",
                )
            flex_window = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.0,
                step=0.05,
                label="Window Growth",
                info="higher = more speed & less accurate ; lower = more consistent accuracy but less speed gain",
            )
            with gr.Row():
                warmup_steps = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=6,
                    step=1,
                    label="Warmup Steps",
                    info="Run the full model before caching starts",
                )
                stop_caching_step = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Stop Caching Step",
                    info="Run the full model for the last few steps",
                )

            with gr.Accordion("Presets", open=False):
                _preset = gr.Dropdown(
                    value=None,
                    label="Preset Name",
                    choices=PresetManager.list_preset(),
                    allow_custom_value=True,
                )
                with gr.Row():
                    _load = gr.Button("Apply Preset", variant="secondary")
                    _save = gr.Button("Save Preset", variant="primary")
                    _del = gr.Button("Delete Preset", variant="stop")

                for comp in (_preset, _load, _save, _del):
                    comp.do_not_save_to_config = True

                args = (w, m, lam, window_size, flex_window, warmup_steps, stop_caching_step)

                _load.click(
                    fn=lambda name: PresetManager.get_preset(name),
                    inputs=[_preset],
                    outputs=[*args],
                    queue=False,
                )
                _save.click(
                    fn=lambda *args: PresetManager.save_preset(*args),
                    inputs=[_preset, *args],
                    outputs=[_preset],
                    queue=False,
                )
                _del.click(
                    fn=lambda name: PresetManager.delete_preset(name),
                    inputs=[_preset],
                    outputs=[_preset],
                    queue=False,
                )

        self.infotext_fields = [
            PasteField(w, "spec_w"),
            PasteField(m, "spec_m"),
            PasteField(lam, "spec_lam"),
            PasteField(window_size, "spec_window_size"),
            PasteField(flex_window, "spec_flex_window"),
            PasteField(warmup_steps, "spec_warmup_steps"),
            PasteField(stop_caching_step, "spec_stop_caching_step"),
        ]
        self.paste_field_names = [field.label for field in self.infotext_fields]

        return [enable, w, m, lam, window_size, flex_window, warmup_steps, stop_caching_step]

    def process_before_every_sampling(self, p, enable: bool, *args, **kwargs):
        if not enable:
            return

        if shared.opts.skip_early_cond > 0.0 or shared.opts.s_min_uncond > 0.0:
            print('Spectrum does not support "Ignore/Skip Negative Prompt" optimizations...')
            return

        unet = p.sd_model.forge_objects.unet
        unet = SpectrumNode.patch(unet, p.steps, *args)
        p.sd_model.forge_objects.unet = unet

        for k, v in zip(["spec_w", "spec_m", "spec_lam", "spec_window_size", "spec_flex_window", "spec_warmup_steps", "spec_stop_caching_step"], args):
            p.extra_generation_params[k] = v
