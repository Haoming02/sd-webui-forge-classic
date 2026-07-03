import os
from functools import partial

import gradio as gr

from modules import call_queue, errors, sd_models, shared
from modules.extras import InterpolationMethod, read_metadata, run_modelmerger
from modules.ui_common import create_refresh_button
from modules.ui_components import FormRow, InputAccordion


def update_method(value: str = None) -> list[bool, bool, bool, str]:
    return [
        gr.update(visible=(value != InterpolationMethod.no_interpolation.value)),
        gr.update(visible=(value != InterpolationMethod.no_interpolation.value)),
        gr.update(visible=(value == InterpolationMethod.add_difference.value)),
        gr.update(info=InterpolationMethod.desc(value)),
    ]


def modelmerger(*args):
    try:
        return run_modelmerger(*args)
    except Exception as e:
        errors.report("Error running ModelMerger", exc_info=True)
        return [gr.skip(), gr.skip(), gr.skip(), gr.skip(), f"Error: {e}"]


class UiCheckpointMerger:
    def __init__(self):
        with gr.Blocks(analytics_enabled=False) as modelmerger_interface:
            with gr.Accordion(label="Save Current Model (only supports SD1 / SDXL / Flux)", open=False):
                with gr.Row():
                    filename_forge = gr.Textbox(value="flux.safetensors", lines=1, max_lines=1, label="Filename", info='(will save in "Stable-diffusion" folder)', scale=2)
                    btn_save_unet_forge = gr.Button("Save UNet Only", scale=1)
                    btn_save_ckpt_forge = gr.Button("Save Checkpoint (UNet + Clip + VAE)", scale=1)
                result_html = gr.HTML("Pending...")

                def save_model(filename: str, *, _unet: bool):
                    sd_models.forge_model_reload()

                    long_filename = os.path.join(sd_models.model_path, filename)
                    os.makedirs(sd_models.model_path, exist_ok=True)

                    try:
                        if _unet:
                            p = shared.sd_model.save_unet(long_filename)
                            gr.Info("Success!", duration=5)
                            return f'UNet saved to "{p}"'
                        else:
                            p = shared.sd_model.save_checkpoint(long_filename)
                            gr.Info("Success", duration=5)
                            return f'Checkpoint saved to "{p}"'
                    except Exception as e:
                        gr.Warning("Failed...", duration=5)
                        return str(e)

                btn_save_unet_forge.click(partial(save_model, _unet=True), inputs=filename_forge, outputs=result_html)
                btn_save_ckpt_forge.click(partial(save_model, _unet=False), inputs=filename_forge, outputs=result_html)

                for comp in (filename_forge, btn_save_unet_forge, btn_save_ckpt_forge):
                    comp.do_not_save_to_config = True

            with gr.Row(equal_height=False):
                with gr.Column(variant="compact"):
                    with FormRow(elem_id="modelmerger_models"):
                        self.primary_model_name = gr.Dropdown(label="Primary Model (A)", choices=sorted(sd_models.checkpoint_tiles()), elem_id="modelmerger_primary_model_name", visible=True)
                        self.secondary_model_name = gr.Dropdown(label="Secondary Model (B)", choices=sorted(sd_models.checkpoint_tiles()), elem_id="modelmerger_secondary_model_name", visible=True)
                        self.tertiary_model_name = gr.Dropdown(label="Tertiary Model (C)", choices=sorted(sd_models.checkpoint_tiles()), elem_id="modelmerger_tertiary_model_name", visible=False)
                        create_refresh_button(
                            [self.primary_model_name, self.secondary_model_name, self.tertiary_model_name],
                            sd_models.list_models,
                            lambda: {"choices": sorted(sd_models.checkpoint_tiles())},
                            "modelmerger_refresh_checkpoint",
                        )

                    with FormRow():
                        self.interp_method = gr.Radio(choices=InterpolationMethod.titles(), value=InterpolationMethod.weighted_sum.value, info=InterpolationMethod.desc(InterpolationMethod.weighted_sum.value), label="Interpolation Method", elem_id="modelmerger_interp_method")
                        self.interp_amount = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Multiplier (M)", elem_id="modelmerger_interp_amount")
                        self.interp_method.change(fn=update_method, inputs=[self.interp_method], outputs=[self.interp_amount, self.secondary_model_name, self.tertiary_model_name, self.interp_method], show_progress=False, queue=False)

                    with FormRow():
                        self.custom_name = gr.Textbox(label="Filename to Save", info="will be based on Interpolation Method if empty", lines=1, max_lines=1, elem_id="modelmerger_custom_name")
                        self.discard_weights = gr.Textbox(label="Layers to Discard", info="in Regex format", lines=1, max_lines=3, elem_id="modelmerger_discard_weights")

                    with InputAccordion(True, label="Save Metadata") as self.save_metadata:
                        with FormRow():
                            self.config_source = gr.CheckboxGroup(choices=["A", "B", "C"], value=["A", "B", "C"], label="Copy Metadata from", elem_id="modelmerger_config_method")
                            self.add_merge_recipe = gr.Checkbox(True, label="Include Merge Recipe", elem_id="modelmerger_add_recipe")

                        self.metadata_json = gr.TextArea(value=None, label="Metadata in JSON Format")
                        self.read_metadata = gr.Button("Preview Metadata from Models")

                        self.read_metadata.click(
                            fn=read_metadata,
                            inputs=[self.primary_model_name, self.secondary_model_name, self.tertiary_model_name],
                            outputs=[self.metadata_json],
                        )

                    with FormRow():
                        self.modelmerger_merge = gr.Button(elem_id="modelmerger_merge", value="Merge", variant="primary")

                with gr.Column(variant="panel", elem_id="modelmerger_results_container"):
                    with gr.Group(elem_id="modelmerger_results_panel"):
                        self.modelmerger_result = gr.HTML(value="Pending...", elem_id="modelmerger_result")

                for comp in (
                    self.primary_model_name,
                    self.secondary_model_name,
                    self.tertiary_model_name,
                    self.interp_method,
                    self.interp_amount,
                    self.custom_name,
                    self.discard_weights,
                    self.save_metadata,
                    self.config_source,
                    self.add_merge_recipe,
                    self.metadata_json,
                    self.read_metadata,
                    self.modelmerger_merge,
                    self.modelmerger_result,
                ):
                    comp.do_not_save_to_config = True

        self.blocks = modelmerger_interface

    def setup_ui(self, dummy_component, sd_model_checkpoint_component):
        self.modelmerger_merge.click(
            fn=lambda: "",
            outputs=[self.modelmerger_result],
            queue=False,
            show_progress=False,
        ).then(
            fn=call_queue.wrap_gradio_gpu_call(modelmerger, extra_outputs=lambda: [gr.skip() for _ in range(4)]),
            js="modelmerger",
            inputs=[
                dummy_component,
                self.primary_model_name,
                self.secondary_model_name,
                self.tertiary_model_name,
                self.interp_method,
                self.interp_amount,
                self.custom_name,
                self.discard_weights,
                self.save_metadata,
                self.config_source,
                self.add_merge_recipe,
            ],
            outputs=[
                self.primary_model_name,
                self.secondary_model_name,
                self.tertiary_model_name,
                sd_model_checkpoint_component,
                self.modelmerger_result,
            ],
        )
