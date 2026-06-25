import os
from functools import partial

import gradio as gr

from modules import call_queue, errors, extras, sd_models, sd_vae, shared
from modules.ui_common import create_refresh_button
from modules.ui_components import FormRow


def update_interp_description(value):
    interp_description_css = "<p style='margin-bottom: 2.5em'>{}</p>"
    interp_descriptions = {"No interpolation": interp_description_css.format("No interpolation will be used. Requires one model; A. Allows for format conversion and VAE baking."), "Weighted sum": interp_description_css.format("A weighted sum will be used for interpolation. Requires two models; A and B. The result is calculated as A * (1 - M) + B * M"), "Add difference": interp_description_css.format("The difference between the last two models will be added to the first. Requires three models; A, B and C. The result is calculated as A + (B - C) * M")}
    return interp_descriptions[value]


def modelmerger(*args):
    try:
        results = extras.run_modelmerger(*args)
    except Exception as e:
        errors.report("Error loading/saving model file", exc_info=True)
        sd_models.list_models()  # to remove the potentially missing models from the list
        return [*[gr.update(choices=sd_models.checkpoint_tiles()) for _ in range(4)], f"Error merging checkpoints: {e}"]
    return results


class UiCheckpointMerger:
    def __init__(self):
        with gr.Blocks(analytics_enabled=False) as modelmerger_interface:
            with gr.Accordion(label="Save Current Model (only supports SD1 / SDXL / Flux)", open=False):
                with gr.Row():
                    filename_forge = gr.Textbox(value="flux.safetensors", lines=1, max_lines=1, label="Filename", info='(will save in "Stable-diffusion" folder)', scale=4)
                    btn_save_unet_forge = gr.Button("Save UNet", scale=1)
                    btn_save_ckpt_forge = gr.Button("Save Checkpoint", scale=1)
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

            with gr.Row(equal_height=False):
                with gr.Column(variant="compact"):
                    self.interp_description = gr.HTML(value=update_interp_description("Weighted sum"), elem_id="modelmerger_interp_description")

                    with FormRow(elem_id="modelmerger_models"):
                        self.primary_model_name = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="modelmerger_primary_model_name", label="Primary model (A)")
                        create_refresh_button(self.primary_model_name, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_A")

                        self.secondary_model_name = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="modelmerger_secondary_model_name", label="Secondary model (B)")
                        create_refresh_button(self.secondary_model_name, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_B")

                        self.tertiary_model_name = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="modelmerger_tertiary_model_name", label="Tertiary model (C)")
                        create_refresh_button(self.tertiary_model_name, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_C")

                    self.custom_name = gr.Textbox(label="Custom Name (Optional)", elem_id="modelmerger_custom_name")
                    self.interp_amount = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label="Multiplier (M) - set to 0 to get model A", value=0.3, elem_id="modelmerger_interp_amount")
                    self.interp_method = gr.Radio(choices=["No interpolation", "Weighted sum", "Add difference"], value="Weighted sum", label="Interpolation Method", elem_id="modelmerger_interp_method")
                    self.interp_method.change(fn=update_interp_description, inputs=[self.interp_method], outputs=[self.interp_description])

                    with FormRow():
                        self.checkpoint_format = gr.Radio(choices=["ckpt", "safetensors"], value="safetensors", label="Checkpoint format", elem_id="modelmerger_checkpoint_format")
                        self.save_as_half = gr.Checkbox(value=False, label="Save as float16", elem_id="modelmerger_save_as_half")

                    with FormRow():
                        with gr.Column():
                            self.config_source = gr.Radio(choices=["A, B or C", "B", "C", "Don't"], value="A, B or C", label="Copy config from", type="index", elem_id="modelmerger_config_method")

                        with gr.Column():
                            with FormRow():
                                self.bake_in_vae = gr.Dropdown(choices=["None"] + list(sd_vae.vae_dict), value="None", label="Bake in VAE", elem_id="modelmerger_bake_in_vae")
                                create_refresh_button(self.bake_in_vae, sd_vae.refresh_vae_list, lambda: {"choices": ["None"] + list(sd_vae.vae_dict)}, "modelmerger_refresh_bake_in_vae")

                    with FormRow():
                        self.discard_weights = gr.Textbox(value="", label="Discard weights with matching name", elem_id="modelmerger_discard_weights")

                    with gr.Accordion("Metadata", open=False) as metadata_editor:
                        with FormRow():
                            self.save_metadata = gr.Checkbox(value=True, label="Save metadata", elem_id="modelmerger_save_metadata")
                            self.add_merge_recipe = gr.Checkbox(value=True, label="Add merge recipe metadata", elem_id="modelmerger_add_recipe")
                            self.copy_metadata_fields = gr.Checkbox(value=True, label="Copy metadata from merged models", elem_id="modelmerger_copy_metadata")

                        self.metadata_json = gr.TextArea("{}", label="Metadata in JSON format")
                        self.read_metadata = gr.Button("Read metadata from selected checkpoints")

                    with FormRow():
                        self.modelmerger_merge = gr.Button(elem_id="modelmerger_merge", value="Merge", variant="primary")

                with gr.Column(variant="compact", elem_id="modelmerger_results_container"):
                    with gr.Group(elem_id="modelmerger_results_panel"):
                        self.modelmerger_result = gr.HTML(elem_id="modelmerger_result", show_label=False)

        self.metadata_editor = metadata_editor
        self.blocks = modelmerger_interface

    def setup_ui(self, dummy_component, sd_model_checkpoint_component):
        self.checkpoint_format.change(lambda fmt: gr.update(visible=fmt == "safetensors"), inputs=[self.checkpoint_format], outputs=[self.metadata_editor], show_progress=False)

        self.read_metadata.click(extras.read_metadata, inputs=[self.primary_model_name, self.secondary_model_name, self.tertiary_model_name], outputs=[self.metadata_json])

        self.modelmerger_merge.click(fn=lambda: "", inputs=[], outputs=[self.modelmerger_result])
        self.modelmerger_merge.click(
            fn=call_queue.wrap_gradio_gpu_call(modelmerger, extra_outputs=lambda: [gr.skip() for _ in range(4)]),
            _js="modelmerger",
            inputs=[
                dummy_component,
                self.primary_model_name,
                self.secondary_model_name,
                self.tertiary_model_name,
                self.interp_method,
                self.interp_amount,
                self.save_as_half,
                self.custom_name,
                self.checkpoint_format,
                self.config_source,
                self.bake_in_vae,
                self.discard_weights,
                self.save_metadata,
                self.add_merge_recipe,
                self.copy_metadata_fields,
                self.metadata_json,
            ],
            outputs=[
                self.primary_model_name,
                self.secondary_model_name,
                self.tertiary_model_name,
                sd_model_checkpoint_component,
                self.modelmerger_result,
            ],
        )

        # Required as a workaround for change() event not triggering when loading values from ui-config.json
        self.interp_description.value = update_interp_description(self.interp_method.value)
