import gradio as gr

from modules import ui_extra_networks_user_metadata
from modules.ui_common import create_refresh_button
from modules_forge import main_entry, presets


class CheckpointUserMetadataEditor(ui_extra_networks_user_metadata.UserMetadataEditor):
    def __init__(self, ui, tabname, page):
        super().__init__(ui, tabname, page)

        self.select_vae = None
        self.sd_version = "Unknown"

    def save_user_metadata(self, name: str, desc: str, notes: str, vae: list[str], sd_version: str):
        user_metadata = self.get_user_metadata(name)
        user_metadata["description"] = desc
        user_metadata["notes"] = notes
        user_metadata["vae_te"] = vae
        user_metadata["sd_arch"] = sd_version

        self.write_user_metadata(name, user_metadata)

    def put_values_into_components(self, name: str):
        user_metadata = self.get_user_metadata(name)
        values = super().put_values_into_components(name)
        vae = user_metadata.get("vae_te", None) or []
        version = user_metadata.get("sd_arch", None) or "Unknown"

        return [*values, vae, version]

    def create_editor(self):
        self.create_default_editor_elems()

        self.sd_version = gr.Radio(choices=["Unknown"] + presets.PresetArch.choices(), value="Unknown", label="Model Arch")

        with gr.Row():
            self.select_vae = gr.Dropdown(choices=list(main_entry.module_list.keys()), value=None, label="Preferred VAE / Text Encoder", elem_id="checkpoint_edit_user_metadata_preferred_vae", multiselect=True)
            create_refresh_button(
                self.select_vae,
                main_entry.refresh_models,
                lambda: gr.update(choices=list(main_entry.module_list.keys())),
                "checkpoint_edit_user_metadata_refresh_modules",
            )

        self.edit_notes = gr.TextArea(label="Notes", lines=1.05)

        self.create_default_buttons()

        viewed_components = [
            self.edit_name,
            self.edit_description,
            self.html_filedata,
            self.html_preview,
            self.edit_notes,
            self.select_vae,
            self.sd_version,
        ]

        self.button_edit.click(
            fn=self.put_values_into_components,
            inputs=[self.edit_name_input],
            outputs=viewed_components,
        ).success(
            fn=lambda: gr.update(visible=True),
            outputs=[self.box],
        )

        edited_components = [
            self.edit_description,
            self.edit_notes,
            self.select_vae,
            self.sd_version,
        ]

        self.setup_save_handler(self.button_save, self.save_user_metadata, edited_components)
