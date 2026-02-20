import gradio as gr
import torch

from modules import scripts, sd_models
from modules.infotext_utils import PasteField
from modules.shared import opts
from modules.ui_common import create_refresh_button
from modules.ui_components import InputAccordion


class ScriptRefiner(scripts.ScriptBuiltinUI):
    section = "accordions"
    create_group = False
    ckpts = []

    def title(self):
        return "Refiner"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if opts.show_refiner else None

    @classmethod
    def refresh_checkpoints(cls):
        from modules_forge.main_entry import refresh_models

        ckpt_list, _ = refresh_models()
        cls.ckpts = ["None"] + ckpt_list

    def ui(self, is_img2img):
        self.refresh_checkpoints()
        with InputAccordion(False, label="Refiner", elem_id=self.elem_id("enable")) as enable_refiner:
            with gr.Row():
                refiner_checkpoint = gr.Dropdown(value="None", label="Checkpoint", info="(use model of same architecture)", elem_id=self.elem_id("checkpoint"), choices=self.ckpts)
                create_refresh_button(refiner_checkpoint, self.refresh_checkpoints, lambda: {"choices": self.ckpts}, self.elem_id("checkpoint_refresh"))
                refiner_switch_at = gr.Slider(value=0.875, label="Switch at", info="(in sigmas)", minimum=0.1, maximum=1.0, step=0.025, elem_id=self.elem_id("switch_at"), tooltip="Wan 2.2 T2V: 0.875 ; Wan 2.2 I2V: 0.9")

        def lookup_checkpoint(title):
            info = sd_models.get_closet_checkpoint_match(title)
            return None if info is None else info.short_title

        self.infotext_fields = [
            PasteField(enable_refiner, lambda d: "Refiner" in d),
            PasteField(refiner_checkpoint, lambda d: lookup_checkpoint(d.get("Refiner")), api="refiner_checkpoint"),
            PasteField(refiner_switch_at, "Refiner switch at", api="refiner_switch_at"),
        ]

        return enable_refiner, refiner_checkpoint, refiner_switch_at

    def setup(self, p, enable_refiner, refiner_checkpoint, refiner_switch_at):
        # the actual implementation is in sd_samplers_common.py apply_refiner()
        if not enable_refiner or refiner_checkpoint in (None, "", "None"):
            p.refiner_checkpoint = None
            p.refiner_switch_at = None
        else:
            p.refiner_checkpoint = refiner_checkpoint
            p.refiner_switch_at = refiner_switch_at

    @torch.inference_mode()
    def postprocess(self, *args, **kwargs):
        from modules import sd_samplers_common

        if sd_samplers_common.ORIGINAL_CHECKPOINT is None:
            return

        import huggingface_guess

        from backend.loader import preprocess_state_dict
        from backend.state_dict import load_state_dict, try_filter_state_dict
        from backend.utils import load_torch_file
        from modules_forge.main_entry import logger

        model = sd_models.model_data.get_sd_model().forge_objects.unet.model.diffusion_model

        sd = load_torch_file(sd_samplers_common.ORIGINAL_CHECKPOINT)
        sd = preprocess_state_dict(sd)

        guess = huggingface_guess.guess(sd)

        sd = try_filter_state_dict(sd, guess.unet_key_prefix)

        logger.info("Restoring state_dict...")
        sd_samplers_common.ORIGINAL_CHECKPOINT = None
        load_state_dict(model, sd)
