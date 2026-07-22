import gradio as gr
import torch

from modules import scripts, sd_models, shared
from modules.infotext_utils import PasteField
from modules.ui_common import create_refresh_button
from modules.ui_components import InputAccordion


class ScriptRefiner(scripts.ScriptBuiltinUI):
    section = "accordions"
    create_group = False
    ckpts = []

    def title(self):
        return "Refiner"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if shared.opts.show_refiner else None

    @classmethod
    def refresh_checkpoints(cls):
        from modules_forge.main_entry import refresh_models

        ckpt_list, _ = refresh_models()
        cls.ckpts = ["None"] + ckpt_list

    def ui(self, is_img2img):
        self.refresh_checkpoints()
        with InputAccordion(False, label="Refiner", elem_id=self.elem_id("enable")) as enable_refiner:
            with gr.Row():
                refiner_checkpoint = gr.Dropdown(
                    value="None",
                    label="Checkpoint",
                    info="(use model of same architecture and quantization)" if shared.opts.refiner_fast_sd else "(use model of same architecture)",
                    elem_id=self.elem_id("checkpoint"),
                    choices=self.ckpts,
                )
                create_refresh_button(refiner_checkpoint, self.refresh_checkpoints, lambda: {"choices": self.ckpts}, self.elem_id("checkpoint_refresh"))
                refiner_switch_at = gr.Slider(
                    value=0.875,
                    label="Switch at",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.025,
                    elem_id=self.elem_id("switch_at"),
                    **(
                        {"info": "(in steps)", "tooltip": "percentage of total steps"}
                        if shared.opts.refiner_use_steps
                        else {
                            "info": "(in sigmas)",
                            "tooltip": "Wan 2.2 I2V: 0.9" if is_img2img else "Wan 2.2 T2V: 0.875",
                        }
                    ),
                )

            with gr.Accordion(label="LoRA Replacements", open=True):
                gr.Markdown(
                    """
Use this setting to load different LoRAs between the normal pass and the refiner pass
- Each line can contain one rule to process the LoRA names
- **Available Rules:**
    - `AAA=BBB` : the `AAA` within a LoRA would be replaced with `BBB` during the refiner pass
    - `-CCC` : any LoRA that contains `CCC` would be removed during the refiner pass
    - `DDD+EEE` : the LoRA named `DDD` would be added with strength `EEE` during the refiner pass
- **Example:**
    - The default value would replace any occurrence of "`high_noise`" with "`low_noise`"
                    """,
                )
                replacements = gr.Textbox(
                    value=lambda: shared.opts.refiner_lora_replacement,
                    lines=4,
                    max_lines=12,
                    placeholder="\n".join(["high_noise=low_noise", "high-noise=low-noise", "-lora_to_remove", "lora_to_add+1.0"]),
                    show_label=False,
                )
                replacements.do_not_save_to_config = True

        def lookup_checkpoint(title):
            info = sd_models.get_closet_checkpoint_match(title)
            return None if info is None else info.short_title

        self.infotext_fields = [
            PasteField(enable_refiner, lambda d: "Refiner" in d),
            PasteField(refiner_checkpoint, lambda d: lookup_checkpoint(d.get("Refiner")), api="refiner_checkpoint"),
            PasteField(refiner_switch_at, "Refiner switch at", api="refiner_switch_at"),
        ]

        return [enable_refiner, refiner_checkpoint, refiner_switch_at, replacements]

    def setup(self, p, enable_refiner: bool, refiner_checkpoint: str, refiner_switch_at: float, replacements: str):
        if not enable_refiner or refiner_checkpoint in (None, "", "None"):
            p.refiner_checkpoint = None
            p.refiner_switch_at = None
            return

        # actual implementation is in sd_samplers_common.py / apply_refiner()
        p.refiner_checkpoint = refiner_checkpoint
        p.refiner_switch_at = refiner_switch_at
        shared.opts.set("refiner_lora_replacement", replacements.strip())
        shared.opts.save(shared.config_filename)

    @torch.inference_mode()
    def postprocess(self, *args, **kwargs):
        from modules import sd_samplers_common, shared

        if sd_samplers_common.ORIGINAL_CHECKPOINT is None:
            return

        sd_model = shared.sd_model

        import huggingface_guess

        from backend.loader import preprocess_state_dict
        from backend.memory_management import LoadedModel, current_loaded_models
        from backend.patcher.unet import UnetPatcher
        from backend.state_dict import (
            convert_quantization,
            load_state_dict,
            try_filter_state_dict,
        )
        from backend.utils import load_torch_file
        from modules_forge.main_entry import logger

        for i, loaded_models in enumerate(current_loaded_models):
            if isinstance(loaded_models.model, UnetPatcher):
                idx = i
                break

        mdl: LoadedModel = current_loaded_models.pop(idx)
        mdl.model_unload()
        del mdl

        model = sd_model.forge_objects.unet.model.diffusion_model

        sd, metadata = load_torch_file(sd_samplers_common.ORIGINAL_CHECKPOINT, return_metadata=True)
        sd, metadata = convert_quantization(sd, metadata)
        sd = preprocess_state_dict(sd)

        guess = huggingface_guess.guess(sd)

        sd = try_filter_state_dict(sd, guess.unet_key_prefix)

        logger.info("Restoring state_dict...")
        load_state_dict(model, sd, ignore_start="llm")

        if sd_samplers_common.ORIGINAL_CHECKPOINT.lower().endswith(".gguf"):

            from backend.memory_management import bake_gguf_model

            sd_model.forge_objects.unet.model.gguf_baked = False
            sd_model.forge_objects.unet.model = bake_gguf_model(sd_model.forge_objects.unet.model)

        sd_samplers_common.ORIGINAL_CHECKPOINT = None


shared.options_templates.update(
    shared.options_section(
        (None, "Refiner"),
        {
            "refiner_lora_replacement": shared.OptionInfo(default="high_noise=low_noise\nhigh-noise=low-noise"),
        },
    )
)
