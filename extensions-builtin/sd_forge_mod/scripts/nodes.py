# https://github.com/Anzhc/Anima-Mod-Guidance-ComfyUI-Node/blob/main/nodes.py

import os.path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.patcher.unet import UnetPatcher
import gradio as gr
import torch
from lib_modulation.adapter import resolve_adapter_path
from lib_modulation.anima_patch import register_modulation_wrapper, unpatch

from modules import scripts
from modules.processing import StableDiffusionProcessing
from modules.prompt_parser import SdConditioning
from modules.ui_components import InputAccordion
from modules_forge.main_entry import module_list


def _extract_pooled_output(pooled: torch.Tensor) -> torch.Tensor:
    if pooled.ndim == 1:
        pooled = pooled.unsqueeze(0)
    return pooled[:1].contiguous()


class AnimaModGuidance:

    @staticmethod
    def patch(
        unet: "UnetPatcher",
        clip_base_conditioning: torch.Tensor,
        clip_positive_conditioning: torch.Tensor,
        clip_negative_conditioning: torch.Tensor,
        w: float,
        start_layer: int,
        end_layer: int,
    ):
        pooled_base = _extract_pooled_output(clip_base_conditioning)
        pooled_positive = _extract_pooled_output(clip_positive_conditioning)
        pooled_negative = _extract_pooled_output(clip_negative_conditioning)

        patched_model = unet.clone()

        register_modulation_wrapper(
            patched_model,
            adapter_path=resolve_adapter_path(),
            clip_base_pooled=pooled_base,
            clip_positive_pooled=pooled_positive,
            clip_negative_pooled=pooled_negative,
            w=w,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        return patched_model


def preprocess_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(k.startswith("transformer.") for k in sd.keys()):
        sd = {f"transformer.{k}": v for k, v in sd.items()}

    return sd


def load_clip(path: str):
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
    from transformers.modeling_utils import no_init_weights

    from backend import memory_management
    from backend.args import dynamic_args
    from backend.loader import HF
    from backend.nn.clip import IntegratedCLIP
    from backend.operations import using_forge_operations
    from backend.state_dict import load_state_dict
    from backend.text_processing.classic_engine import ClassicTextProcessingEngine
    from backend.utils import load_torch_file

    tokenizer_path = os.path.join(HF, "stabilityai", "stable-diffusion-xl-base-1.0", "tokenizer")
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    tokenizer._eventual_warn_about_too_long_sequence = lambda *args, **kwargs: None

    config_path = os.path.join(HF, "stabilityai", "stable-diffusion-xl-base-1.0", "text_encoder")
    config = CLIPTextConfig.from_pretrained(config_path)
    to_args = dict(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype())

    with no_init_weights():
        with using_forge_operations(**to_args, manual_cast_enabled=True):
            text_encoder = IntegratedCLIP(CLIPTextModel, config, add_text_projection=True).to(**to_args)

    sd = load_torch_file(path)
    sd = preprocess_state_dict(sd)
    load_state_dict(text_encoder, sd, ignore_errors=["transformer.text_projection.weight", "transformer.text_model.embeddings.position_ids", "logit_scale", "transformer.logit_scale"])

    return ClassicTextProcessingEngine(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        embedding_dir=dynamic_args["embedding_dir"],
        embedding_key="clip_l",
        embedding_expected_shape=2048,
        text_projection=False,
        minimal_clip_skip=2,
        clip_skip=2,
        return_pooled=True,
        final_layer_norm=False,
    )


class ModulationGuidanceForForge(scripts.ScriptBuiltinUI):
    sorting_priority = 260209268

    def title(self):
        return "Modulation Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enable:
            modules = list(module_list.keys())
            clip = gr.Dropdown(choices=modules, value=next(iter(modules), None))
            pos = gr.Textbox(label="clip_positive_conditioning")
            neg = gr.Textbox(label="clip_negative_conditioning")
            w = gr.Slider(label="w", value=3.0, minimum=-20.0, maximum=20.0, step=0.01)
            start = gr.Slider(label="start_layer", value=0, minimum=0, maximum=1024, step=1)
            end = gr.Slider(label="end_layer", value=-1, minimum=-1, maximum=1024, step=1)

        for comp in (comps := (enable, clip, pos, neg, w, start, end)):
            comp.do_not_save_to_config = True

        return comps

    def process_before_every_sampling(self, p: StableDiffusionProcessing, enable: bool, clip: str, pos: str, neg: str, w: float, start: int, end: int, **kwargs):
        if not enable:
            return

        clip_l = load_clip(module_list[clip])

        _, _base = clip_l(SdConditioning([p.prompt], is_negative_prompt=False, width=p.width, height=p.height))
        _, _pos = clip_l(SdConditioning([pos], is_negative_prompt=False, width=p.width, height=p.height))
        _, _neg = clip_l(SdConditioning([neg], is_negative_prompt=True, width=p.width, height=p.height))

        unet = p.sd_model.forge_objects.unet

        _unet = AnimaModGuidance.patch(unet, _base, _pos, _neg, w, start, end)

        p.sd_model.forge_objects.unet = _unet

    def postprocess(self, *args, **kwargs):
        unpatch()
