# https://github.com/Anzhc/Anima-Mod-Guidance-ComfyUI-Node/blob/main/nodes.py

import os.path

import gradio as gr
import torch
from lib_modulation.adapter import resolve_adapter_path, validate_adapter_for_model
from lib_modulation.anima_patch import register_modulation_wrapper

from modules import scripts
from modules.processing import StableDiffusionProcessing
from modules.prompt_parser import SdConditioning
from modules.ui_components import InputAccordion
from modules_forge.main_entry import module_list


def _extract_pooled_output(pooled, input_name):
    if pooled is None:
        raise RuntimeError(f"{input_name} is missing pooled_output. Provide CLIP-based conditioning with pooled embeddings.")
    if not torch.is_tensor(pooled):
        raise RuntimeError(f"{input_name}.pooled_output must be a tensor, got {type(pooled)}.")

    if pooled.ndim == 1:
        pooled = pooled.unsqueeze(0)
    elif pooled.ndim != 2:
        raise RuntimeError(f"{input_name}.pooled_output must have rank 1 or 2, got rank {pooled.ndim}.")

    if pooled.shape[0] < 1:
        raise RuntimeError(f"{input_name}.pooled_output has invalid empty batch.")

    return pooled[:1].detach().float().cpu().contiguous()


def _validate_anima_model(model):
    base_model = getattr(model, "model", None)
    if base_model is None:
        raise RuntimeError("Invalid MODEL input: missing internal model object.")

    diffusion_model = getattr(base_model, "diffusion_model", None)
    if diffusion_model is None:
        raise RuntimeError("Invalid Anima model: diffusion_model is missing.")
    return diffusion_model


class AnimaModGuidance:

    @staticmethod
    def patch(
        model,
        clip_base_conditioning,
        clip_positive_conditioning,
        clip_negative_conditioning,
        w,
        start_layer,
        end_layer,
    ):
        diffusion_model = _validate_anima_model(model)

        pooled_base = _extract_pooled_output(clip_base_conditioning, "clip_base_conditioning")
        pooled_positive = _extract_pooled_output(clip_positive_conditioning, "clip_positive_conditioning")
        pooled_negative = _extract_pooled_output(clip_negative_conditioning, "clip_negative_conditioning")

        resolved_adapter_path = resolve_adapter_path()
        adapter_meta = validate_adapter_for_model(resolved_adapter_path, diffusion_model)

        pooled_dim = int(pooled_base.shape[1])
        if pooled_dim != adapter_meta["pooled_dim"]:
            raise RuntimeError("clip_base_conditioning pooled dim mismatch: " f"got {pooled_dim}, expected {adapter_meta['pooled_dim']}.")
        if int(pooled_positive.shape[1]) != adapter_meta["pooled_dim"]:
            raise RuntimeError("clip_positive_conditioning pooled dim mismatch: " f"got {int(pooled_positive.shape[1])}, expected {adapter_meta['pooled_dim']}.")
        if int(pooled_negative.shape[1]) != adapter_meta["pooled_dim"]:
            raise RuntimeError("clip_negative_conditioning pooled dim mismatch: " f"got {int(pooled_negative.shape[1])}, expected {adapter_meta['pooled_dim']}.")

        patched_model = model.clone()
        register_modulation_wrapper(
            patched_model,
            adapter_path=resolved_adapter_path,
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

    from backend.args import dynamic_args

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

        unet = p.sd_model.forge_objects.unet

        clip = load_clip(module_list[clip])

        _, _base = clip(SdConditioning([p.prompt], is_negative_prompt=False, width=p.width, height=p.height))
        _, _pos = clip(SdConditioning([pos], is_negative_prompt=False, width=p.width, height=p.height))
        _, _neg = clip(SdConditioning([neg], is_negative_prompt=True, width=p.width, height=p.height))

        _unet = AnimaModGuidance.patch(unet, _base, _pos, _neg, w, start, end)

        p.sd_model.forge_objects.unet = _unet
