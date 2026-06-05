"""Ideogram 4.0 txt2img UI (spec §3).

An AlwaysVisible built-in script that adds the Ideogram 4.0 panel to txt2img:
a structured JSON-caption builder, sampler-preset / advanced (mu, std) controls,
transparent-background and resolution presets, and a JSON preview + live
CaptionVerifier validation.

The panel is hidden by default; ``modules_forge.main_entry`` shows it (and hides
the sampler/scheduler/VAE) when the model-type preset is set to ``ideogram4``.

On generate, ``before_process`` assembles the ordered JSON caption (or uses the
main Prompt box in plain-text mode), runs the verifier, and stashes everything on
the processing object for ``modules.ideogram4.processing.process_images_ideogram4``.
"""

import html
import re

import gradio as gr

from modules import scripts
from modules.ideogram4 import ui_state
from modules.ideogram4.caption import (
    MEDIUM_CHOICES,
    PHOTO_MEDIUM,
    assemble_caption,
    dumps,
)
from modules.ideogram4.caption_verifier import CaptionVerifier
from modules.ideogram4.sampler_configs import DEFAULT_PRESET, PRESETS, get_preset

MAX_ELEMENTS = 5

# spec §3.3 resolution presets (label -> (width, height)); "Custom" leaves sliders alone
RESOLUTION_PRESETS = {
    "Custom": None,
    "1024 × 1024 (1:1)": (1024, 1024),
    "1536 × 1024 (3:2)": (1536, 1024),
    "1024 × 1536 (2:3)": (1024, 1536),
    "1920 × 1088 (16:9)": (1920, 1088),
    "2048 × 768 (8:3)": (2048, 768),
    "1024 × 1792 (4:7)": (1024, 1792),
    "1600 × 400 (4:1)": (1600, 400),
}


def _split_palette(value) -> list[str]:
    if not value:
        return []
    return [p for p in re.split(r"[,\s]+", str(value).strip()) if p]


def _parse_bbox(value):
    if not value:
        return None
    parts = [p for p in re.split(r"[,\s]+", str(value).strip()) if p]
    return parts if len(parts) == 4 else None


def _to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _validation_html(warnings) -> str:
    if not warnings:
        return "<div class='ideogram4-valid' style='color:#2e7d32'>✓ Caption looks valid.</div>"
    items = "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings)
    return (
        "<div class='ideogram4-warn' style='color:#b26a00'>"
        f"⚠ {len(warnings)} warning(s) — generation will still proceed:"
        f"<ul style='margin:.25em 0 0 1em'>{items}</ul></div>"
    )


class ScriptIdeogram4(scripts.ScriptBuiltinUI):
    # section left as the default None so the panel renders in the generic "scripts"
    # area (ui.py setup_ui -> setup_ui_for_section(None)); visibility is then driven
    # by the model-type preset in modules_forge.main_entry._wire_ideogram4.
    create_group = False

    def __init__(self):
        self.field_order: list[str] = []
        self.field_components: list = []

    def title(self):
        return "Ideogram 4.0"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # ---- ui -----------------------------------------------------------------
    def _reg(self, name, component):
        """Register a value-bearing component so ui() return order == before_process arg order."""
        self.field_order.append(name)
        self.field_components.append(component)
        return component

    def ui(self, is_img2img):
        self.field_order = []
        self.field_components = []

        if is_img2img:
            # Ideogram 4.0 is txt2img-only in this release; contribute no controls.
            return []

        eid = self.elem_id

        with gr.Group(visible=False, elem_id=eid("ideogram4_panel")) as panel:
            gr.Markdown("### Ideogram 4.0 — structured JSON caption")

            plain_text = self._reg("plain_text", gr.Checkbox(
                label="Plain-text mode (use the main Prompt box instead of the builder)",
                value=False, elem_id=eid("ideogram4_plain_text")))

            with gr.Group() as builder_group:
                high_level_description = self._reg("high_level_description", gr.Textbox(
                    label="High-level description (recommended)", lines=2,
                    placeholder="A medium-shot photograph of a barista pouring latte art in a cozy cafe.",
                    elem_id=eid("ideogram4_hld")))

                with gr.Accordion("Style", open=True):
                    medium = self._reg("medium", gr.Dropdown(
                        label="medium", choices=MEDIUM_CHOICES, value="photograph",
                        elem_id=eid("ideogram4_medium")))
                    aesthetics = self._reg("aesthetics", gr.Textbox(
                        label="aesthetics", placeholder="moody, cinematic, desaturated",
                        elem_id=eid("ideogram4_aesthetics")))
                    lighting = self._reg("lighting", gr.Textbox(
                        label="lighting", placeholder="golden hour, rim light",
                        elem_id=eid("ideogram4_lighting")))
                    photo = self._reg("photo", gr.Textbox(
                        label="photo (camera / lens)", placeholder="35mm, f/1.4, bokeh",
                        visible=True, elem_id=eid("ideogram4_photo")))
                    art_style = self._reg("art_style", gr.Textbox(
                        label="art_style", placeholder="flat vector illustration",
                        visible=False, elem_id=eid("ideogram4_art_style")))
                    style_palette = self._reg("style_palette", gr.Textbox(
                        label="color_palette (≤16, e.g. #1B1B2F, #FF6B35)",
                        placeholder="#FF6B35, #F7C59F, #004E89",
                        elem_id=eid("ideogram4_style_palette")))

                with gr.Accordion("Composition", open=True):
                    background = self._reg("background", gr.Textbox(
                        label="background (required)", lines=2,
                        placeholder="A calm ocean stretching to a low horizon...",
                        elem_id=eid("ideogram4_background")))
                    element_count = self._reg("element_count", gr.Slider(
                        label="Number of elements", minimum=0, maximum=MAX_ELEMENTS,
                        step=1, value=1, elem_id=eid("ideogram4_element_count")))

                    element_groups = []
                    for i in range(MAX_ELEMENTS):
                        with gr.Group(visible=(i == 0)) as el_group:
                            gr.Markdown(f"**Element {i + 1}**")
                            el_type = self._reg(f"el{i}_type", gr.Radio(
                                label="type", choices=["obj", "text"], value="obj",
                                elem_id=eid(f"ideogram4_el{i}_type")))
                            el_text = self._reg(f"el{i}_text", gr.Textbox(
                                label="text (rendered literally)", visible=False,
                                elem_id=eid(f"ideogram4_el{i}_text")))
                            el_desc = self._reg(f"el{i}_desc", gr.Textbox(
                                label="desc", lines=2, elem_id=eid(f"ideogram4_el{i}_desc")))
                            el_bbox = self._reg(f"el{i}_bbox", gr.Textbox(
                                label="bbox y_min, x_min, y_max, x_max (0–1000, optional)",
                                placeholder="200, 300, 800, 900",
                                elem_id=eid(f"ideogram4_el{i}_bbox")))
                            el_palette = self._reg(f"el{i}_palette", gr.Textbox(
                                label="color_palette (≤5, optional)",
                                elem_id=eid(f"ideogram4_el{i}_palette")))
                        element_groups.append(el_group)

                        # type radio toggles the literal-text field for this slot
                        el_type.change(
                            fn=lambda t: gr.update(visible=(t == "text")),
                            inputs=[el_type], outputs=[el_text],
                            queue=False, show_progress=False)

            with gr.Accordion("Inference", open=True):
                sampler_preset = self._reg("sampler_preset", gr.Dropdown(
                    label="Sampler Preset", choices=list(PRESETS.keys()), value=DEFAULT_PRESET,
                    elem_id=eid("ideogram4_preset")))
                transparent = self._reg("transparent", gr.Checkbox(
                    label="Transparent background", value=False,
                    elem_id=eid("ideogram4_transparent")))
                resolution_preset = self._reg("resolution_preset", gr.Dropdown(
                    label="Resolution Preset", choices=list(RESOLUTION_PRESETS.keys()),
                    value="Custom", elem_id=eid("ideogram4_resolution")))
                with gr.Accordion("Advanced (mu / std)", open=False):
                    _default = get_preset(DEFAULT_PRESET)
                    mu = self._reg("mu", gr.Number(
                        label="mu (logit-normal mean)", value=_default.mu,
                        elem_id=eid("ideogram4_mu")))
                    std = self._reg("std", gr.Number(
                        label="std (logit-normal std)", value=_default.std,
                        elem_id=eid("ideogram4_std")))

            with gr.Accordion("JSON preview & validation", open=False):
                build_btn = gr.Button("Build / refresh JSON", elem_id=eid("ideogram4_build"))
                json_preview = gr.Textbox(
                    label="Caption JSON (preview)", lines=8, show_copy_button=True,
                    elem_id=eid("ideogram4_json"))
                validation_html = gr.HTML(elem_id=eid("ideogram4_validation"))

        # ---- intra-panel interactions ---------------------------------------
        medium.change(
            fn=lambda m: (gr.update(visible=(m == PHOTO_MEDIUM)), gr.update(visible=(m != PHOTO_MEDIUM))),
            inputs=[medium], outputs=[photo, art_style], queue=False, show_progress=False)

        plain_text.change(
            fn=lambda v: gr.update(visible=not v), inputs=[plain_text],
            outputs=[builder_group], queue=False, show_progress=False)

        element_count.change(
            fn=lambda n: [gr.update(visible=(i < int(n))) for i in range(MAX_ELEMENTS)],
            inputs=[element_count], outputs=element_groups, queue=False, show_progress=False)

        # Sampler Preset auto-fills mu / std (spec §3.3, §5 "プリセット連動")
        sampler_preset.change(
            fn=self._on_preset, inputs=[sampler_preset], outputs=[mu, std],
            queue=False, show_progress=False)

        build_btn.click(
            fn=self._build_preview, inputs=list(self.field_components),
            outputs=[json_preview, validation_html], queue=False, show_progress=False)

        # expose handles for main_entry to wire preset-driven visibility / sizes
        ui_state.group = panel
        ui_state.sampler_preset = sampler_preset
        ui_state.resolution_preset = resolution_preset
        ui_state.resolution_map = RESOLUTION_PRESETS

        self.infotext_fields = [
            (sampler_preset, "Ideogram preset"),
            (mu, "Ideogram mu"),
            (std, "Ideogram std"),
            (transparent, "Ideogram transparent"),
        ]

        return list(self.field_components)

    # ---- callbacks ----------------------------------------------------------
    @staticmethod
    def _on_preset(preset_name):
        preset = get_preset(preset_name)
        return gr.update(value=preset.mu), gr.update(value=preset.std)

    def _args_to_dict(self, args) -> dict:
        return dict(zip(self.field_order, args))

    def _build_data(self, d: dict) -> dict:
        style = {
            "medium": d.get("medium"),
            "aesthetics": d.get("aesthetics"),
            "lighting": d.get("lighting"),
            "photo": d.get("photo"),
            "art_style": d.get("art_style"),
            "color_palette": _split_palette(d.get("style_palette")),
        }
        try:
            n = int(d.get("element_count") or 0)
        except (TypeError, ValueError):
            n = 0
        elements = []
        for i in range(min(n, MAX_ELEMENTS)):
            elements.append({
                "type": d.get(f"el{i}_type") or "obj",
                "bbox": _parse_bbox(d.get(f"el{i}_bbox")),
                "text": d.get(f"el{i}_text"),
                "desc": d.get(f"el{i}_desc"),
                "color_palette": _split_palette(d.get(f"el{i}_palette")),
            })
        return {
            "high_level_description": d.get("high_level_description"),
            "style_description": style,
            "compositional_deconstruction": {
                "background": d.get("background"),
                "elements": elements,
            },
        }

    def _build_preview(self, *args):
        d = self._args_to_dict(args)
        if d.get("plain_text"):
            return (
                "(plain-text mode — the main Prompt box above is sent as the caption)",
                "<div>Plain-text mode: the JSON builder is bypassed.</div>",
            )
        caption = assemble_caption(self._build_data(d))
        warnings = CaptionVerifier().verify(caption)
        return dumps(caption), _validation_html(warnings)

    # ---- generation ---------------------------------------------------------
    def before_process(self, p, *args):
        from modules import shared
        from modules.processing import StableDiffusionProcessingTxt2Img

        if self.is_img2img or not args:
            return
        if getattr(shared.opts, "forge_preset", "") != "ideogram4":
            return
        if not isinstance(p, StableDiffusionProcessingTxt2Img):
            return

        d = self._args_to_dict(args)

        if d.get("plain_text"):
            caption = p.prompt if isinstance(p.prompt, str) else (p.prompt[0] if p.prompt else "")
            warnings = []
        else:
            built = assemble_caption(self._build_data(d))
            caption = dumps(built)
            warnings = CaptionVerifier().verify(built)

        preset = get_preset(d.get("sampler_preset") or DEFAULT_PRESET)
        mu = _to_float(d.get("mu"), preset.mu)
        std = _to_float(d.get("std"), preset.std)

        p.ideogram4_enabled = True
        p.ideogram4_caption = caption
        p.ideogram4_warnings = warnings
        p.ideogram4_params = {
            "preset": preset.name,
            "steps": preset.steps,
            "guidance_scale": preset.base_guidance,
            "guidance_schedule": preset.guidance_schedule,
            "mu": mu,
            "std": std,
            "transparent": bool(d.get("transparent")),
            "model_path": getattr(shared.opts, "ideogram4_model_path", ""),
            "quantization": getattr(shared.opts, "ideogram4_quantization", "nf4"),
        }

        # keep p consistent with what the pipeline will actually use
        p.steps = preset.steps
        p.cfg_scale = preset.base_guidance
