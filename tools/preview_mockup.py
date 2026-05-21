"""
Standalone visual mockup of the quicksettings row + new language selector.

This is NOT the full WebUI. It loads no models and runs no diffusion. It only
renders the layout so you can verify where the Language dropdown lands and how
it picks up the it_IT translations.

Run via the Claude Preview MCP (`preview_start mockup`) or directly:

    python tools/preview_mockup.py

then open http://127.0.0.1:7861
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr so flag emojis in language labels (🇮🇹, 🇯🇵, ...)
# do not crash Gradio's startup logging on the default Windows cp1252 console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import gradio as gr

REPO = Path(__file__).resolve().parent.parent
LOC_DIR = REPO / "localizations"
STYLE_CSS = (REPO / "style.css").read_text(encoding="utf-8") if (REPO / "style.css").is_file() else ""

# Include the same flag-injector script Forge would load from javascript/
_FLAG_JS_PATH = REPO / "javascript" / "forge_language_flags.js"
FLAG_JS = _FLAG_JS_PATH.read_text(encoding="utf-8") if _FLAG_JS_PATH.is_file() else ""

EXTRA_CSS = """
body { background: #1f2937; color: #f3f4f6; }
.gradio-container { max-width: 100% !important; padding: 1rem !important; }
.preview-header { padding: 0.5em 1em; background: #111827; border-bottom: 1px solid #374151;
    font-family: ui-sans-serif, system-ui; color: #9ca3af; font-size: 0.85em; }
.preview-header b { color: #f3f4f6; }
"""


LANGUAGE_DISPLAY: dict[str, str] = {
    "None": "English",
    "it_IT": "Italiano",
    "es_ES": "Español",
    "fr_FR": "Français",
    "de_DE": "Deutsch",
    "zh_CN": "简体中文",
    "ja_JP": "日本語",
}

HIDDEN_LANGUAGES: set[str] = {"en_US"}


def available_languages() -> list[tuple[str, str]]:
    """List of (display_label, locale_code) tuples for the dropdown."""
    codes = ["None"]
    if LOC_DIR.is_dir():
        for p in sorted(LOC_DIR.glob("*.json")):
            if p.stem in HIDDEN_LANGUAGES:
                continue
            codes.append(p.stem)
    return [(LANGUAGE_DISPLAY.get(c, c), c) for c in codes]


def load_translations(name: str) -> dict[str, str]:
    if not name or name == "None":
        return {}
    path = LOC_DIR / f"{name}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def labels_for(name: str) -> dict[str, str]:
    """Return translated labels for the components in the mockup."""
    table = load_translations(name)
    keys = ["UI Preset", "Checkpoint", "VAE / Text Encoder", "Diffusion in Low Bits", "Language"]
    return {k: table.get(k, k) for k in keys}


def build_ui(initial_lang: str = "None") -> gr.Blocks:
    initial = labels_for(initial_lang)

    # Gradio 4 takes css in Blocks(); Gradio 6 takes css in launch().
    gradio_major = int(gr.__version__.split(".", 1)[0])
    blocks_kwargs: dict = {"title": "Forge Neo i18n - Preview"}
    if gradio_major < 6:
        blocks_kwargs["css"] = STYLE_CSS + EXTRA_CSS
    if FLAG_JS:
        # `head` is rendered into <head> as raw HTML, so a script tag here
        # actually executes (unlike the same script wrapped in gr.HTML).
        blocks_kwargs["head"] = f"<script>{FLAG_JS}</script>"

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(
            "<div class='preview-header'>"
            "<b>Mockup</b> · just the Forge Neo <code>#quicksettings</code> row · "
            "the <b>Language</b> dropdown is the new element (anchored to the far right)."
            "</div>"
        )

        with gr.Row(elem_id="quicksettings", variant="compact"):
            ui_preset = gr.Dropdown(
                label=initial["UI Preset"],
                value="sd",
                choices=["sd", "xl", "flux", "sd3", "lumina"],
                elem_id="forge_ui_preset",
            )
            ui_checkpoint = gr.Dropdown(
                label=initial["Checkpoint"],
                value="example-checkpoint.safetensors",
                choices=["example-checkpoint.safetensors", "another.safetensors"],
                elem_id="setting_sd_model_checkpoint",
                elem_classes=["model_selection"],
            )
            ui_vae = gr.Dropdown(
                label=initial["VAE / Text Encoder"],
                value=[],
                choices=["vae-ft-mse.safetensors", "clip_l.safetensors"],
                multiselect=True,
                elem_id="setting_sd_modules",
                elem_classes=["model_selection"],
            )
            gr.Button(value="↻", elem_id="forge_refresh_checkpoint")
            ui_dtype = gr.Dropdown(
                label=initial["Diffusion in Low Bits"],
                value="Automatic",
                choices=["Automatic", "float8-e4m3fn", "int8", "bnb-nf4"],
                elem_id="forge_ui_dtype",
            )
            ui_language = gr.Dropdown(
                label=initial["Language"],
                value=initial_lang,
                choices=available_languages(),
                elem_id="forge_ui_language",
                elem_classes=["language_selector"],
            )

        gr.Markdown(
            "The **Language** selector is pinned to the right via "
            "`#quicksettings > div#forge_ui_language { margin-left: auto }` in "
            "`style.css`. Changing the language below re-applies the translated "
            "labels in place (no reload — preview only)."
        )

        def relabel(name: str):
            t = labels_for(name)
            return (
                gr.update(label=t["UI Preset"]),
                gr.update(label=t["Checkpoint"]),
                gr.update(label=t["VAE / Text Encoder"]),
                gr.update(label=t["Diffusion in Low Bits"]),
                gr.update(label=t["Language"]),
            )

        ui_language.change(
            fn=relabel,
            inputs=[ui_language],
            outputs=[ui_preset, ui_checkpoint, ui_vae, ui_dtype, ui_language],
            queue=False,
        )

    return demo


if __name__ == "__main__":
    initial = os.environ.get("PREVIEW_LANG", "it_IT")
    port = int(os.environ.get("PREVIEW_PORT", "7861"))
    demo = build_ui(initial_lang=initial)

    gradio_major = int(gr.__version__.split(".", 1)[0])
    launch_kwargs: dict = {}
    if gradio_major >= 6:
        launch_kwargs["css"] = STYLE_CSS + EXTRA_CSS

    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inbrowser=False,
        prevent_thread_lock=False,
        **launch_kwargs,
    )
