import gradio as gr

from modules import script_callbacks


def builder_ui():
    return [(gr.Blocks(), "JSON Builder", "sd-forge-json-builder")]


script_callbacks.on_ui_tabs(builder_ui)
