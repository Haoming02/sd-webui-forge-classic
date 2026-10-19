import gradio as gr

from modules import script_callbacks


def builder_ui():

    with gr.Blocks() as TAB:
        gr.Textbox
        pass

    return [(TAB, "JSON Builder", "sd-forge-json-builder")]


script_callbacks.on_ui_tabs(builder_ui)
