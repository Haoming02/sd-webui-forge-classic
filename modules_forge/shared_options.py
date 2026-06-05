def register(options_templates, options_section, OptionInfo):
    import gradio as gr

    from modules.ui_components import FormColorPicker

    options_templates.update(
        options_section(
            (None, "Forge Hidden Options"),
            {
                "VERSION_UID": OptionInfo(None, "internal version for breaking-changes"),
                "forge_preset": OptionInfo("sd"),
                "forge_additional_modules": OptionInfo([]),
                "forge_unet_storage_dtype": OptionInfo("Automatic"),
            },
        )
    )
    options_templates.update(
        options_section(
            ("ideogram4", "Ideogram 4.0"),
            {
                "ideogram4_model_path": OptionInfo("", "Model path").info("local diffusers folder for Ideogram 4.0 (or a Hugging Face repo id such as ideogram-ai/ideogram-4-nf4)"),
                "ideogram4_quantization": OptionInfo("nf4", "Quantization", gr.Radio, {"choices": ["nf4", "fp8"]}).info("nf4 requires CUDA and is Diffusers-compatible; fp8 needs the official non-diffusers loader"),
                "ideogram4_hf_token": OptionInfo("", "HF token").info("only needed when loading gated weights by repo id; falls back to the HF_TOKEN environment variable"),
            },
        )
    )
    options_templates.update(
        options_section(
            ("ui_forgecanvas", "Forge Canvas", "ui"),
            {
                "forge_canvas_height": OptionInfo(512, "Canvas Height").info("in pixels").needs_reload_ui(),
                "forge_canvas_toolbar_always": OptionInfo(False, "Always Visible Toolbar").info("disabled: toolbar only appears when hovering the canvas").needs_reload_ui(),
                "forge_canvas_consistent_brush": OptionInfo(False, "Fixed Brush Size").info("disabled: the brush size is <b>pixel-space</b>, the brush stays small when zoomed out ; enabled: the brush size is <b>canvas-space</b>, the brush stays big when zoomed in").needs_reload_ui(),
                "forge_canvas_plain": OptionInfo(False, "Plain Background").info("disabled: checkerboard pattern ; enabled: solid color").needs_reload_ui(),
                "forge_canvas_plain_color": OptionInfo("#808080", "Solid Color for Plain Background", FormColorPicker, {}).needs_reload_ui(),
            },
        )
    )
