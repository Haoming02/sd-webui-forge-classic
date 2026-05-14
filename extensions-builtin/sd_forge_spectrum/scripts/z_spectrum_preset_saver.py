import os
import json
import gradio as gr
from modules import scripts
import sys

def apply_spectrum_patch():
    # We need to get the original class without causing A1111 to double-load it.
    # The safest way is to find it among subclasses of scripts.Script.
    SpectrumForForge_class = None
    for cls in scripts.Script.__subclasses__():
        if cls.__name__ == "SpectrumForForge":
            SpectrumForForge_class = cls
            break

    if not SpectrumForForge_class:
        print("[Spectrum Preset Saver] Could not find SpectrumForForge class. Is the extension installed and loaded?")
        return

    preset_file = os.path.join(os.path.dirname(__file__), "spectrum_presets.json")

    def load_presets():
        if os.path.exists(preset_file):
            try:
                with open(preset_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_presets(presets):
        with open(preset_file, "w") as f:
            json.dump(presets, f, indent=4)

    if not hasattr(SpectrumForForge_class, '_original_ui'):
        SpectrumForForge_class._original_ui = SpectrumForForge_class.ui

        def wrapped_ui(self, *args, **kwargs):
            from modules.ui_components import InputAccordionImpl
            
            # Capture the accordion created by the original UI
            captured_accordion = []
            original_enter = InputAccordionImpl.__enter__
            
            def patched_enter(self_acc):
                captured_accordion.append(self_acc)
                return original_enter(self_acc)
                
            InputAccordionImpl.__enter__ = patched_enter
            try:
                # Call the original UI first
                elements = self._original_ui(*args, **kwargs)
            finally:
                InputAccordionImpl.__enter__ = original_enter
            
            # Elements returned: [enable, w, m, lam, window_size, flex_window, warmup_steps, stop_caching_step]
            enable = elements[0]
            param_components = list(elements[1:])
            
            # If we found the original accordion, inject our preset UI inside it.
            # Otherwise fallback to a standard Row.
            parent_context = captured_accordion[0] if captured_accordion else gr.Row()
            
            with parent_context:
                gr.Markdown("### Spectrum Presets")
                with gr.Row():
                    preset_name = gr.Textbox(label="Preset Name", placeholder="Enter preset name...", scale=3)
                    save_btn = gr.Button("Save Preset", scale=1)
                    refresh_btn = gr.Button("🔄", scale=0, min_width=40)
                with gr.Row():
                    preset_dropdown = gr.Dropdown(label="Load Preset", choices=list(load_presets().keys()), scale=4)
                    reset_btn = gr.Button("↺ Reset", scale=1)
                    delete_btn = gr.Button("🗑️ Delete", scale=1, variant="stop")
                
                def on_save(name, *current_params):
                    if not name:
                        return gr.update()
                    presets = load_presets()
                    presets[name] = list(current_params)
                    save_presets(presets)
                    return gr.update(choices=list(presets.keys()), value=name)
                    
                save_btn.click(
                    fn=on_save,
                    inputs=[preset_name] + param_components,
                    outputs=[preset_dropdown]
                )
                
                def on_load(name):
                    presets = load_presets()
                    if name in presets:
                        values = presets[name]
                        # Ensure we don't crash if params count changed in an update
                        updates = []
                        for i, comp in enumerate(param_components):
                            if i < len(values):
                                updates.append(gr.update(value=values[i]))
                            else:
                                updates.append(gr.update())
                        return updates
                    return [gr.update() for _ in param_components]
                    
                preset_dropdown.change(
                    fn=on_load,
                    inputs=[preset_dropdown],
                    outputs=param_components
                )
                
                def on_refresh():
                    presets = load_presets()
                    return gr.update(choices=list(presets.keys()))
                    
                refresh_btn.click(
                    fn=on_refresh,
                    inputs=[],
                    outputs=[preset_dropdown]
                )
                
                def on_delete(name):
                    if not name:
                        return gr.update()
                    presets = load_presets()
                    if name in presets:
                        del presets[name]
                        save_presets(presets)
                    return gr.update(choices=list(presets.keys()), value=None)
                    
                delete_btn.click(
                    fn=on_delete,
                    inputs=[preset_dropdown],
                    outputs=[preset_dropdown]
                )
                
                def on_reset():
                    # Default values from spectrum.py
                    return [
                        gr.update(value=None),   # preset_dropdown
                        gr.update(value=0.25),   # w
                        gr.update(value=6),      # m
                        gr.update(value=0.5),    # lam
                        gr.update(value=2),      # window_size
                        gr.update(value=0.0),    # flex_window
                        gr.update(value=6),      # warmup_steps
                        gr.update(value=0.9)     # stop_caching_step
                    ]
                    
                reset_btn.click(
                    fn=on_reset,
                    inputs=[],
                    outputs=[preset_dropdown] + param_components
                )
                
            return elements

        SpectrumForForge_class.ui = wrapped_ui

apply_spectrum_patch()
