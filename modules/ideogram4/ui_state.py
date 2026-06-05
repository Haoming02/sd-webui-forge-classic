"""Shared handles to the Ideogram 4.0 txt2img UI components.

Populated by ``modules/processing_scripts/ideogram4.py`` in ``ui()`` and consumed
by ``modules_forge.main_entry.forge_main_entry()`` to wire preset-driven
visibility / value updates.

Both sides import this module by its canonical name, so they share one object —
this avoids the timing fragility of ``on_after_component`` (which is finalized
before ``ui()`` runs) and the duplicate-module hazard of importing a script file
directly.
"""

# the gr.Group panel that is shown only when the model-type preset is "ideogram4"
group = None

# gr.Dropdown of Sampler Presets (V4_*) — drives the Steps slider
sampler_preset = None

# gr.Dropdown of resolution presets (WxH) — drives the Width/Height sliders
resolution_preset = None

# label -> (width, height) | None  (mirror of the script's RESOLUTION_PRESETS)
resolution_map = {}


def reset():
    global group, sampler_preset, resolution_preset, resolution_map
    group = None
    sampler_preset = None
    resolution_preset = None
    resolution_map = {}
