import os
import sys

import gradio as gr
from ldm_patched.modules.model_management import xformers_enabled
from modules import (
    options,
    sd_models_types,
    shared_cmd_options,
    shared_gradio_themes,
    shared_items,
    util,
)
from modules.paths_internal import (
    data_path,
    default_sd_model_file,
    extensions_builtin_dir,
    extensions_dir,
    models_path,
    script_path,
    sd_configs_path,
    sd_default_config,
    sd_model_file,
)  # noqa: F401

cmd_opts = shared_cmd_options.cmd_opts
parser = shared_cmd_options.parser

parallel_processing_allowed = True
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}
config_filename = cmd_opts.ui_settings_file
styles_filename = cmd_opts.styles_file = cmd_opts.styles_file if len(cmd_opts.styles_file) > 0 else [os.path.join(data_path, "styles.csv")]

demo = None

device = None

weight_load_location = None

xformers_available = xformers_enabled()

state = None

prompt_styles = None

interrogator = None

face_restorers = []

options_templates = None
opts = None
restricted_opts = None

sd_model: sd_models_types.WebuiSdModel = None

settings_components = None
"""assigned from ui.py, a mapping on setting names to gradio components repsponsible for those settings"""

tab_names = []

latent_upscale_default_mode = "Latent"
latent_upscale_modes = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}

sd_upscalers = []

clip_model = None

progress_print_out = sys.stdout

gradio_theme = gr.themes.Base()

total_tqdm = None

mem_mon = None

options_section = options.options_section
OptionInfo = options.OptionInfo
OptionHTML = options.OptionHTML

natural_sort_key = util.natural_sort_key
listfiles = util.listfiles
html_path = util.html_path
html = util.html
walk_files = util.walk_files

reload_gradio_theme = shared_gradio_themes.reload_gradio_theme

list_checkpoint_tiles = shared_items.list_checkpoint_tiles
refresh_checkpoints = shared_items.refresh_checkpoints
list_samplers = shared_items.list_samplers
