"""
this module defines internal paths used by program and is safe to import before dependencies are installed in launch.py
"""

import os
import shlex
import sys
from pathlib import Path

normalized_filepath = lambda filepath: str(Path(filepath).absolute())

commandline_args = os.environ.get("COMMANDLINE_ARGS", "")
sys.argv += shlex.split(commandline_args)

cwd = os.getcwd()
modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)

from backend.args import parser

parser.add_argument("--data-dir", type=str, default=os.path.dirname(modules_path), help="base path where all user data is stored")
parser.add_argument("--model-ref", type=str, default=None, help="base path for all models")
cmd_opts_pre, _ = parser.parse_known_args()

data_path = cmd_opts_pre.data_dir

models_path = cmd_opts_pre.model_ref or os.path.join(data_path, "models")
extensions_dir = os.path.join(data_path, "extensions")
default_output_dir = os.path.join(data_path, "output")
extensions_builtin_dir = os.path.join(script_path, "extensions-builtin")
config_states_dir = os.path.join(script_path, "config_states")
presets_dir = os.path.join(script_path, "presets")

roboto_ttf_file = os.path.join(modules_path, "Roboto-Regular.ttf")
