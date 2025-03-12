import os
import sys
from modules.paths_internal import models_path, script_path, data_path, extensions_dir, extensions_builtin_dir, cwd  # noqa: F401

import modules.safe  # noqa: F401


sys.path.insert(0, script_path)

sd_path = os.path.normpath(os.path.join(script_path, "modules_forge", "diffusion_engine"))


path_dirs = [
    (sd_path, "ldm", "Stable Diffusion"),
    (sd_path, "sgm", "Stable Diffusion XL"),
    ("ldm_patched", "k_diffusion", "k_diffusion"),
]

paths = {}

for path, target_file, name in path_dirs:
    must_exist_path = os.path.abspath(os.path.join(script_path, path, target_file))
    if not os.path.exists(must_exist_path):
        print(f'Error: {name} was not found at path "{must_exist_path}"', file=sys.stderr)
        continue

    path = os.path.abspath(path)
    sys.path.append(path)
    paths[name] = path


import ldm_patched.utils.path_utils as ldm_patched_path_utils

ldm_patched_path_utils.base_path = data_path
ldm_patched_path_utils.models_dir = models_path
