import os
import sys
from modules.paths_internal import models_path, script_path, data_path, extensions_dir, extensions_builtin_dir, cwd  # noqa: F401

import modules.safe  # noqa: F401


def mute_sdxl_imports():
    """create fake modules that SDXL wants to import but doesn't actually use for our purposes"""

    class Dummy:
        pass

    module = Dummy()
    module.LPIPS = None
    sys.modules['taming.modules.losses.lpips'] = module

    module = Dummy()
    module.StableDataModuleFromConfig = None
    sys.modules['sgm.data'] = module


# data_path = cmd_opts_pre.data
sys.path.insert(0, script_path)

sd_path = os.path.abspath(os.path.join(script_path, 'repositories' ,'stable-diffusion-stability-ai'))
assert os.path.exists(sd_path), "Couldn't find Stable Diffusion"

mute_sdxl_imports()

path_dirs = [
    (sd_path, 'ldm', 'Stable Diffusion', []),
    ('ldm_patched', 'k_diffusion/sampling.py', 'k_diffusion', []),
    (os.path.join(sd_path, '../generative-models'), 'sgm', 'Stable Diffusion XL', ["sgm"]),
]

paths = {}

for path, target_file, name, options in path_dirs:
    must_exist_path = os.path.abspath(os.path.join(script_path, path, target_file))
    if not os.path.exists(must_exist_path):
        print(f"Warning: {name} not found at path {must_exist_path}", file=sys.stderr)
        continue

    path = os.path.abspath(path)
    if "sgm" in options:
        # SDXL Repo has a scripts dir with __init__.py in it, which breaks every extension's scripts dir
        # so we import sgm then remove it from the sys.path
        sys.path.insert(0, path)
        import sgm  # noqa: F401
        sys.path.pop(0)
    else:
        sys.path.append(path)
    paths[name] = path


import ldm_patched.utils.path_utils as ldm_patched_path_utils

ldm_patched_path_utils.base_path = data_path
ldm_patched_path_utils.models_dir = models_path
