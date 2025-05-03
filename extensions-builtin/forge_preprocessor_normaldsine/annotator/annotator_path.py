from modules_forge.shared import preprocessor_dir
import os

try:
    from lib_controlnet.logging import logger

    print = logger.info
except ImportError:
    pass

models_path = preprocessor_dir
clip_vision_path = os.path.join(preprocessor_dir, "clip_vision")

os.makedirs(models_path, exist_ok=True)
os.makedirs(clip_vision_path, exist_ok=True)

print(f"Preprocessor location: {models_path}")
# This file is here to satisfy the import made by the ln 8 in __init__.py, and 
# is a copy of /forge_legacy_preprocessors/annotators/annotator_path.py
