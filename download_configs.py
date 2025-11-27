# reference: https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/main/download_supported_configs.py

import os
import shutil

from huggingface_hub import snapshot_download

PIPELINE_PATHS = (
    # "Qwen/Qwen-Image",
    # "neta-art/Neta-Lumina-Diffusers",
    "Tongyi-MAI/Z-Image-Turbo",
)

for pretrained in PIPELINE_PATHS:
    try:
        local_dir = os.path.join("backend", "huggingface", pretrained.removesuffix("-Diffusers"))
        os.makedirs(local_dir, exist_ok=True)

        snapshot_download(pretrained, local_dir=local_dir, allow_patterns=["*.json", "*.txt"], token=None, force_download=True)

        shutil.rmtree(os.path.join(local_dir, ".cache"))

        _files = []
        for dirpath, _, filenames in os.walk(local_dir):
            for filename in filenames:
                if filename.endswith(".safetensors.index.json"):
                    os.remove(os.path.join(dirpath, filename))
                elif filename.endswith((".json", ".txt")):
                    _files.append(os.path.join(dirpath, filename))

        for file in _files:
            with open(file, "r", newline="\n", encoding="utf-8") as infile:
                lines = infile.readlines()
            with open(file, "w", newline="\r\n", encoding="utf-8") as outfile:
                outfile.writelines(lines)

        print(pretrained)
    except Exception as e:
        print(e)
