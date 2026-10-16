"""Package a local MOSS-Image HF directory into Forge checkpoints/modules.

Adapted from OpenMOSS/MOSS-Image's comfyui/export.py (Apache-2.0).
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def read_component(directory: Path, stem: str):
    index = directory / f"{stem}.safetensors.index.json"
    if not index.exists():
        return load_file(directory / f"{stem}.safetensors")
    mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    state = {}
    for filename in sorted(set(mapping.values())):
        shard = (directory / filename).resolve()
        if not shard.is_relative_to(directory.resolve()):
            raise ValueError("Shard path escapes the component directory")
        values = load_file(shard)
        if values.keys() != {key for key, file in mapping.items() if file == filename}:
            raise ValueError(f"Shard contents do not match index: {filename}")
        state.update(values)
    return state


def convert(source: Path, models: Path, name: str = "moss-image"):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("name must be a filename prefix, not a path")
    config = json.loads((source / "transformer/config.json").read_text(encoding="utf-8"))
    if config.get("_class_name") not in ("MOSImageTransformer", "MOSSImageTransformer"):
        raise ValueError("Expected a MOSS-Image transformer directory")
    for key, expected in {"text_dim": 4096, "freq_dim": 256, "latent_channels": 128, "rope_base": 10000.0}.items():
        if config.get(key) != expected:
            raise ValueError(f"Unsupported MOSS-Image {key}: {config.get(key)}")
    targets = [models / f"Stable-diffusion/{name}-dit.safetensors", models / f"text_encoder/{name}-qwen35.safetensors", models / f"VAE/{name}-vae.safetensors"]
    for target in targets:
        if target.exists() or target.with_suffix(".partial").exists():
            raise FileExistsError(target)
    for component, stem, target in zip(("transformer", "text_encoder", "vae"), ("diffusion_pytorch_model", "model", "diffusion_pytorch_model"), targets):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".partial")
        single = source / component / f"{stem}.safetensors"
        if single.exists():
            shutil.copyfile(single, temporary)
        else:
            state = read_component(source / component, stem)
            save_file(state, temporary, metadata={"format": "pt", "model": "moss-image"})
            del state
        temporary.rename(target)
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local HF bundle containing transformer/, text_encoder/ and vae/")
    parser.add_argument("--models", type=Path, default=Path(__file__).resolve().parents[1] / "models", help="Forge models directory")
    parser.add_argument("--name", default="moss-image", help="Output filename prefix")
    args = parser.parse_args()
    for path in convert(args.source, args.models, args.name):
        print(path)  # noqa: T201


if __name__ == "__main__":
    main()
