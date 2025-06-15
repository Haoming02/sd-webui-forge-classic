from __future__ import annotations

import logging
import os
import os.path
from urllib.parse import urlparse

import spandrel
import spandrel_extra_arches
import torch

from modules import shared
from modules.errors import display
from modules.upscaler import UpscalerLanczos, UpscalerNearest, UpscalerNone

spandrel_extra_arches.install()
logger = logging.getLogger(__name__)


def load_file_from_url(url: str, *, model_dir: str, progress: bool = True, file_name: str | None = None) -> str:
    """
    Download a file from `url` into `model_dir`, using the file present if possible.
    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file

        download_url_to_file(url, cached_file, progress=progress)
    return cached_file


def load_models(
    model_path: str,
    model_url: str = None,
    command_path: str = None,
    ext_filter=None,
    download_name=None,
    ext_blacklist=None,
) -> list:
    """
    A one-and-done loader to try finding the desired models in specified directories.

    - download_name: Specify to download from model_url immediately.
    - model_url: If no other models are found, this will be downloaded on upscale.
    - model_path: The location to store/find models in.
    - command_path: A command-line argument to search for models in first.
    - ext_filter: An optional list of filename extensions to filter by

    @return: A list of paths containing the desired model(s)
    """
    output: set[str] = set()

    try:
        folders = [model_path]

        if command_path != model_path and command_path is not None:
            if os.path.isdir(command_path):
                folders.append(command_path)
            elif os.path.isfile(command_path):
                output.add(command_path)

        for place in folders:
            for full_path in shared.walk_files(place, allowed_extensions=ext_filter):
                if os.path.islink(full_path) and not os.path.exists(full_path):
                    print(f"Skipping broken symlink: {full_path}")
                    continue
                if ext_blacklist is not None and any(full_path.endswith(x) for x in ext_blacklist):
                    continue
                if os.path.isfile(full_path):
                    output.add(full_path)

        if model_url is not None and len(output) == 0:
            if download_name is not None:
                output.add(load_file_from_url(model_url, model_dir=folders[0], file_name=download_name))
            else:
                output.add(model_url)

    except Exception as e:
        display(e, "load_models")

    return sorted(output, key=lambda mdl: mdl.lower())


def friendly_name(file: str) -> str:
    if file.startswith("http"):
        file = urlparse(file).path

    file = os.path.basename(file)
    model_name, _ = os.path.splitext(file)
    return model_name


def load_upscalers():
    from modules.esrgan_model import UpscalerESRGAN

    commandline_model_path = shared.cmd_opts.esrgan_models_path
    upscaler = UpscalerESRGAN(commandline_model_path)
    upscaler.user_path = commandline_model_path
    upscaler.model_download_path = commandline_model_path or upscaler.model_path

    shared.sd_upscalers = [
        *UpscalerNone().scalers,
        *UpscalerLanczos().scalers,
        *UpscalerNearest().scalers,
        *sorted(upscaler.scalers, key=lambda s: s.name.lower()),
    ]


def load_spandrel_model(
    path: str | os.PathLike,
    *,
    device: str | torch.device | None,
    prefer_half: bool = False,
    dtype: str | torch.dtype | None = None,
    expected_architecture: str | None = None,
) -> spandrel.ModelDescriptor:
    model_descriptor = spandrel.ModelLoader(device=device).load_from_file(str(path))
    arch = model_descriptor.architecture
    logger.info(f'Loaded {arch.name} Model: "{os.path.basename(path)}"')

    half = False
    if prefer_half:
        if model_descriptor.supports_half:
            model_descriptor.model.half()
            half = True
        elif model_descriptor.supports_bfloat16:
            model_descriptor.model.bfloat16()
            half = True
        else:
            logger.warning(f"Model {path} does not support half precision...")

    if dtype:
        model_descriptor.model.to(dtype=dtype)

    logger.debug("Loaded %s from %s (device=%s, half=%s, dtype=%s)", arch, path, device, half, dtype)

    model_descriptor.model.eval()
    return model_descriptor
