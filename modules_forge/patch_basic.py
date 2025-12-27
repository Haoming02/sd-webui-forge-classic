import os
import time
import warnings
from functools import wraps
from pathlib import Path

import gradio.networking
import httpx
import safetensors.torch
import torch
from tqdm import tqdm

from modules.errors import display


def gradio_url_ok_fix(url: str) -> bool:
    try:
        for _ in range(5):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                r = httpx.head(url, timeout=999, verify=False)
            if r.status_code in (200, 401, 302):
                return True
            time.sleep(0.500)
    except (ConnectionError, httpx.ConnectError):
        return False
    return False


def build_loaded(module, loader_name):
    original_loader_name = f"{loader_name}_origin"

    if not hasattr(module, original_loader_name):
        setattr(module, original_loader_name, getattr(module, loader_name))

    original_loader = getattr(module, original_loader_name)

    @wraps(original_loader)
    def loader(*args, **kwargs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                return original_loader(*args, **kwargs)
        except Exception as e:
            display(e, f"{module.__name__}.{loader_name}")

            exc = "\n"
            for path in list(args) + list(kwargs.values()):
                if isinstance(path, str) and os.path.isfile(path):
                    exc += f'Failed to read file "{path}"\n'
                    backup_file = f"{path}.corrupted"
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                    os.replace(path, backup_file)
                    exc += f'Forge has moved the corrupted file to "{backup_file}"\n'
                    exc += "Please try downloading the model again\n"
            print(exc)
            raise ValueError from None

    setattr(module, loader_name, loader)


def always_show_tqdm(*args, **kwargs):
    kwargs["disable"] = False
    if "name" in kwargs:
        del kwargs["name"]
    return tqdm(*args, **kwargs)


def long_path_prefix(path: Path) -> Path:
    if os.name == "nt" and not str(path).startswith("\\\\?\\") and not path.exists():
        return Path("\\\\?\\" + str(path))
    return path


def patch_all_basics():
    import logging

    from huggingface_hub import file_download

    file_download.tqdm = always_show_tqdm
    file_download.logger.setLevel(logging.ERROR)

    from huggingface_hub.file_download import _download_to_tmp_and_move as original_download_to_tmp_and_move

    @wraps(original_download_to_tmp_and_move)
    def patched_download_to_tmp_and_move(incomplete_path: Path, destination_path: Path, *args, **kwargs):
        incomplete_path = long_path_prefix(incomplete_path)
        destination_path = long_path_prefix(destination_path)
        return original_download_to_tmp_and_move(incomplete_path, destination_path, *args, **kwargs)

    file_download._download_to_tmp_and_move = patched_download_to_tmp_and_move

    gradio.networking.url_ok = gradio_url_ok_fix
    build_loaded(safetensors.torch, "load_file")
    build_loaded(torch, "load")
