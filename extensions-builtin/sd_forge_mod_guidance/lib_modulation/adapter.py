# https://github.com/Anzhc/Anima-Mod-Guidance-ComfyUI-Node/blob/main/adapter.py

import os.path
from functools import lru_cache

import torch

from backend.utils import load_torch_file
from modules.paths import models_path

from .logging import logger

ADAPTER_URL = "https://huggingface.co/yresearch/cosmos-pooled/resolve/main/checkpoint_4000.pt"

ADAPTER_PATH: os.PathLike = os.path.abspath(os.path.join(models_path, "modulation_guidance", os.path.basename(ADAPTER_URL)))


def resolve_adapter_path():
    if not os.path.isfile(ADAPTER_PATH):
        logger.info(f'Downloading Adapter to "{ADAPTER_PATH}"...')
        os.makedirs(os.path.dirname(ADAPTER_PATH), exist_ok=True)
        torch.hub.download_url_to_file(ADAPTER_URL, ADAPTER_PATH)
    return ADAPTER_PATH


def _load_adapter_cpu(resolved_path):
    EXPECTED_KEYS = (
        "scales",
        "text_embedder_clip.linear_1.weight",
        "text_embedder_clip.linear_1.bias",
        "text_embedder_clip.linear_2.weight",
        "text_embedder_clip.linear_2.bias",
    )

    state_dict = load_torch_file(resolved_path)
    assert not any([k for k in EXPECTED_KEYS if k not in state_dict])

    return state_dict


@lru_cache(maxsize=1, typed=False)
def get_typed_adapter(path: os.PathLike, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    state_dict = _load_adapter_cpu(path)
    typed_state = {key: value.to(device=device, dtype=dtype) for key, value in state_dict.items()}
    return typed_state
