import os
import shutil
import threading
import urllib.request

import torch

import folder_paths

ADAPTER_URL = "https://huggingface.co/yresearch/cosmos-pooled/resolve/main/checkpoint_4000.pt"
ADAPTER_DIR_NAME = "modulation_guidance"
ADAPTER_FILE_NAME = "checkpoint_4000.pt"

EXPECTED_KEYS = (
    "scales",
    "text_embedder_clip.linear_1.weight",
    "text_embedder_clip.linear_1.bias",
    "text_embedder_clip.linear_2.weight",
    "text_embedder_clip.linear_2.bias",
)

_CACHE_LOCK = threading.Lock()
_CPU_ADAPTER_CACHE = {}
_TYPED_ADAPTER_CACHE = {}


def _default_adapter_path():
    return os.path.abspath(
        os.path.join(folder_paths.models_dir, ADAPTER_DIR_NAME, ADAPTER_FILE_NAME)
    )


def _download_adapter(adapter_path):
    os.makedirs(os.path.dirname(adapter_path), exist_ok=True)
    tmp_path = f"{adapter_path}.tmp"
    try:
        with urllib.request.urlopen(ADAPTER_URL, timeout=120) as response, open(tmp_path, "wb") as tmp_file:
            shutil.copyfileobj(response, tmp_file)
        os.replace(tmp_path, adapter_path)
    except Exception as exc:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise RuntimeError(
            "Failed to download modulation adapter from "
            f"{ADAPTER_URL}: {exc}"
        ) from exc


def resolve_adapter_path(adapter_mode, adapter_path):
    if adapter_mode == "auto_download":
        resolved = _default_adapter_path()
        if not os.path.isfile(resolved) or os.path.getsize(resolved) <= 0:
            _download_adapter(resolved)
        return resolved

    if adapter_mode == "local_file":
        if not adapter_path or not adapter_path.strip():
            raise RuntimeError(
                "adapter_mode=local_file requires a non-empty adapter_path."
            )
        resolved = os.path.abspath(os.path.expanduser(adapter_path.strip()))
        if not os.path.isfile(resolved):
            raise RuntimeError(f"Adapter file not found: {resolved}")
        return resolved

    raise RuntimeError(f"Unsupported adapter_mode: {adapter_mode}")


def _load_adapter_cpu(path):
    resolved_path = os.path.abspath(path)
    with _CACHE_LOCK:
        cached = _CPU_ADAPTER_CACHE.get(resolved_path)
        if cached is not None:
            return cached

    try:
        raw = torch.load(resolved_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load adapter file '{resolved_path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise RuntimeError(
            f"Adapter file '{resolved_path}' is invalid: expected a dict, got {type(raw)}."
        )

    missing = [k for k in EXPECTED_KEYS if k not in raw]
    if missing:
        raise RuntimeError(
            "Adapter file is missing required keys: "
            + ", ".join(missing)
        )

    state = {}
    for key in EXPECTED_KEYS:
        value = raw[key]
        if not torch.is_tensor(value):
            raise RuntimeError(
                f"Adapter key '{key}' must be a tensor, got {type(value)}."
            )
        state[key] = value.detach().float().cpu().contiguous()

    with _CACHE_LOCK:
        _CPU_ADAPTER_CACHE[resolved_path] = state
    return state


def _infer_model_shape(diffusion_model):
    blocks = getattr(diffusion_model, "blocks", None)
    model_channels = getattr(diffusion_model, "model_channels", None)
    use_adaln_lora = getattr(diffusion_model, "use_adaln_lora", None)

    if blocks is None or model_channels is None:
        raise RuntimeError(
            "Unsupported Anima model internals: missing 'blocks' or 'model_channels'."
        )
    if not use_adaln_lora:
        raise RuntimeError(
            "Unsupported Anima model internals: modulation requires use_adaln_lora=True."
        )
    if not isinstance(model_channels, int) or model_channels <= 0:
        raise RuntimeError(
            f"Invalid model_channels value: {model_channels}"
        )

    num_blocks = len(blocks)
    adaln_dim = model_channels * 3
    return num_blocks, adaln_dim


def validate_adapter_for_model(path, diffusion_model):
    resolved_path = os.path.abspath(path)
    state = _load_adapter_cpu(resolved_path)
    num_blocks, adaln_dim = _infer_model_shape(diffusion_model)

    scales = state["scales"]
    if scales.ndim != 2:
        raise RuntimeError(
            f"Adapter 'scales' must be rank-2, got rank {scales.ndim}."
        )
    if scales.shape[0] != num_blocks:
        raise RuntimeError(
            "Adapter/model block mismatch: "
            f"adapter has {scales.shape[0]} blocks, model has {num_blocks}."
        )
    if scales.shape[1] != adaln_dim:
        raise RuntimeError(
            "Adapter/model channel mismatch: "
            f"adapter adaln dim is {scales.shape[1]}, expected {adaln_dim}."
        )

    l1_w = state["text_embedder_clip.linear_1.weight"]
    l1_b = state["text_embedder_clip.linear_1.bias"]
    l2_w = state["text_embedder_clip.linear_2.weight"]
    l2_b = state["text_embedder_clip.linear_2.bias"]

    if l1_w.ndim != 2 or l1_b.ndim != 1 or l2_w.ndim != 2 or l2_b.ndim != 1:
        raise RuntimeError("Adapter text_embedder_clip tensors have invalid ranks.")

    pooled_dim = int(l1_w.shape[1])
    if l1_w.shape[0] != adaln_dim:
        raise RuntimeError(
            "Adapter linear_1 output mismatch: "
            f"{l1_w.shape[0]} vs expected {adaln_dim}."
        )
    if l1_b.shape[0] != adaln_dim:
        raise RuntimeError(
            "Adapter linear_1 bias mismatch: "
            f"{l1_b.shape[0]} vs expected {adaln_dim}."
        )
    if l2_w.shape != (adaln_dim, adaln_dim):
        raise RuntimeError(
            "Adapter linear_2 weight mismatch: "
            f"{tuple(l2_w.shape)} vs expected ({adaln_dim}, {adaln_dim})."
        )
    if l2_b.shape[0] != adaln_dim:
        raise RuntimeError(
            "Adapter linear_2 bias mismatch: "
            f"{l2_b.shape[0]} vs expected {adaln_dim}."
        )

    return {
        "resolved_path": resolved_path,
        "num_blocks": num_blocks,
        "adaln_dim": adaln_dim,
        "pooled_dim": pooled_dim,
    }


def get_typed_adapter(path, diffusion_model, device, dtype):
    meta = validate_adapter_for_model(path, diffusion_model)
    cache_key = (meta["resolved_path"], str(device), str(dtype))
    with _CACHE_LOCK:
        cached = _TYPED_ADAPTER_CACHE.get(cache_key)
        if cached is not None:
            return cached, meta

    cpu_state = _load_adapter_cpu(meta["resolved_path"])
    typed_state = {
        key: value.to(device=device, dtype=dtype) for key, value in cpu_state.items()
    }

    with _CACHE_LOCK:
        _TYPED_ADAPTER_CACHE[cache_key] = typed_state
    return typed_state, meta
