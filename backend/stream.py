import torch

from backend import memory_management
from modules.devices import device


def stream_context():
    if torch.cuda.is_available():
        return torch.cuda.stream
    if torch.xpu.is_available():
        return torch.xpu.stream


def get_current_stream():
    return memory_management.current_stream(device)


def get_new_stream():
    return memory_management.get_offload_stream(device)


def should_use_stream():
    return current_stream is not None and mover_stream is not None


current_stream = get_current_stream()
mover_stream = get_new_stream()
