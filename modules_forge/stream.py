# https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14855

import torch

from ldm_patched.modules import model_management
from ldm_patched.modules.args_parser import args

from contextlib import contextmanager

def stream_context() -> torch.cuda.StreamContext | torch.xpu.StreamContext:
    if torch.cuda.is_available():
        return torch.cuda.stream

    if model_management.is_intel_xpu():
        return torch.xpu.stream

    return None


def get_current_stream(synchronize: bool = True) -> torch.cuda.Stream | torch.xpu.Stream:
    try:
        if torch.cuda.is_available():
            device = torch.device(torch.cuda.current_device())
            stream = torch.cuda.current_stream(device)
            if synchronize: # stream.synchronize() blocks, don't call if we don't have to.
                with torch.cuda.stream(stream):
                    torch.zeros((1, 1)).to(device, torch.float32)
                stream.synchronize()
            return stream
        if model_management.is_intel_xpu():
            device = torch.device("xpu")
            stream = torch.xpu.current_stream(device)
            if synchronize:
                with torch.xpu.stream(stream):
                    torch.zeros((1, 1)).to(device, torch.float32)
                stream.synchronize()
            return stream
    except Exception:
        return None


def get_new_stream() -> torch.cuda.Stream | torch.xpu.Stream:
    try:
        if torch.cuda.is_available():
            device = torch.device(torch.cuda.current_device())
            stream = torch.cuda.Stream(device)
            with torch.cuda.stream(stream):
                torch.zeros((1, 1)).to(device, torch.float32)
            stream.synchronize()
            return stream
        if model_management.is_intel_xpu():
            device = torch.device("xpu")
            stream = torch.xpu.Stream(device)
            with torch.xpu.stream(stream):
                torch.zeros((1, 1)).to(device, torch.float32)
            stream.synchronize()
            return stream
    except Exception:
        return None

# Returns False if we're running in a stream other than the default/main stream.
def on_default_stream() -> bool:
    return get_current_stream(False) == current_stream

@contextmanager
def async_stream(target_stream: torch.cuda.Stream | torch.xpu.Stream, synchronize: bool = False):
    if target_stream is None or not using_stream or not on_default_stream():
        yield # Don't use a stream is we can't, or if we're already on a stream besides the default.
    else:
        with stream_context()(target_stream):
            yield
        if synchronize:
            target_stream.synchronize()
        else:
            current_stream.wait_stream(target_stream)

current_stream = None
mover_stream = None
using_stream = False

if args.cuda_stream:
    current_stream = get_current_stream()
    mover_stream = get_new_stream()
    using_stream = current_stream is not None and mover_stream is not None
