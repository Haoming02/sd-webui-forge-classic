"""Runtime patches applied to PyTorch when running on top of ZLUDA."""

import torch

from modules_forge import zluda_installer


def _noop(*args, **kwargs):
    return None


def test(device: torch.device) -> Exception | None:
    """Basic sanity check; ZLUDA fails loudly here rather than mid-generation."""

    try:
        a = torch.randn((2, 4), device=device)
        b = torch.randn((4, 8), device=device)
        assert torch.mm(a, b).sum().is_nonzero()
    except Exception as e:
        return e
    return None


def initialize() -> bool:
    """
    Disable every CUDA feature ZLUDA does not implement.

    Returns ``False`` when the device fails the sanity check, in which case the
    caller should fall back to the CPU instead of crashing later.
    """

    torch.backends.cudnn.enabled = zluda_installer.MIOpen_enabled

    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        if not zluda_installer.MIOpen_enabled:
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp = _noop
    else:
        torch.backends.cuda.enable_cudnn_sdp = _noop

    # Neither FlashAttention nor the memory-efficient CUDA kernels exist under
    # ZLUDA; leaving them enabled makes SDPA fall over instead of using math.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_flash_sdp = _noop
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp = _noop
    torch.backends.cuda.enable_math_sdp(True)

    device = torch.device(torch.cuda.current_device())
    error = test(device)
    if error is not None:
        print(f"ZLUDA: device failed the basic operation test (index={device.index})")
        print(error)
        return False

    return True
