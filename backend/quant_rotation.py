# https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/main/convrot.py

# Group-wise Hadamard rotation for INT8 quantization quality improvement
# reference: https://github.com/newgrit1004/ComfyUI-ZImage-Triton
# License: MIT


import math

import torch

# Cache Hadamard matrices by (size, device, dtype) to avoid recomputation
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def build_hadamard(size: int, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:

    if not torch.compiler.is_compiling():
        if (cache_key := (size, str(device), dtype)) in _HADAMARD_CACHE:
            return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    # Base H4 from Theorem 3.3 (Eq 9 in the paper)
    # Notice how every row and column sums to exactly 2
    H4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)

    H = H4
    current_size = 4

    # Kronecker construction for larger sizes: H_{4^{k+1}} = H_{4^k} \otimes H_4
    while current_size < size:
        H = torch.kron(H, H4)
        current_size *= 4

    # Normalize to make it orthogonal
    H_normalized = H / (size**0.5)

    if not torch.compiler.is_compiling():
        _HADAMARD_CACHE[cache_key] = H_normalized

    return H_normalized


def rotate_activation(x: torch.Tensor, H: torch.Tensor, group_size: int) -> torch.Tensor:
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    n_groups = features // group_size

    # (..., features) → (..., n_groups, group_size)
    x_grouped = x.view(*orig_shape[:-1], n_groups, group_size)
    H_dev = H.to(dtype=x.dtype, device=x.device)
    x_rot = torch.matmul(x_grouped, H_dev)
    return x_rot.view(orig_shape)
