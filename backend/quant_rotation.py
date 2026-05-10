# https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/main/convrot.py

# Group-wise Hadamard rotation for INT8 quantization quality improvement
# reference: https://github.com/newgrit1004/ComfyUI-ZImage-Triton
# License: MIT


import torch

# Cache Hadamard matrices by (size, device, dtype) to avoid recomputation
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a normalized REGULAR orthogonal Hadamard matrix (ConvRot).

    Size must be a power of 4 (e.g., 4, 16, 64, 256, 1024...).
    Uses the Kronecker construction from Theorem 3.3 to avoid the all-1s
    column of standard Sylvester Hadamard matrices, which amplifies
    row-wise outliers in diffusion models.
    """
    import math

    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
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
    _HADAMARD_CACHE[cache_key] = H_normalized

    return H_normalized


def rotate_weight(
    weight: torch.Tensor,
    H: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Rotate weight matrix offline: W_rot = W @ H_block^T.

    For Linear(in, out) with weight shape (out, in):
    Each row of W is split into groups of group_size and rotated by H^T.

    Args:
        weight: Shape (out_features, in_features).
        H: Normalized Hadamard matrix, shape (group_size, group_size).
        group_size: Group size for block-diagonal rotation.

    Returns:
        Rotated weight, same shape as input.
    """
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size

    # (out, in) → (out, n_groups, group_size)
    W_grouped = weight.view(out_f, n_groups, group_size)
    # Apply H^T to each group: (..., group_size) @ (group_size, group_size)
    H_t = H.T.to(dtype=weight.dtype, device=weight.device)
    W_rot = torch.matmul(W_grouped, H_t)
    return W_rot.reshape(out_f, in_f)


def rotate_activation(
    x: torch.Tensor,
    H: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Rotate activation online: x_rot = x @ H_block.

    Group-wise Hadamard spreads outliers across channels within each group.

    Args:
        x: Shape (..., features). Last dim must be divisible by group_size.
        H: Normalized Hadamard matrix, shape (group_size, group_size).
        group_size: Group size for block-diagonal rotation.

    Returns:
        Rotated activation, same shape as input.
    """
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
