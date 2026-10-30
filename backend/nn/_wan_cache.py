import torch
import torch.nn.functional as F

from backend.operations import ForgeOperations as ops

CACHE_T = 2

STRIP_ELEMS = 2**24


def torch_cat_if_needed(xl, dim):
    xl = [x for x in xl if x is not None and x.shape[dim] > 0]
    if len(xl) > 1:
        return torch.cat(xl, dim)
    elif len(xl) == 1:
        return xl[0]
    else:
        return None


class CausalConv3d(ops.Conv3d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = 2 * self.padding[0]
        self.padding = (0, self.padding[1], self.padding[2])

    def forward(self, x, cache_x=None):
        if cache_x is None and x.shape[2] == 1:
            return super().forward(x, autopad="causal_zero")

        if self._padding > 0:
            padding_needed = self._padding
            if cache_x is not None:
                cache_x = cache_x.to(x.device)
                padding_needed = max(0, padding_needed - cache_x.shape[2])
            padding_shape = list(x.shape)
            padding_shape[2] = padding_needed
            padding = torch.zeros(padding_shape, device=x.device, dtype=x.dtype)
            x = torch_cat_if_needed([padding, cache_x, x], dim=2)
            del cache_x

        return super().forward(x)


def strip_apply(fn, x, scale=1, halo=1, out=None, accumulate=False):
    n = -(-x.numel() * scale * scale // STRIP_ELEMS)
    if n <= 1 and out is None:
        return fn(x, 0)
    size = x.shape[-2]

    step = max(halo, -(-size // n))
    put = torch.Tensor.add_ if accumulate else torch.Tensor.copy_
    held = None
    for a in range(0, size, step):
        b = min(size, a + step)
        lo = max(0, a - halo)
        y = fn(x.narrow(-2, lo, min(size, b + halo) - lo), lo).narrow(-2, (a - lo) * scale, (b - a) * scale)
        if out is None:
            out = y.new_empty(*y.shape[:-2], size * scale, y.shape[-1])
        if held is not None:
            put(out.narrow(-2, a * scale - held.shape[-2], held.shape[-2]), held)
        keep = min(halo, b - a)
        put(out.narrow(-2, a * scale, (b - a - keep) * scale), y.narrow(-2, 0, (b - a - keep) * scale))
        held = y.narrow(-2, (b - a - keep) * scale, keep * scale).clone() if keep else None
    if held is not None:
        put(out.narrow(-2, size * scale - held.shape[-2], held.shape[-2]), held)

    return out


def cached_chain(layers, x, feat_cache=None, feat_idx=[0]):
    convs = [layer for layer in layers if isinstance(layer, CausalConv3d)]
    old = new = [None] * len(convs)
    if feat_cache is not None:
        i = feat_idx[0]
        feat_idx[0] += len(convs)
        old = feat_cache[i : i + len(convs)]
        new = [x.new_empty(x.shape[0], conv.in_channels, min(CACHE_T, x.shape[2]), *x.shape[-2:]) for conv in convs]
        feat_cache[i : i + len(convs)] = new

    def fn(s, lo):
        k = 0
        for layer in layers:
            if isinstance(layer, CausalConv3d):
                if new[k] is not None:
                    off = min(lo, k)
                    new[k].narrow(-2, lo + off, s.shape[-2] - off).copy_(s[:, :, -CACHE_T:, off:])
                s = layer(s, None if old[k] is None else old[k].narrow(-2, lo, s.shape[-2]))
                k += 1
            else:
                s = layer(s)
        return s

    return fn


def strip_residual(block, x, feat_cache=None, feat_idx=[0]):
    fn = cached_chain(block.residual, x, feat_cache, feat_idx)
    if block.in_dim == block.out_dim:
        return strip_apply(fn, x, halo=2, out=x, accumulate=True)
    out = x.narrow(1, 0, block.out_dim) if block.in_dim > block.out_dim else None
    return strip_apply(lambda s, lo: fn(s, lo).add_(block.shortcut(s)), x, halo=2, out=out)


def strip_down(resample, x):
    rows = x.shape[-2] // 2
    n = -(-x.numel() // STRIP_ELEMS)
    if n <= 1:
        return resample(x)
    step = -(-rows // n)
    out = None
    for a in range(0, rows, step):
        b = min(rows, a + step)
        hi = min(x.shape[-2], 2 * b + 1)
        y = resample[1](F.pad(x.narrow(-2, 2 * a, hi - 2 * a), (0, 1, 0, int(hi == x.shape[-2]))))
        if out is None:
            out = y.new_empty(*y.shape[:-2], rows, y.shape[-1])
        out.narrow(-2, a, b - a).copy_(y)
    return out


def keep_two_frames(feat_cache, i, old):
    for j, o in enumerate(old):
        if o is not None and feat_cache[i + j].shape[2] < CACHE_T:
            feat_cache[i + j] = torch.cat([o[:, :, -1:], feat_cache[i + j]], 2)
