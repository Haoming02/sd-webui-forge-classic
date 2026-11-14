# https://github.com/molbal/ComfyUI-GGUF/blob/v26-10-30/kquant_triton_backend.py
# Apache-2.0

import logging
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.operations_gguf import ParameterGGUF

import torch

logger = logging.getLogger(__name__)
_failed_routes = set()


@cache
def _supported_device(device):
    return torch.version.cuda is not None and torch.cuda.get_device_capability(device) in {(8, 6), (8, 9)}


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
else:

    @triton.jit
    def _decode(q_ptr, w_ptr, QTYPE: tl.constexpr, DECODE_DTYPE: tl.constexpr):
        block = tl.program_id(0).to(tl.int64)
        local = tl.arange(0, 256)

        if QTYPE == "Q2_K":
            base = q_ptr + block * 84
            group = local // 16
            d = tl.load((base + 80).to(tl.pointer_type(tl.float16))).to(DECODE_DTYPE).to(tl.float32)
            dmin = tl.load((base + 82).to(tl.pointer_type(tl.float16))).to(DECODE_DTYPE).to(tl.float32)
            scale = tl.load(base + group)
            dl = (d * (scale & 15).to(tl.float32)).to(DECODE_DTYPE).to(tl.float32)
            ml = (dmin * (scale >> 4).to(tl.float32)).to(DECODE_DTYPE).to(tl.float32)

            q = (tl.load(base + 16 + local // 128 * 32 + local % 32) >> (local // 32 % 4 * 2)) & 3
            w = (dl * q.to(tl.float32)).to(DECODE_DTYPE).to(tl.float32) - ml
        elif QTYPE == "Q3_K":
            base = q_ptr + block * 110
            group = local // 16
            d = tl.load((base + 108).to(tl.pointer_type(tl.float16))).to(DECODE_DTYPE).to(tl.float32)
            lo = (tl.load(base + 96 + group % 8) >> (group // 8 * 4)) & 15
            hi = (tl.load(base + 104 + group % 4) >> (group // 4 * 2)) & 3
            scale = (lo.to(tl.int32) | (hi.to(tl.int32) << 4)) - 32
            scaled = (d * scale.to(tl.float32)).to(DECODE_DTYPE).to(tl.float32)

            qs = tl.load(base + 32 + local // 128 * 32 + local % 32)
            ql = (qs >> (local // 32 % 4 * 2)) & 3
            qh = ((tl.load(base + local % 32) >> (local // 32)) & 1) ^ 1
            q = ql.to(tl.int32) - (qh.to(tl.int32) << 2)
            w = scaled * q.to(tl.float32)
        elif QTYPE == "Q4_K":
            base = q_ptr + block * 160
            group = local // 32
            d = tl.load((base + group * 2).to(tl.pointer_type(DECODE_DTYPE))).to(tl.float32)
            dm = tl.load((base + 16 + group * 2).to(tl.pointer_type(DECODE_DTYPE))).to(tl.float32)

            q = (tl.load(base + 32 + local // 2) >> (local % 2 * 4)) & 15
            product = (d * q.to(tl.float32)).to(DECODE_DTYPE).to(tl.float32)
            w = product - dm
        elif QTYPE == "Q5_K":
            base = q_ptr + block * 176
            d = tl.load(base.to(tl.pointer_type(tl.float16))).to(DECODE_DTYPE).to(tl.float32)
            dm = tl.load((base + 2).to(tl.pointer_type(tl.float16))).to(DECODE_DTYPE).to(tl.float32)
            group = local // 32
            within = local % 32
            byte = tl.load(base + 48 + group // 2 * 32 + within)
            high = tl.load(base + 16 + within)
            q = ((byte >> (group % 2 * 4)) & 15) | (((high >> group) & 1) << 4)

            si = group % 4
            slo = tl.load(base + 4 + si)
            mlo = tl.load(base + 8 + si)
            shi = tl.load(base + 12 + si)
            scale = tl.where(group < 4, slo & 63, (shi & 15) | ((slo >> 2) & 48))
            minimum = tl.where(group < 4, mlo & 63, (shi >> 4) | ((mlo >> 2) & 48))
            scaled = (d * scale.to(tl.float32)).to(DECODE_DTYPE).to(tl.float32)
            mins = (dm * minimum.to(tl.float32)).to(DECODE_DTYPE).to(tl.float32)
            product = (scaled * q.to(tl.float32)).to(DECODE_DTYPE).to(tl.float32)
            w = product - mins
        elif QTYPE == "Q6_K":
            base = q_ptr + block * 210
            d = tl.load((base + 208).to(tl.pointer_type(tl.float16))).to(DECODE_DTYPE).to(tl.float32)
            group = local // 32 % 4
            within = local % 32
            ql = tl.load(base + local // 128 * 64 + group % 2 * 32 + within)
            qh = tl.load(base + 128 + local // 128 * 32 + within)
            q = ((ql >> (group // 2 * 4)) & 15) | (((qh >> (group * 2)) & 3) << 4)
            scale = tl.load(base + 192 + local // 16).to(tl.int8)
            scaled = (d * scale.to(tl.float32)).to(DECODE_DTYPE).to(tl.float32)
            w = scaled * (q.to(tl.float32) - 32)
        else:
            raise NotImplementedError

        tl.store(w_ptr + block * 256 + local, w.to(DECODE_DTYPE))


def try_dequantize(tensor: "ParameterGGUF", input: torch.Tensor):
    if triton is None or not tensor.baked or tensor.device != input.device or not _supported_device(input.device):
        return None
    if input.dtype not in (torch.float16, torch.bfloat16) or tensor.computation_dtype not in (torch.float16, torch.bfloat16):
        return None

    quant = tensor.gguf_cls
    qtype = quant.qtype.name
    shape = tensor.shape
    if len(shape) != 2 or input.ndim < 2 or input.shape[-1] != shape[1] or not shape[1] or shape[1] % 256:
        return None

    rows = input.numel() // shape[1]

    if qtype == "Q2_K":
        supported, block_bytes = 0 < rows <= 8192, 84
    elif qtype == "Q3_K":
        supported, block_bytes = 0 < rows <= 8192, 110
    elif qtype == "Q4_K":
        supported, block_bytes = 0 < rows <= 8192, 160
    elif qtype == "Q5_K":
        supported, block_bytes = 0 < rows <= 1024 and tuple(shape) in {(4096, 4096), (16384, 4096)}, 176
    elif qtype == "Q6_K":
        supported, block_bytes = 0 < rows <= 1024 and tuple(shape) == (4096, 16384), 210
    else:
        return None
    route = (input.device, qtype, tensor.computation_dtype, input.dtype)
    if not supported or route in _failed_routes:
        return None

    data = tensor.data
    blocks = shape[0] * shape[1] // 256
    if not blocks or data.dtype != torch.uint8 or data.shape != (blocks, block_bytes) or not data.is_contiguous():
        return None

    weight = torch.empty(shape, device=input.device, dtype=input.dtype)
    decode_dtype = tl.float16 if tensor.computation_dtype == torch.float16 else tl.bfloat16

    try:
        _decode[(blocks,)](data, weight, qtype, decode_dtype, num_warps=4, num_stages=1, enable_fp_fusion=False)
    except torch.OutOfMemoryError:
        raise
    except Exception:
        _failed_routes.add(route)
        logger.warning("Triton %s decoding failed on %s, using PyTorch dequantization", qtype, input.device, exc_info=True)
        return None

    return weight
