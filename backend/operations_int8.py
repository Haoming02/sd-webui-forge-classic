"""
int8 GEMM for the Linear layers of a DiT. `torch._int_mm` reaches the dp4a / IMMA hardware, which is several
times faster than fp16 on every GPU without fp8 support. Weights are quantised once at load, per output
channel; activations per token at every call; a LoRA is added as a low-rank side branch, not merged.
"""

import logging
from functools import partial

import torch

from backend import memory_management
from backend.logging import setup_logger

logger = logging.getLogger("int8")
setup_logger(logger)

# models whose Linear GEMMs are wide enough for the quantise / rescale passes to pay for themselves
INT8_MODELS = ("IntegratedFluxTransformer2DModel", "IntegratedChromaTransformer2DModel", "NextDiT")
MIN_ROWS = 17  # torch._int_mm needs M > 16; a modulation GEMV is padded up to it
BLOCK_BYTES = 128 * 1024**2  # budget for one row block's int32 + fp32 temporaries
ARENA_BYTES = 512 * 1024**2  # the Windows allocator rounds ~50 MB blocks up by half; pack the weights instead


class ParameterInt8(torch.nn.Parameter):
    """One flat uint8 blob: the int8 weight transposed to [K, N], then the fp32 per-output-channel scale [N]"""

    def __new__(cls, data, *, real_shape=None, computation_dtype=None, requires_grad=False):
        return super().__new__(cls, data, requires_grad=False)

    def __init__(self, data, *, real_shape=None, computation_dtype=None, requires_grad=False):
        super().__init__()
        if real_shape is not None:
            self.real_shape = torch.Size(real_shape)
            self.computation_dtype = computation_dtype or torch.float16
            self.arena: torch.Tensor | None = None

    @property
    def shape(self):
        return self.real_shape

    def copy_with_data(self, data):
        new = ParameterInt8(data, real_shape=self.real_shape, computation_dtype=self.computation_dtype)
        new.arena = self.arena if data.data_ptr() == self.data.data_ptr() else None
        return new

    def detach(self):  # torch.nn.Parameter(p) keeps the subclass only if detach() returns it
        return self.copy_with_data(self.data.detach())

    def to(self, *args, **kwargs):  # the blob never changes dtype, only device
        kwargs.pop("dtype", None)
        args = tuple(a for a in args if not isinstance(a, torch.dtype))
        return self.copy_with_data(self.data.to(*args, **kwargs))

    def pin_memory(self, device=None):
        return self.copy_with_data(torch.Tensor.pin_memory(self, device=device))

    def w8(self) -> torch.Tensor:
        n, k = self.real_shape
        return self.data[: k * n].view(torch.int8).view(k, n)

    def scale(self) -> torch.Tensor:
        n, k = self.real_shape
        return self.data[k * n :].view(torch.float32)

    def dequantize(self, dtype=None) -> torch.Tensor:
        return (self.w8().t().to(torch.float32) * self.scale()[:, None]).to(dtype or self.computation_dtype)


def blob_size(n: int, k: int) -> int:
    return k * n + 4 * n


def quantize(weight: torch.Tensor, computation_dtype=torch.float16, out: torch.Tensor = None) -> ParameterInt8:
    w = weight.detach().to(torch.float32)
    s = w.abs().amax(dim=1).clamp_min_(1e-8) / 127.0
    q = torch.round(w * (1.0 / s)[:, None]).clamp_(-127, 127).to(torch.int8).t().contiguous()
    blob = torch.cat([q.view(-1).view(torch.uint8), s.view(torch.uint8)])
    if out is None:
        blob = blob.to("cpu")
    else:
        out.copy_(blob)
        blob = out
    return ParameterInt8(blob, real_shape=weight.shape, computation_dtype=computation_dtype)


def int_mm_works(device: torch.device) -> bool:
    try:
        a = torch.zeros(MIN_ROWS, 8, dtype=torch.int8, device=device)
        torch._int_mm(a, a.new_zeros(8, 8))
        return True
    except Exception as e:
        logger.warning(f"int8 GEMM unavailable on {device}: {e}")
        return False


def quantize_model(model: torch.nn.Module) -> int:
    name = type(model).__name__
    if name not in INT8_MODELS:
        logger.info(f"Not quantising {name}: its Linear layers are too small to gain from int8")
        return 0

    device = memory_management.get_torch_device()
    if not int_mm_works(device):
        return 0

    from backend.loader_gguf import dequantize
    from backend.operations import ForgeOperations, ForgeOperationsGGUF

    supported = (ForgeOperations.Linear, ForgeOperationsGGUF.Linear)  # the Linear classes whose forward we hook
    count, skipped, arena, at = 0, 0, None, 0

    for module in model.modules():
        if not isinstance(module, supported) or module.weight is None or module.weight.ndim != 2:
            continue
        n, k = module.weight.shape
        if n % 8 or k % 8:
            skipped += 1
            continue

        source = module.weight
        if getattr(source, "gguf_cls", None) is not None:
            source = dequantize(source.to(device), torch.float16)
        else:
            source = source.to(device=device, dtype=torch.float16)

        size = blob_size(n, k)
        if arena is None or at + size > arena.numel():
            arena, at = torch.empty(max(size, ARENA_BYTES), dtype=torch.uint8), 0
        weight = quantize(source, getattr(module.weight, "computation_dtype", torch.float16), out=arena[at : at + size])
        weight.arena = arena
        at += size

        module.weight = weight
        module.convert_weight = convert_weight  # the patcher merges a LoRA through these
        module.set_weight = partial(set_weight, module)
        count += 1

    logger.info(f"Quantised {count} Linear layers of {name} to int8 ({skipped} skipped)")
    return count


def convert_weight(weight: ParameterInt8, inplace=False) -> torch.Tensor:
    return weight.dequantize()


def set_weight(layer, out_weight: torch.Tensor, inplace_update=False, seed=None, return_weight=False):
    if return_weight:
        return out_weight
    layer.weight = quantize(out_weight, layer.weight.computation_dtype).to(out_weight.device)
    layer.__dict__.pop("_int8_lora", None)


def lora_entries(layer) -> list | None:
    """(strength, adapter, offset) for every plain up/down LoRA; None if anything needs the merge instead"""
    entries = []
    for fn in layer.weight_function:
        if hasattr(fn, "patches"):  # LowVramPatch
            patches = fn.patches.get(fn.key, [])
        elif hasattr(fn, "patch"):  # OnlineLoRAPatch
            patches = fn.patch
        else:
            return None
        for strength, v, strength_model, offset, function in patches:
            weights = getattr(v, "weights", None)
            if getattr(v, "name", None) != "lora" or weights is None or function is not None or strength_model != 1.0:
                return None
            if weights[3] is not None or weights[4] is not None or weights[5] is not None:  # mid / dora / reshape
                return None
            entries.append((strength, v, offset))
    return entries


def lora_lowrank(layer, x2: torch.Tensor, entries: list) -> list:
    """x @ down^T for every LoRA: [tokens, rank], small enough to keep whole; up^T is applied per block"""
    sig = tuple((id(v), float(strength)) for strength, v, _ in entries)
    cached = layer.__dict__.get("_int8_lora")
    if cached is None or cached[0] != sig:
        mats = []
        for strength, v, offset in entries:
            up, down, alpha = v.weights[0], v.weights[1], v.weights[2]
            scale = strength * (alpha / down.shape[0] if alpha is not None else 1.0)
            down_t = down.flatten(start_dim=1).to(device=x2.device, dtype=x2.dtype).t().contiguous()
            up_t = (up.flatten(start_dim=1).to(device=x2.device, dtype=torch.float32) * scale).to(x2.dtype).t().contiguous()
            mats.append((down_t, up_t, offset))
        layer._int8_lora = cached = (sig, mats)

    low = []
    for down_t, up_t, offset in cached[1]:
        src = x2 if offset is None or offset[0] == 0 else x2[:, offset[1] : offset[1] + offset[2]]
        low.append((src @ down_t, up_t, offset))
    return low


def int8_linear(x: torch.Tensor, weight: ParameterInt8, bias: torch.Tensor, layer, lora: list) -> torch.Tensor:
    """
    Row blocks: _int_mm gives int32 and the rescale needs fp32, so holding the whole output costs 8 bytes per
    element against the 2 of the fp16 result - at hires that is over a gigabyte for one layer.
    """
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    rows, n = x2.shape[0], weight.real_shape[0]
    w8, s_w = weight.w8(), weight.scale()
    low = lora_lowrank(layer, x2, lora) if lora else ()

    out = torch.empty(rows, n, dtype=x.dtype, device=x.device)
    block = max(MIN_ROWS, BLOCK_BYTES // (8 * n))
    for i in range(0, rows, block):
        j = min(i + block, rows)
        xb = x2[i:j]
        if j - i < MIN_ROWS:
            xb = torch.nn.functional.pad(xb, (0, 0, 0, MIN_ROWS - (j - i)))

        s_x = xb.abs().amax(dim=1).float().clamp_min_(1e-8) / 127.0
        x8 = torch.round(xb.float() * (1.0 / s_x)[:, None]).clamp_(-127, 127).to(torch.int8)
        ob = torch._int_mm(x8, w8)[: j - i] * s_x[: j - i, None]  # the int32 sum is exact, the scaling stays fp32
        ob.mul_(s_w[None, :])
        ob = ob.to(x.dtype)
        if bias is not None:
            ob += bias
        for lo, up_t, offset in low:
            if offset is None or offset[0] != 0:
                ob.addmm_(lo[i:j], up_t)
            else:  # a LoRA trained on one part of a fused weight (q / k / v of a qkv)
                ob[:, offset[1] : offset[1] + offset[2]].addmm_(lo[i:j], up_t)
        out[i:j] = ob

    return out.reshape(*shape[:-1], n)


def forward(layer, x: torch.Tensor) -> torch.Tensor:
    from backend.operations import main_stream_worker, weights_manual_cast

    lora = lora_entries(layer) if layer.weight_function else []
    functions, layer.weight_function = layer.weight_function, []  # the blob goes through the cast untouched
    try:
        weight, bias, signal = weights_manual_cast(layer, x, skip_weight_dtype=True)
    finally:
        layer.weight_function = functions

    with main_stream_worker(weight, bias, signal):
        if lora is not None:
            return int8_linear(x, weight, bias, layer, lora)
        w = weight.dequantize(x.dtype)  # a patch the side branch cannot express: fp16 with the merge
        for f in functions:
            w = f(w)
        return torch.nn.functional.linear(x, w, bias)
