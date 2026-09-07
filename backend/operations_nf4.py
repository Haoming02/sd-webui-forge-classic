import json

import torch

# bitsandbytes 4-bit layout (`QuantState.as_dict(packed=True)`): `<key>` uint8 [numel/2, 1] two codes per byte (high nibble first),
# `<key>.absmax` float32 per block, `<key>.quant_map` float32 [16] codebook, `<key>.quant_state.bitsandbytes__nf4` JSON
# with blocksize / shape (+ nested_* when the absmax is itself 8-bit quantized)

QUANT_STATE_KEYS = ("quant_state.bitsandbytes__nf4", "quant_state.bitsandbytes__fp4")


class ParameterNF4(torch.nn.Parameter):
    # one uint8 buffer: packed weights followed by the absmax bytes, so that moving and pinning see a single tensor (like ParameterGGUF)

    def __init__(self, torch_tensor, *, real_shape=None, blocksize=64, code=None, packed_bytes=0, nested=None, no_init=False):
        super().__init__()
        if no_init:
            return

        self.real_shape: torch.Size = torch.Size(real_shape)
        self.blocksize: int = blocksize
        self.code: torch.Tensor = code  # float32 [16]
        self.packed_bytes: int = packed_bytes
        self.nested: dict | None = nested  # {"code": float32 [256], "blocksize": int, "offset": float, "absmax_bytes": int}
        self.computation_dtype = torch.float16

    def __new__(cls, torch_tensor, *, real_shape=None, blocksize=64, code=None, packed_bytes=0, nested=None, no_init=False):
        return super().__new__(cls, torch_tensor, requires_grad=False)

    @property
    def shape(self):
        return self.real_shape

    def copy_with_data(self, data):
        new = ParameterNF4(data, no_init=True)
        new.real_shape = self.real_shape
        new.blocksize = self.blocksize
        new.code = self.code.to(data.device)
        new.packed_bytes = self.packed_bytes
        new.nested = None if self.nested is None else {**self.nested, "code": self.nested["code"].to(data.device)}
        new.computation_dtype = self.computation_dtype
        return new

    def to(self, *args, **kwargs):
        device, _, non_blocking, _ = torch._C._nn._parse_to(*args, **{k: v for k, v in kwargs.items() if k != "copy"})
        return self.copy_with_data(self.data.to(device=device, non_blocking=non_blocking, copy=kwargs.get("copy", False)))

    def pin_memory(self, device=None):
        return self.copy_with_data(torch.Tensor.pin_memory(self, device=device))


def load_nf4_parameter(state_dict: dict, key: str, device: torch.device, computation_dtype: torch.dtype) -> ParameterNF4 | None:
    qs_key = next((f"{key}.{k}" for k in QUANT_STATE_KEYS if f"{key}.{k}" in state_dict), None)
    if qs_key is None:
        return None

    meta = json.loads(bytes(state_dict[qs_key].tolist()).decode())
    packed = state_dict[key].reshape(-1)
    absmax = state_dict[f"{key}.absmax"]
    nested = None

    def aligned(*parts):  # float32 sections must start at a multiple of 4 bytes
        out = []
        for part in parts:
            if out and (pad := -sum(x.numel() for x in out) % 4):
                out.append(torch.zeros(pad, dtype=torch.uint8, device=part.device))
            out.append(part.reshape(-1).view(torch.uint8))
        return torch.cat(out)

    if "nested_absmax" in meta or f"{key}.nested_absmax" in state_dict:
        nested = {
            "code": state_dict[f"{key}.nested_quant_map"].to(torch.float32),
            "blocksize": int(meta["nested_blocksize"]),
            "offset": float(meta["nested_offset"]),
            "absmax_bytes": absmax.numel(),
        }
        buffer = aligned(packed, absmax, state_dict[f"{key}.nested_absmax"].to(torch.float32))
    else:
        buffer = aligned(packed, absmax.to(torch.float32))

    param = ParameterNF4(
        buffer.to(device),
        real_shape=meta["shape"],
        blocksize=int(meta["blocksize"]),
        code=state_dict[f"{key}.quant_map"].to(device=device, dtype=torch.float32),
        packed_bytes=packed.numel(),
        nested=nested,
    )
    param.computation_dtype = computation_dtype
    return param


def with_4bit_shapes(state_dict: dict) -> dict:
    # for architecture detection: replace each packed [numel/2, 1] weight by a meta tensor of its real shape
    keys = [k for k in state_dict if any(k.endswith(q) for q in QUANT_STATE_KEYS)]
    if not keys:
        return state_dict

    sd = dict(state_dict)
    for k in keys:
        meta = json.loads(bytes(state_dict[k].tolist()).decode())
        sd[k.split(".weight.")[0] + ".weight"] = torch.empty(meta["shape"], dtype=torch.bfloat16, device="meta")
    return sd


def _blockwise(values: torch.Tensor, absmax: torch.Tensor, blocksize: int) -> torch.Tensor:
    full = (values.numel() // blocksize) * blocksize
    out = values[:full].view(-1, blocksize) * absmax[: full // blocksize].view(-1, 1)
    if full < values.numel():
        out = torch.cat([out.reshape(-1), values[full:] * absmax[full // blocksize]])
    return out.reshape(-1)


def dequantize_nf4(weight: torch.Tensor) -> torch.Tensor:
    if not isinstance(weight, ParameterNF4):
        return weight

    data = weight.data
    dtype = weight.computation_dtype
    packed = data[: weight.packed_bytes]

    align = lambda n: (n + 3) // 4 * 4
    if weight.nested is None:
        absmax = data[align(weight.packed_bytes) :].view(torch.float32)
    else:
        start = align(weight.packed_bytes)
        n = weight.nested["absmax_bytes"]
        codes = data[start : start + n]
        absmax2 = data[align(start + n) :].view(torch.float32)
        absmax = _blockwise(weight.nested["code"][codes.long()], absmax2, weight.nested["blocksize"]) + weight.nested["offset"]

    # bf16 has too few mantissa bits for the blockwise multiply (4e-3 relative error vs 2e-4 in fp16); use fp32 like bitsandbytes
    math_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
    code = weight.code.to(math_dtype)
    byte = torch.arange(256, device=data.device)
    table = torch.stack([code[byte >> 4], code[byte & 0x0F]], dim=1)
    values = table.index_select(0, packed.int()).view(-1)[: weight.real_shape.numel()]

    return _blockwise(values, absmax.to(math_dtype), weight.blocksize).reshape(weight.real_shape).to(dtype)
