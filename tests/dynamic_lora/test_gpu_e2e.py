"""Dynamic LoRA GPU end-to-end: real int8_tensorwise-quantized MixedPrecision Linear.

Proves: quantized weights stay quantized (no dequant round-trip), weight_function
stays empty (quantized fast path preserved), and the low-rank correction equals a
dense reference within int8/bf16 noise. Run: python tests/dynamic_lora/test_gpu_e2e.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

import backend.operations  # noqa: F401  (module init order)
from backend.operations_mixed_precision import mixed_precision_ops
from backend.patcher.base import ModelPatcher, wipe_dynamic_lora
from backend.patcher.lora import weight_adapter
from backend.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class

if not torch.cuda.is_available():
    print("SKIP: no CUDA")
    raise SystemExit(0)

dev = torch.device("cuda:0")
torch.manual_seed(1)

ops = mixed_precision_ops({}, torch.bfloat16)
in_f, out_f, rank = 64, 32, 8
qformat = "int8_tensorwise"
qconf = QUANT_ALGOS[qformat]
layout = get_layout_class(qconf["comfy_tensor_layout"])
QJSON = torch.tensor(list(json.dumps({"format": qformat}).encode("utf-8")), dtype=torch.uint8)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = ops.Linear(in_f, out_f)
        self.computation_dtype = torch.bfloat16


def main():
    net = Net()
    Wf = torch.randn(out_f, in_f, device=dev)
    qdata, params = layout.quantize(Wf)
    net.lin._orig_shape = (out_f, in_f)
    net.lin.load_state_dict({"weight": qdata, "weight_scale": params.scale.float(), "comfy_quant": QJSON, "bias": torch.randn(out_f, device=dev, dtype=torch.bfloat16)})
    assert isinstance(net.lin.weight, QuantizedTensor), "layer should be quantized"

    p = ModelPatcher(net, dev, torch.device("cpu"))
    up, down = torch.randn(out_f, rank), torch.randn(rank, in_f)
    alpha, strength = 2.0 * rank, 0.8
    p.add_patches({"lin.weight": weight_adapter.LoRAAdapter(set(), (up, down, alpha, None, None, None))}, strength_patch=strength, filename="f", online_mode=True)
    assert p.dynamic_loras and not p.weight_wrapper_patches, "patch should take the dynamic path"
    p.patch_model(device_to=dev, load_weights=True)

    lin = p.model.lin
    assert isinstance(lin.weight, QuantizedTensor), "weights must stay quantized after load"
    assert len(lin.weight_function) == 0, "no wrapper functions may be registered"
    assert hasattr(lin, "_dyn_down") and lin._dyn_down.device.type == "cuda", "dynamic tensors built on GPU"

    x = torch.randn(4, 5, in_f, device=dev, dtype=torch.bfloat16)
    out = lin(x)
    wipe_dynamic_lora(lin)
    base = lin(x)

    delta_expected = strength * alpha / rank * (x.reshape(-1, in_f).float() @ down.T.to(dev).float() @ up.T.to(dev).float())
    delta_actual = (out.float() - base.float()).reshape(-1, out_f)
    rel = (delta_actual - delta_expected).abs().max().item() / delta_expected.abs().max().item()
    print(f"delta rel_err: {rel:.4f}")
    assert rel < 0.02, "low-rank correction must match dense reference"

    Wd = lin.weight.dequantize().float()
    ref = torch.nn.functional.linear(x.float(), (Wd + strength * alpha / rank * (up.to(dev) @ down.to(dev))).bfloat16().float(), lin.bias.float())
    rel_full = (out.float().reshape(-1, out_f) - ref.reshape(-1, out_f)).abs().max().item() / ref.abs().max().item()
    print(f"full rel_err vs dequantized dense ref: {rel_full:.4f}")
    print("GPU E2E OK")


if __name__ == "__main__":
    main()
