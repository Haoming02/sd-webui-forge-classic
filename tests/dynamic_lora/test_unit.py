"""Dynamic LoRA unit tests (CPU, no GPU/model files needed).

Covers: LoRA/LoKr/diff conversion to runtime low-rank form, multi-LoRA batching,
zero & live strength changes (lora-ctl scheduler contract), fallbacks to the
OnlineLoRAPatch wrapper for unsupported patch shapes, offline merge path,
clone/reload/unpatch lifecycle, and VRAM accounting.

Run: python tests/dynamic_lora/test_unit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

import backend.operations  # noqa: F401  (module init order, mirrors app bootstrap)
from backend import utils
from backend.operations_mixed_precision import mixed_precision_ops
from backend.patcher.base import ModelPatcher
from backend.patcher.lora import weight_adapter

torch.manual_seed(0)

W = torch.randn(6, 8)
b = torch.randn(6)
x = torch.randn(2, 3, 8)
up, down = torch.randn(6, 4), torch.randn(4, 8)
alpha, strength = 8.0, 0.7
adapter = weight_adapter.LoRAAdapter(set(), (up, down, alpha, None, None, None))
scale = strength * alpha / 4

fails = []


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ops = mixed_precision_ops({}, torch.float32)
        self.lin = ops.Linear(8, 6)
        self.computation_dtype = torch.float32


def fresh():
    net = Net()
    net.lin.load_state_dict({"weight": W.clone(), "bias": b.clone()})
    return net


def p_of(net):
    return ModelPatcher(net, torch.device("cpu"), torch.device("cpu"))


def load(p):
    p.patch_model(device_to=torch.device("cpu"), lowvram_model_memory=0, load_weights=True)


def n_wrap(p):
    return sum(len(v) for v in p.weight_wrapper_patches.values())


def check(name, out, ref):
    err = (out - ref).abs().max().item()
    ok = err < 2e-4
    print(f"{'PASS' if ok else 'FAIL'} {name}: max_err={err:.2e}")
    if not ok:
        fails.append(name)


def main():
    net = fresh()
    p = p_of(net)
    p.add_patches({"lin.weight": adapter}, strength_patch=strength, filename="f", online_mode=True)
    assert p.dynamic_loras and not n_wrap(p)
    load(p)
    assert hasattr(net.lin, "_dyn_down")
    check("lora-dynamic", p.model.lin(x), torch.nn.functional.linear(x, W + scale * (up @ down), b))

    net = fresh()
    p = p_of(net)
    up2, down2, s2, a2 = torch.randn(6, 2), torch.randn(2, 8), 1.3, 4.0
    p.add_patches({"lin.weight": adapter}, strength_patch=strength, online_mode=True)
    p.add_patches({"lin.weight": weight_adapter.LoRAAdapter(set(), (up2, down2, a2, None, None, None))}, strength_patch=s2, online_mode=True)
    load(p)
    check("lora-multi", p.model.lin(x), torch.nn.functional.linear(x, W + scale * (up @ down) + s2 * a2 / 2 * (up2 @ down2), b))

    net = fresh()
    p = p_of(net)
    p.add_patches({"lin.weight": adapter}, strength_patch=0.0, online_mode=True)
    assert len(p.dynamic_loras.get("lin.weight", [])) == 1
    load(p)
    check("zero-strength", p.model.lin(x), torch.nn.functional.linear(x, W, b))
    e = p.dynamic_loras["lin.weight"][0]
    e["strength"] = -1.0
    check("live-strength-neg", p.model.lin(x), torch.nn.functional.linear(x, W - alpha / 4 * (up @ down), b))
    e["strength"] = strength
    check("live-strength-restore", p.model.lin(x), torch.nn.functional.linear(x, W + scale * (up @ down), b))

    net = fresh()
    p = p_of(net)
    diff, st = torch.randn(6, 8), 0.5
    p.add_patches({"lin.weight": ("diff", (diff,))}, strength_patch=st, online_mode=True)
    assert p.dynamic_loras and not n_wrap(p)
    load(p)
    check("diff-dynamic", p.model.lin(x), torch.nn.functional.linear(x, W + st * diff, b))

    net = fresh()
    p = p_of(net)
    w1, w2, st = torch.randn(2, 4), torch.randn(3, 2), 0.9
    p.add_patches({"lin.weight": weight_adapter.LoKrAdapter(set(), (w1, w2, 4.0, None, None, None, None, None, None))}, strength_patch=st, online_mode=True)
    assert p.dynamic_loras
    load(p)
    check("lokr-dynamic", p.model.lin(x), torch.nn.functional.linear(x, W + st * torch.kron(w1, w2), b))

    net = fresh()
    p = p_of(net)
    w1a, w1b, w2a, w2b, st = torch.randn(2, 5), torch.randn(5, 4), torch.randn(3, 3), torch.randn(3, 2), 0.6
    p.add_patches({"lin.weight": weight_adapter.LoKrAdapter(set(), (None, None, 6.0, w1a, w1b, w2a, w2b, None, None))}, strength_patch=st, online_mode=True)
    load(p)
    check("lokr-decomposed", p.model.lin(x), torch.nn.functional.linear(x, W + st * (6.0 / w2b.shape[0]) * torch.kron(w1a @ w1b, w2a @ w2b), b))

    net = fresh()
    p = p_of(net)
    mid = torch.randn(1, 4, 2, 2)
    p.add_patches({"lin.weight": weight_adapter.LoRAAdapter(set(), (up, down, alpha, mid, None, None))}, strength_patch=strength, online_mode=True)
    assert not p.dynamic_loras and n_wrap(p) == 1
    print("PASS locon-mid-fallback")

    net = fresh()
    p = p_of(net)
    p.add_patches({"lin.weight": adapter}, strength_patch=strength, strength_model=0.8, online_mode=True)
    assert n_wrap(p) == 1 and not p.dynamic_loras
    load(p)
    check("strength-model-wrapper", p.model.lin(x), torch.nn.functional.linear(x, 0.8 * W + scale * (up @ down), b))

    net = fresh()
    p = p_of(net)
    p.add_patches({"lin.weight": adapter}, strength_patch=strength, online_mode=False)
    assert not p.dynamic_loras
    load(p)
    check("offline-merge", p.model.lin(x), torch.nn.functional.linear(x, W + scale * (up @ down), b))

    p2 = p.clone()
    p2.dynamic_loras.setdefault("lin.weight", []).append({"kind": "diff", "name": None, "strength": 0.1, "factor": 1.0, "tensors": (torch.ones(6, 8),)})
    p2.load(device_to=torch.device("cpu"), full_load=True)
    check("clone-reload", p2.model.lin(x), torch.nn.functional.linear(x, W + scale * (up @ down) + 0.1 * torch.ones(6, 8), b))
    p2.unpatch_model(torch.device("cpu"), unpatch_weights=True)
    assert not hasattr(p2.model.lin, "_dyn_down")
    print("PASS lifecycle")

    p3 = p_of(fresh())
    p3.add_patches({"lin.weight": adapter}, strength_patch=strength, online_mode=True)
    assert p3.has_online_lora()
    assert utils.nested_compute_size(p3.dynamic_loras, element_size=4) > 0
    print("PASS accounting")

    net = fresh()
    p = p_of(net)
    p.add_patches({"lin.bias": ("diff", (torch.randn(6),))}, strength_patch=1.0, online_mode=True)
    assert n_wrap(p) == 1 and not p.dynamic_loras
    print("PASS bias-fallback")

    print()
    if fails:
        print("FAILURES:", fails)
        raise SystemExit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
