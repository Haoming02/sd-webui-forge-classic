"""Report a quantized checkpoint's layer formats (int8_tensorwise / convrot / fp8 / etc).

Answers "what will the dynamic LoRA path + which kernels apply" for a given ckpt.
Run: python tests/dynamic_lora/inspect_ckpt.py path/to/model.safetensors
"""

import collections
import json
import sys
from pathlib import Path

from safetensors import safe_open


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path or not Path(path).exists():
        print("usage: python tests/dynamic_lora/inspect_ckpt.py <model.safetensors>")
        raise SystemExit(2)

    with safe_open(path, framework="pt") as st:
        keys = list(st.keys())
        print("total keys:", len(keys))
        cq = [k for k in keys if k.endswith("comfy_quant")]
        print("quantized layers (comfy_quant):", len(cq))
        fmts = collections.Counter()
        with safe_open(path, framework="pt") as st:
            for k in cq:
                try:
                    fmts[json.loads(st.get_tensor(k).numpy().tobytes()).get("format")] += 1
                except Exception:
                    fmts["<unreadable>"] += 1
        print("formats:", dict(fmts))
        sample = [k for k in keys if k.endswith(".weight") and ("diffusion_model" in k or "model." in k)][:3]
        print("sample weight keys:", sample)


if __name__ == "__main__":
    main()
