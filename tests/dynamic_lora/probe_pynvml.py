"""pynvml budget probe: shows what --pynvml does to forge's free-memory math.

Simulates an external VRAM consumer (llama.cpp etc.) by committing memory in a
CHILD process (touch the pages! - untouaced torch.empty pages are lazily
committed on Windows and invisible to NVML, which invalidates the measurement),
then evaluates forge's get_free_memory() formula with and without the pynvml
correction.

Key finding (4070S + llama ~6GB, 0.5 fraction):
  - WITHOUT pynvml: torch mem_get_info LIES (reports ~11.6 GB free) -> webui
    over-commits -> silent WDDM sysmem spill -> erratic iteration times.
  - WITH pynvml: reports reality (~4-5 GB) -> but combined with
    setting_allocated_vram=0.5 the same headroom is double-counted
    (external once via NVML, once via the fraction) -> over-conservative.
  => correct pairing: --pynvml + setting_allocated_vram=1.0

Run: python tests/dynamic_lora/probe_pynvml.py [external_gb]
"""

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CHILD = """
import sys, time
import torch
gb = float(sys.argv[1])
ext = torch.empty(int(gb * 1024**3), dtype=torch.uint8, device="cuda")
ext.fill_(1)  # touch pages so NVML accounts them
torch.cuda.synchronize()
print("committed", flush=True)
time.sleep(90)
"""


def main():
    gb = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    child = subprocess.Popen([sys.executable, "-c", CHILD, str(gb)], stdout=subprocess.PIPE, text=True)
    if child.stdout.readline().strip() != "committed":
        child.kill()
        raise RuntimeError("child failed to commit VRAM (is CUDA available?)")
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
        my_reserved = torch.cuda.memory_reserved()
        print(f"[webui-like parent] total {round(total/1e6)} | mem_get_info.free {round(free/1e6)} | my reserved {round(my_reserved/1e6)}")

        import pynvml

        pynvml.nvmlInit()
        nv = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
        print(f"[nvml] total {round(nv.total/1e6)} used {round(nv.used/1e6)} free {round(nv.free/1e6)}")

        external = nv.used - my_reserved
        print("forge get_free WITHOUT pynvml:", round(free / 1e6), "MB")
        print("forge get_free WITH  pynvml:  ", round((free - external) / 1e6), "MB")
        print("(compare against setting_allocated_vram fraction before trusting either)")
    finally:
        child.kill()


if __name__ == "__main__":
    main()
