import threading
import time
from collections import defaultdict

import torch

from backend import memory_management


class MemUsageMonitor(threading.Thread):
    run_flag = None
    device = None
    disabled = False
    opts = None
    data = None

    def __init__(self, name, device, opts):
        threading.Thread.__init__(self)
        self.name = name
        self.device = device
        self.opts = opts
        self.device_type = self._get_device_type(device)

        self.daemon = True
        self.run_flag = threading.Event()
        self.data = defaultdict(int)

        if self.device_type is None:
            self.disabled = True
            return

        try:
            self._mem_get_info()
            self._memory_stats()
        except Exception as e:
            print(f"Warning: caught exception '{e}', memory monitor disabled")
            self.disabled = True

    def _get_device_type(self, device):
        if memory_management.is_device_cuda(device):
            return "cuda"
        if memory_management.is_device_xpu(device) and hasattr(torch, "xpu"):
            return "xpu"
        return None

    def _mem_get_info(self):
        if self.device_type == "cuda":
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            return torch.cuda.mem_get_info(index)
        if self.device_type == "xpu":
            stats = torch.xpu.memory_stats(self.device)
            total = torch.xpu.get_device_properties(self.device).total_memory
            active = stats.get("active_bytes.all.current", 0)
            free = total - active
            return free, total
        raise RuntimeError("Unsupported device for memory monitor")

    def _memory_stats(self):
        if self.device_type == "cuda":
            return torch.cuda.memory_stats(self.device)
        if self.device_type == "xpu":
            return torch.xpu.memory_stats(self.device)
        return {}

    def _reset_peak_memory_stats(self):
        if self.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        elif self.device_type == "xpu" and hasattr(torch.xpu, "reset_peak_memory_stats"):
            torch.xpu.reset_peak_memory_stats()

    def _memory_summary(self):
        if self.device_type == "cuda":
            return torch.cuda.memory_summary()
        if self.device_type == "xpu" and hasattr(torch.xpu, "memory_summary"):
            return torch.xpu.memory_summary()
        return None

    def run(self):
        if self.disabled:
            return

        while True:
            self.run_flag.wait()

            self._reset_peak_memory_stats()
            self.data.clear()

            if self.opts.memmon_poll_rate <= 0:
                self.run_flag.clear()
                continue

            self.data["min_free"] = self._mem_get_info()[0]

            while self.run_flag.is_set():
                free, total = self._mem_get_info()
                self.data["min_free"] = min(self.data["min_free"], free)

                time.sleep(1 / self.opts.memmon_poll_rate)

    def dump_debug(self):
        print(self, "recorded data:")
        for k, v in self.read().items():
            print(k, -(v // -(1024**2)))

        print(self, "raw torch memory stats:")
        tm = self._memory_stats()
        for k, v in tm.items():
            if "bytes" not in k:
                continue
            print("\t" if "peak" in k else "", k, -(v // -(1024**2)))

        summary = self._memory_summary()
        if summary:
            print(summary)

    def monitor(self):
        self.run_flag.set()

    def read(self):
        if not self.disabled:
            free, total = self._mem_get_info()
            self.data["free"] = free
            self.data["total"] = total

            torch_stats = self._memory_stats()
            self.data["active"] = torch_stats.get("active.all.current", torch_stats.get("active_bytes.all.current", 0))
            self.data["active_peak"] = torch_stats.get("active_bytes.all.peak", torch_stats.get("active.all.peak", 0))
            self.data["reserved"] = torch_stats.get("reserved_bytes.all.current", 0)
            self.data["reserved_peak"] = torch_stats.get("reserved_bytes.all.peak", 0)
            self.data["system_peak"] = total - self.data["min_free"]

        return self.data

    def stop(self):
        self.run_flag.clear()
        return self.read()
