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

        self.daemon = True
        self.run_flag = threading.Event()
        self.data = defaultdict(int)

        if memory_management.is_intel_xpu():
            self._backend = torch.xpu
        else:
            self._backend = torch.cuda

        try:
            self.cuda_mem_get_info()
            self._backend.memory_stats(self.device)
        except Exception as e:  # AMD or whatever
            memory_management.logger.warning(f'Caught Exception "{e}"\nMemory Monitor Disabled...')
            self.disabled = True

    def cuda_mem_get_info(self):
        index = self.device.index if self.device.index is not None else self._backend.current_device()
        return self._backend.mem_get_info(index)

    def run(self):
        if self.disabled:
            return

        while True:
            self.run_flag.wait()

            self._backend.reset_peak_memory_stats()
            self.data.clear()

            if self.opts.memmon_poll_rate <= 0:
                self.run_flag.clear()
                continue

            self.data["min_free"] = self.cuda_mem_get_info()[0]

            while self.run_flag.is_set():
                free, total = self.cuda_mem_get_info()
                self.data["min_free"] = min(self.data["min_free"], free)

                time.sleep(1 / self.opts.memmon_poll_rate)

    def dump_debug(self):
        print(self, "recorded data:")
        for k, v in self.read().items():
            print(k, -(v // -(1024**2)))

        print(self, "raw torch memory stats:")
        tm = self._backend.memory_stats(self.device)
        for k, v in tm.items():
            if "bytes" not in k:
                continue
            print("\t" if "peak" in k else "", k, -(v // -(1024**2)))

        print(self._backend.memory_summary())

    def monitor(self):
        self.run_flag.set()

    def read(self):
        if not self.disabled:
            free, total = self.cuda_mem_get_info()
            self.data["free"] = free
            self.data["total"] = total

            torch_stats = self._backend.memory_stats(self.device)
            self.data["active"] = torch_stats.get("active_bytes.all.current") or torch_stats.get("active.all.current") or 0
            self.data["active_peak"] = torch_stats.get("active_bytes.all.peak") or torch_stats.get("active.all.peak") or 0
            self.data["reserved"] = torch_stats["reserved_bytes.all.current"]
            self.data["reserved_peak"] = torch_stats["reserved_bytes.all.peak"]
            self.data["system_peak"] = total - self.data["min_free"]

        return self.data

    def stop(self):
        self.run_flag.clear()
        return self.read()
