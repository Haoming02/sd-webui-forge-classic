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

        try:
            self.device_mem_get_info()
            self.device_memory_stats(self.device)
        except Exception as e:  # AMD or whatever
            print(f"Warning: caught exception '{e}', memory monitor disabled")
            self.disabled = True

    def device_mem_get_info(self):
        if memory_management.is_device_xpu(self.device):
            stats = torch.xpu.memory_stats(self.device)
            reserved = stats["reserved_bytes.all.current"]
            total = torch.xpu.get_device_properties(self.device).total_memory
            return total - reserved, total

        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        return torch.cuda.mem_get_info(index)

    def device_memory_stats(self, device):
        if memory_management.is_device_xpu(device):
            return torch.xpu.memory_stats(device)

        return torch.cuda.memory_stats(device)

    def reset_peak_memory_stats(self):
        if memory_management.is_device_xpu(self.device):
            torch.xpu.reset_peak_memory_stats(self.device)
            return

        torch.cuda.reset_peak_memory_stats()

    def memory_summary(self):
        if memory_management.is_device_xpu(self.device):
            return torch.xpu.memory_summary(self.device)

        return torch.cuda.memory_summary()

    def get_stat_value(self, stats, *keys):
        for key in keys:
            if key in stats:
                return stats[key]
        return 0

    def run(self):
        if self.disabled:
            return

        while True:
            self.run_flag.wait()

            self.reset_peak_memory_stats()
            self.data.clear()

            if self.opts.memmon_poll_rate <= 0:
                self.run_flag.clear()
                continue

            self.data["min_free"] = self.device_mem_get_info()[0]

            while self.run_flag.is_set():
                free, total = self.device_mem_get_info()
                self.data["min_free"] = min(self.data["min_free"], free)

                time.sleep(1 / self.opts.memmon_poll_rate)

    def dump_debug(self):
        print(self, "recorded data:")
        for k, v in self.read().items():
            print(k, -(v // -(1024**2)))

        print(self, "raw torch memory stats:")
        tm = self.device_memory_stats(self.device)
        for k, v in tm.items():
            if "bytes" not in k:
                continue
            print("\t" if "peak" in k else "", k, -(v // -(1024**2)))

        print(self.memory_summary())

    def monitor(self):
        self.run_flag.set()

    def read(self):
        if not self.disabled:
            free, total = self.device_mem_get_info()
            self.data["free"] = free
            self.data["total"] = total

            torch_stats = self.device_memory_stats(self.device)
            self.data["active"] = self.get_stat_value(torch_stats, "active.all.current", "active_bytes.all.current")
            self.data["active_peak"] = self.get_stat_value(torch_stats, "active_bytes.all.peak", "active.all.peak")
            self.data["reserved"] = self.get_stat_value(torch_stats, "reserved_bytes.all.current")
            self.data["reserved_peak"] = self.get_stat_value(torch_stats, "reserved_bytes.all.peak")
            self.data["system_peak"] = total - self.data["min_free"]

        return self.data

    def stop(self):
        self.run_flag.clear()
        return self.read()
