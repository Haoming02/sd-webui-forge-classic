import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class QuantOpsRuntimeTests(unittest.TestCase):
    def test_quant_ops_disables_triton_on_xpu(self):
        quant_ops = importlib.import_module("backend.quant_ops")

        with mock.patch.object(quant_ops.torch.xpu, "is_available", return_value=True):
            self.assertTrue(quant_ops.should_disable_triton_backend())

        with mock.patch.object(quant_ops.torch.xpu, "is_available", return_value=False):
            self.assertFalse(quant_ops.should_disable_triton_backend())


class MemmonRuntimeTests(unittest.TestCase):
    @staticmethod
    def load_memmon_module():
        module_name = "tests.memmon_under_test"
        module_path = Path("modules/memmon.py")

        backend_module = types.ModuleType("backend")
        memory_management_module = types.SimpleNamespace(
            is_intel_xpu=lambda: False,
            logger=types.SimpleNamespace(warning=lambda *args, **kwargs: None),
        )
        backend_module.memory_management = memory_management_module

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)

        previous_backend = sys.modules.get("backend")
        sys.modules["backend"] = backend_module
        try:
            assert spec.loader is not None
            spec.loader.exec_module(module)
        finally:
            if previous_backend is None:
                sys.modules.pop("backend", None)
            else:
                sys.modules["backend"] = previous_backend

        return module

    def test_memmon_stat_lookup_falls_back_to_alternate_keys(self):
        memmon = self.load_memmon_module()
        monitor = memmon.MemUsageMonitor.__new__(memmon.MemUsageMonitor)

        stats = {"active_bytes.all.current": 12}

        self.assertEqual(
            monitor.get_stat_value(stats, "active.all.current", "active_bytes.all.current"),
            12,
        )
        self.assertEqual(monitor.get_stat_value(stats, "missing"), 0)


if __name__ == "__main__":
    unittest.main()
