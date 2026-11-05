#!/usr/bin/env python3
"""Pure-CPU mock of the patched BEAM Ascend DSMI controller (no NPU writes).

Place in parsa/tests/ and run from parsa root:
    python3 -m unittest tests.test_beam_npu_clock_mock -v

Optional override: BEAM_VLLM_SOURCE=/absolute/path/to/vllm-beam-npu-0.23.0
"""

import ctypes
import importlib.util
import os
import unittest
from pathlib import Path

SOURCE = Path(os.environ.get(
    "BEAM_VLLM_SOURCE",
    Path(__file__).resolve().parents[1] / "setup" / "vllm-beam-npu-0.23.0",
))
MODULE_PATH = SOURCE / "vllm" / "v1" / "worker" / "beam_ascend.py"


def load_module():
    if not MODULE_PATH.is_file():
        raise FileNotFoundError(f"BEAM NPU controller not found: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("beam_npu_clock_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeDsmi:
    def __init__(self, module, ignore_writes=False):
        self.module = module
        self.rated = 1800
        self.current = 1800
        self.ignore_writes = ignore_writes
        self.calls = []

    def dsmi_get_device_count(self, ptr):
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int)).contents.value = 1
        return 0

    def dsmi_list_device(self, devices, count):
        assert count == 1
        devices[0] = 0
        return 0

    def dsmi_get_device_frequency(self, device_id, kind, ptr):
        assert (device_id, kind) == (0, 9)
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint)).contents.value = self.rated
        return 0

    def dsmi_get_aicore_info(self, device_id, ptr):
        assert device_id == 0
        info = ctypes.cast(ptr, ctypes.POINTER(self.module._AicoreInfo)).contents
        info.freq = self.rated
        info.curfreq = self.current
        return 0

    def dsmi_get_device_power_info(self, device_id, ptr):
        assert device_id == 0
        info = ctypes.cast(ptr, ctypes.POINTER(self.module._PowerInfo)).contents
        info.power = 930  # 93.0 W
        return 0

    def dsmi_set_device_info(self, device_id, kind, subkind, ptr, length):
        assert (device_id, kind, subkind) == (0, 8, 12)
        assert length == ctypes.sizeof(self.module._StressPayload)
        payload = ctypes.cast(ptr, ctypes.POINTER(self.module._StressPayload)).contents
        requested = int(payload.soc_stress_cfg.value)
        self.calls.append(requested)
        if not self.ignore_writes:
            self.current = requested
        return 0


class TestBeamNpuClockMock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_actuation_and_restoration(self):
        backend = FakeDsmi(self.module)
        ctl = self.module.AscendClockController(0, backend=backend)
        self.assertIsNone(ctl.set_clock(None))
        self.assertEqual(backend.calls, [])
        self.assertEqual(ctl.set_clock(1500), 1500)
        self.assertEqual(ctl.set_clock(1650), 1650)
        self.assertEqual(ctl.get_clock_mhz(), 1650)
        self.assertAlmostEqual(ctl.get_power_w(), 93.0)
        ctl.close()
        self.assertEqual(backend.calls, [1500, 1650, 1800])
        self.assertEqual(backend.current, 1800)

    def test_out_of_catalog_rejected(self):
        backend = FakeDsmi(self.module)
        ctl = self.module.AscendClockController(0, backend=backend)
        with self.assertRaises(ValueError):
            ctl.set_clock(1800)
        self.assertEqual(backend.calls, [])

    def test_successful_dsmi_status_does_not_mean_clock_changed(self):
        backend = FakeDsmi(self.module, ignore_writes=True)
        ctl = self.module.AscendClockController(0, backend=backend)
        observed = ctl.set_clock(1500)
        self.assertEqual(observed, 1800)
        self.assertNotEqual(observed, 1500)
        ctl.close()
        # Demonstrates why requested/observed logging is necessary.
        self.assertEqual(backend.calls, [1500, 1800])


if __name__ == "__main__":
    unittest.main(verbosity=2)
