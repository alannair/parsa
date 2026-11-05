# Copyright (C) 2026 M.R. Siavash Katebzadeh <mr@katebzadeh.xyz>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import energy_meter


class EnergyMonitorCadenceTests(unittest.TestCase):
    def test_monitor_waits_only_for_remaining_period(self):
        self._run_monitor(0.05, [0.0, 0.02, 0.05, 0.07, 0.10], [0.03, 0.03])

    def test_monitor_custom_period_and_overrun_do_not_catch_up(self):
        self._run_monitor(0.1, [0.0, 0.12, 0.12, 0.25, 0.25], [])

    def _run_monitor(self, interval, clock_values, expected_sleeps):
        from unittest.mock import mock_open

        clock = iter(clock_values)
        sleeps = []
        snapshots = []

        def sleep(value):
            sleeps.append(value)

        def take_snapshot(devices):
            snapshots.append(devices)
            if len(snapshots) > 2:
                raise KeyboardInterrupt
            return {}

        with patch.object(energy_meter, "_warmup_freq_reads"), patch.object(
            energy_meter, "snapshot", side_effect=take_snapshot
        ), patch.object(energy_meter.time, "monotonic", side_effect=lambda: next(clock)), patch.object(
            energy_meter.time, "sleep", side_effect=sleep
        ), patch("builtins.open", mock_open()):
            self.assertEqual(energy_meter.main(["monitor", "--devices", "0", "--output", "/tmp/energy-test.jsonl",
                                                "--interval", str(interval)]), 0)
        self.assertEqual(len(snapshots), 3)
        self.assertEqual(len(sleeps), len(expected_sleeps))
        for actual, expected in zip(sleeps, expected_sleeps):
            self.assertAlmostEqual(actual, expected)


class EnergyMeterTests(unittest.TestCase):
    def test_load_freq_module_uses_src_package(self):
        freq_module = energy_meter._load_freq_module()
        self.assertEqual(freq_module.__name__, "src.freq")

    def test_sample_device_reads_freq_before_optional_npu_smi(self):
        class FakeAscendML:
            @staticmethod
            def get_info(device):
                return {"aic_freq": 1300, "aic_volt": 850}

        fake_freq = type("FakeFreq", (), {"AscendML": FakeAscendML})()
        with patch.object(energy_meter, "_load_freq_module", return_value=fake_freq):
            with patch.object(energy_meter, "_dsmi_metrics", return_value=None):
                with patch.object(energy_meter, "_npu_smi_power", return_value=None):
                    sample = energy_meter._sample_device(0)
        self.assertEqual(sample["frequency"], 1300.0)
        self.assertEqual(sample["volt"], 850.0)
        self.assertIsNone(sample["power_sample"])

    def test_npu_smi_power_returns_none_when_unavailable(self):
        with patch.object(energy_meter.shutil, "which", return_value=None):
            self.assertIsNone(energy_meter._npu_smi_power(0))

    def test_snapshot_json_includes_freq_volt_in_per_npu(self):
        class FakeAscendML:
            @staticmethod
            def get_info(device):
                return {"aic_freq": 1300, "aic_volt": 850}

        fake_freq = type("FakeFreq", (), {"AscendML": FakeAscendML})()
        with patch.object(energy_meter, "_load_freq_module", return_value=fake_freq):
            with patch.object(energy_meter, "_dsmi_metrics", return_value=None):
                with patch.object(energy_meter, "_npu_smi_power", return_value=None):
                    payload = energy_meter.snapshot([0])
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        parsed = json.loads(line)
        sample = parsed["per_npu"]["0"]
        self.assertEqual(sample["frequency"], 1300.0)
        self.assertEqual(sample["volt"], 850.0)


    def test_snapshot_uses_dsmi_sampler_not_legacy(self):
        metrics = {"power_w": 10.5, "frequency_mhz": 1300, "volt_mv": 850}
        with patch.object(energy_meter, "_sample_device_legacy") as legacy:
            with patch.object(energy_meter, "_dsmi_metrics", return_value=metrics) as dsmi:
                payload = energy_meter.snapshot([0, 1])
        legacy.assert_not_called()
        self.assertEqual(dsmi.call_count, 2)
        self.assertEqual(payload["per_npu"]["0"]["power_sample"], 10.5)
        self.assertEqual(payload["per_npu"]["1"]["frequency"], 1300.0)

    def test_legacy_sampler_uses_ascendml_and_npu_smi(self):
        class FakeAscendML:
            @staticmethod
            def get_info(device):
                return {"aic_freq": 1200, "aic_volt": 800}

        fake_freq = type("FakeFreq", (), {"AscendML": FakeAscendML})()
        with patch.object(energy_meter, "_load_freq_module", return_value=fake_freq):
            with patch.object(energy_meter, "_dsmi_metrics") as dsmi:
                with patch.object(
                    energy_meter,
                    "_npu_smi_power",
                    return_value="NPU Real-time Power(W)         : 92.3\n",
                ):
                    sample = energy_meter._sample_device_legacy(0)
        dsmi.assert_not_called()
        self.assertEqual(sample["frequency"], 1200.0)
        self.assertEqual(sample["power_sample"], 92.0)

    def test_metric_value_accepts_ctypes_like_values(self):
        class FakeUInt:
            def __init__(self, value):
                self.value = value

            def __int__(self):
                return self.value

        self.assertEqual(energy_meter._metric_value(FakeUInt(1400)), 1400.0)


if __name__ == "__main__":
    unittest.main()
