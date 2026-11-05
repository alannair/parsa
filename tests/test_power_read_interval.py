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

"""Opt-in query benchmark (not sensor-refresh measurement).

Run with RUN_ASCEND_POWER_INTERVAL_TEST=1 ASCEND_POWER_TEST_DEVICE=0
python3 tests/test_power_read_interval.py -v. Query speed is not guaranteed
hardware telemetry refresh rate.
"""

import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


def _configuration(environment):
    if environment.get("RUN_ASCEND_POWER_INTERVAL_TEST") != "1":
        raise ValueError("Hardware worker requires RUN_ASCEND_POWER_INTERVAL_TEST=1")
    value = environment.get("ASCEND_POWER_TEST_DEVICE", "")
    if not value or not value.isdecimal():
        raise ValueError("ASCEND_POWER_TEST_DEVICE must be one explicit nonnegative physical device")
    return int(value)


def _measure(reader_name, reader, clock=time.perf_counter_ns, max_reads=10, duration_ns=2_000_000_000):
    try:
        reader()  # Warm-up is deliberately excluded.
    except Exception as exc:
        return {"backend": reader_name, "attempted_samples": 0, "successful_samples": 0,
                "errors": ["warm-up %s: %s" % (type(exc).__name__, str(exc)[:256])],
                "latency_ms": None, "read_start_interval_ms": None,
                "successful_reads_per_second": None}
    deadline = clock() + duration_ns
    starts, latencies, errors, attempted_starts, attempts = [], [], [], [], 0
    final_attempt_completion = None
    while attempts < max_reads and clock() < deadline:
        started = clock()
        attempted_starts.append(started)
        attempts += 1
        try:
            reader()
        except Exception as exc:
            errors.append("%s: %s" % (type(exc).__name__, str(exc)[:256]))
            completed = clock()
        else:
            completed = clock()
            starts.append(started)
            latencies.append(completed - started)
        final_attempt_completion = completed
    if latencies:
        def stats(values):
            return {"min": min(values) / 1e6, "median": statistics.median(values) / 1e6,
                    "max": max(values) / 1e6}
        intervals = stats([b - a for a, b in zip(starts, starts[1:])]) if len(starts) >= 2 else None
        elapsed = final_attempt_completion - attempted_starts[0]
        return {"backend": reader_name, "attempted_samples": attempts,
                "successful_samples": len(latencies), "errors": errors,
                "latency_ms": stats(latencies), "read_start_interval_ms": intervals,
                "successful_reads_per_second": len(latencies) * 1e9 / elapsed if elapsed > 0 else None}
    return {"backend": reader_name, "attempted_samples": attempts, "successful_samples": 0, "errors": errors,
            "latency_ms": None, "read_start_interval_ms": None, "successful_reads_per_second": None}


def _finite_nonnegative(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("reader did not return a finite nonnegative reading: %r" % (value,))
    return value


def _parse_power_output(text):
    if not isinstance(text, str):
        raise ValueError("npu-smi power output is unavailable")
    pattern = re.compile(
        r"^\s*\|?\s*(?:NPU\s+Real-time\s+Power|Power(?:\s+Dissipation)?)"
        r"(?P<label_unit>\s*\(\s*W\s*\))?\s*:\s*"
        r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?(?:inf|nan))"
        r"\s*(?P<value_unit>W)?\s*\|?\s*$", re.IGNORECASE)
    for line in text.splitlines():
        match = pattern.fullmatch(line)
        if match and (match.group("label_unit") or match.group("value_unit")):
            return _finite_nonnegative(float(match.group("value")))
    raise ValueError("npu-smi output has no explicit watt-valued power reading")


def _run_phases(device, energy_meter):
    def power():
        text = energy_meter._npu_smi_power(device)
        return _parse_power_output(text)

    phases = [_measure("npu-smi power query (existing _npu_smi_power)", power)]
    backend = "native DSMI metrics bundle (power/frequency/voltage query)"
    try:
        freq = energy_meter._load_freq_module()
        dsmi = freq.dsmi
        if dsmi is None or device not in dsmi.device_ids:
            raise RuntimeError("selected physical device is unavailable through DSMI")
        getter = dsmi.get_device_metrics

        def metrics():
            result = getter(device)
            if not isinstance(result, dict):
                raise ValueError("native DSMI metrics returned no bundle: %r" % (result,))
            _finite_nonnegative(result.get("power_w"))
            return result

        phases.append(_measure(backend, metrics))
    except (Exception, SystemExit) as exc:
        phases.append({"backend": backend, "attempted_samples": 0, "successful_samples": 0,
                       "errors": ["initialization %s: %s" % (type(exc).__name__, str(exc)[:256])],
                       "latency_ms": None, "read_start_interval_ms": None,
                       "successful_reads_per_second": None})
    return phases


def _worker(device):
    # Gate and explicit device validation precede imports and hardware access.
    configured = _configuration(os.environ)
    if device != configured:
        raise ValueError("worker device differs from explicit configuration")
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from src import energy_meter
    return _run_phases(device, energy_meter)


def _child_main(args):
    if len(args) != 2 or args[0] != "--_power_interval_worker":
        raise ValueError("invalid private worker invocation")
    try:
        phases = _worker(int(args[1]))
        failures = [phase for phase in phases if not phase["successful_samples"] or phase["errors"]]
        data = {"ok": not failures, "phases": phases}
        if failures:
            data["error"] = "; ".join("%s: %s" % (phase["backend"], "; ".join(phase["errors"]))
                                      for phase in failures)
    except BaseException as exc:
        data = {"ok": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:1024])}
    Path(os.environ["ASCEND_POWER_RESULT_FILE"]).write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    if not data["ok"]:
        raise RuntimeError(data["error"])


def _run_hardware():
    device = _configuration(os.environ)
    with tempfile.TemporaryDirectory(prefix="ascend-power-interval-") as directory:
        result = Path(directory) / "result.json"
        environment = dict(os.environ, ASCEND_POWER_RESULT_FILE=str(result))
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--_power_interval_worker", str(device)],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True, env=environment)
        try:
            process.wait(timeout=15)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("unreaped benchmark process group") from exc
            if process.returncode is None:
                raise RuntimeError("unreaped benchmark child")
            raise
        try:
            report = json.loads(result.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("hardware benchmark worker exited %d without a valid report: %s" %
                               (process.returncode, exc)) from exc
        print("DISCLAIMER: observed query timings are not guaranteed hardware telemetry refresh rate.")
        print(json.dumps(report, indent=2, allow_nan=False))
        if process.returncode != 0 or not report.get("ok"):
            raise RuntimeError("hardware benchmark worker failed: %s" % report.get("error", process.returncode))
        failed = [phase for phase in report["phases"] if phase["successful_samples"] == 0 or phase["errors"]]
        if failed:
            raise RuntimeError("benchmark phases had errors or no usable reads: %s" % json.dumps(failed))


@unittest.skipUnless(os.environ.get("RUN_ASCEND_POWER_INTERVAL_TEST") == "1", "Opt-in hardware test")
class PowerReadIntervalHardwareTests(unittest.TestCase):
    def test_read_intervals(self):
        _run_hardware()


class PowerReadIntervalUnitTests(unittest.TestCase):
    def test_native_bundle_replaces_legacy_frequency_and_is_initialized_once(self):
        getter = mock.Mock(return_value={"power_w": 90.3, "frequency_mhz": 900, "volt_mv": 750})
        freq = mock.Mock(dsmi=mock.Mock(device_ids=[4], get_device_metrics=getter),
                         AscendML=mock.Mock(get_info=mock.Mock(side_effect=AssertionError("legacy API used"))))
        meter = mock.Mock(_npu_smi_power=mock.Mock(return_value="NPU Real-time Power(W): 90.3"),
                          _load_freq_module=mock.Mock(return_value=freq))
        phases = _run_phases(4, meter)
        meter._load_freq_module.assert_called_once_with()
        self.assertEqual(getter.call_count, 11)  # One warm-up plus ten measured calls.
        self.assertEqual(meter._npu_smi_power.call_count, 11)
        self.assertEqual(phases[0]["successful_samples"], 10)
        self.assertEqual(phases[1]["successful_samples"], 10)
        self.assertIn("bundle", phases[1]["backend"])
        freq.AscendML.get_info.assert_not_called()

    def test_warmup_failure_is_local_and_preserves_other_phase_results(self):
        freq = mock.Mock(dsmi=mock.Mock(device_ids=[4], get_device_metrics=mock.Mock(
            return_value={"power_w": 90.0, "frequency_mhz": None, "volt_mv": None})))
        meter = mock.Mock(_npu_smi_power=mock.Mock(side_effect=OSError("smi warmup failed")),
                          _load_freq_module=mock.Mock(return_value=freq))
        phases = _run_phases(4, meter)
        self.assertEqual(phases[0]["attempted_samples"], 0)
        self.assertEqual(phases[0]["successful_samples"], 0)
        self.assertIsNone(phases[0]["latency_ms"])
        self.assertEqual(phases[1]["successful_samples"], 10)

    def test_native_power_failures_and_initialization_failure_keep_cli_phase(self):
        for native_result in (None, {"power_w": float("nan")}, {"power_w": -1}):
            with self.subTest(native_result=native_result):
                freq = mock.Mock(dsmi=mock.Mock(device_ids=[4], get_device_metrics=mock.Mock(return_value=native_result)))
                meter = mock.Mock(_npu_smi_power=mock.Mock(return_value="NPU Real-time Power(W): 90.0"),
                                  _load_freq_module=mock.Mock(return_value=freq))
                phases = _run_phases(4, meter)
                self.assertEqual(phases[0]["successful_samples"], 10)
                self.assertEqual(phases[1]["successful_samples"], 0)
                self.assertTrue(phases[1]["errors"])
                freq.AscendML.get_info.assert_not_called()

        meter = mock.Mock(_npu_smi_power=mock.Mock(return_value="NPU Real-time Power(W): 90.0"),
                          _load_freq_module=mock.Mock(side_effect=OSError("DSMI init failed")))
        phases = _run_phases(4, meter)
        self.assertEqual(phases[0]["successful_samples"], 10)
        self.assertEqual(phases[1]["successful_samples"], 0)
        self.assertIn("DSMI init failed", phases[1]["errors"][0])

    def test_child_failure_report_retains_successful_phase_results(self):
        phases = [{"backend": "CLI power", "successful_samples": 10, "errors": []},
                  {"backend": "native bundle", "successful_samples": 0, "errors": ["missing power"]}]
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            with mock.patch.dict(os.environ, {"ASCEND_POWER_RESULT_FILE": str(result_path)}), \
                    mock.patch(__name__ + "._worker", return_value=phases):
                with self.assertRaisesRegex(RuntimeError, "native bundle"):
                    _child_main(["--_power_interval_worker", "4"])
            report = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertFalse(report["ok"])
        self.assertEqual(report["phases"], phases)

    def test_native_initialization_exit_keeps_cli_timings(self):
        meter = mock.Mock(_npu_smi_power=mock.Mock(return_value="NPU Real-time Power(W): 90.0"),
                          _load_freq_module=mock.Mock(side_effect=SystemExit(3)))
        try:
            phases = _run_phases(4, meter)
        except SystemExit:
            self.fail("Native initialization exit discarded the completed CLI phase")
        self.assertEqual(phases[0]["successful_samples"], 10)
        self.assertEqual(phases[1]["successful_samples"], 0)
        self.assertIn("SystemExit: 3", phases[1]["errors"][0])

    def test_phase_statistics_exclude_warmup_and_calculate_read_start_intervals(self):
        result = _measure("reader", lambda: 1, clock=iter([0, 1, 2, 4, 10, 11, 14, 2_000_000_000]).__next__)
        self.assertEqual(result["successful_samples"], 2)
        self.assertEqual(result["latency_ms"], {"min": 0.000002, "median": 0.0000025, "max": 0.000003})
        self.assertEqual(result["read_start_interval_ms"], {"min": 0.000009, "median": 0.000009, "max": 0.000009})
        self.assertEqual(result["attempted_samples"], 2)

    def test_config_requires_opt_in_and_explicit_nonnegative_device(self):
        with self.assertRaises(ValueError):
            _configuration({})
        with self.assertRaises(ValueError):
            _configuration({"RUN_ASCEND_POWER_INTERVAL_TEST": "1"})
        self.assertEqual(_configuration({"RUN_ASCEND_POWER_INTERVAL_TEST": "1", "ASCEND_POWER_TEST_DEVICE": "2"}), 2)

    def test_read_limits_and_errors_are_reported_without_fabricated_timings(self):
        clock = iter([0, 1, *range(2, 80), 2_000_000_000]).__next__
        result = _measure("fake", mock.Mock(side_effect=[None, ValueError("bad")] + [None] * 20), clock=clock)
        self.assertLessEqual(result["successful_samples"], 10)
        self.assertTrue(result["errors"])
        self.assertIsNotNone(result["latency_ms"])

    def test_reading_validation_rejects_unavailable_nonfinite_and_negative_values(self):
        for value in (None, float("nan"), float("inf"), -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _finite_nonnegative(value)

    def test_power_parser_requires_explicit_power_value_and_units(self):
        for text, expected in (("Power(W): 89", 89.0), ("Power: 89 W", 89.0),
                               ("Power Dissipation(W): 89.0", 89.0),
                               ("| Power(W): 89 |", 89.0)):
            with self.subTest(text=text):
                self.assertEqual(_parse_power_output(text), expected)
        for text in ("Power: -1 W", "Power: N/A\nVoltage: 0.9 V", "unrecognized", None,
                     "Power(W): inf", "Power: 89", "NPU Real-time Power(W): -1",
                     "NPU Real-time Power(W): N/A\nChip Count: 1",
                     "NPU Real-time Power(mW): 90300"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                _parse_power_output(text)

    def test_power_parser_accepts_npu_real_time_label(self):
        outputs = (
            ("        NPU ID                         : 7\n"
             "        Chip Count                     : 1\n\n"
             "        NPU Real-time Power(W)         : 90.3\n"
             "        Chip ID                        : 0\n", 90.3),
            ("        NPU Real-time Power(W)         : 90.0\n", 90.0),
        )
        for text, expected in outputs:
            with self.subTest(text=text):
                try:
                    reading = _parse_power_output(text)
                except ValueError as exc:
                    self.fail("Valid npu-smi output rejected: %s" % exc)
                self.assertEqual(reading, expected)

    def test_timeout_cleanup_reaps_after_successful_group_kill(self):
        process = mock.Mock(pid=321, returncode=None)
        def wait(timeout):
            if timeout == 15:
                raise subprocess.TimeoutExpired("worker", timeout)
            process.returncode = 0
            return 0
        process.wait.side_effect = wait
        environment = {"RUN_ASCEND_POWER_INTERVAL_TEST": "1", "ASCEND_POWER_TEST_DEVICE": "0"}
        with mock.patch.dict(os.environ, environment, clear=True), \
                mock.patch.object(subprocess, "Popen", return_value=process), \
                mock.patch.object(os, "killpg") as killpg:
            with self.assertRaises(subprocess.TimeoutExpired):
                _run_hardware()
        self.assertEqual([call.args[1] for call in killpg.call_args_list],
                         [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(process.wait.call_args_list,
                         [mock.call(timeout=15), mock.call(timeout=1), mock.call(timeout=2)])

    def test_failed_attempts_count_toward_throughput_elapsed_time(self):
        clock = iter([0, 1, 2, 10, 11, 12, 20, 2_000_000_000]).__next__
        result = _measure("reader", mock.Mock(side_effect=[None, ValueError("bad"), None]), clock=clock)
        self.assertEqual(result["attempted_samples"], 2)
        self.assertEqual(result["successful_samples"], 1)
        self.assertEqual(result["successful_reads_per_second"], 1e9 / 18)

    def test_worker_failure_is_saved_for_parent_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            with mock.patch.dict(os.environ, {"ASCEND_POWER_RESULT_FILE": str(result_path)}), \
                    mock.patch(__name__ + "._worker", side_effect=OSError("driver unavailable")):
                with self.assertRaisesRegex(RuntimeError, "driver unavailable"):
                    _child_main(["--_power_interval_worker", "0"])
            report = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertFalse(report["ok"])
            self.assertEqual(report["error"], "OSError: driver unavailable")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--_power_interval_worker"]:
        _child_main(sys.argv[1:])
    else:
        unittest.main()
