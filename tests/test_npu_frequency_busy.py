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

"""Opt-in Ascend frequency readback under sustained matrix-multiplication load.

Use an exclusively allocated physical device and supported frequencies:

    RUN_ASCEND_FREQ_TEST=1 ASCEND_FREQ_TEST_DEVICE=0 \
    ASCEND_FREQ_TEST_MHZ=900,1300 ASCEND_FREQ_TEST_RESTORE_MHZ=1650 \
    python3 tests/test_npu_frequency_busy.py -v

To verify 1800 MHz under load:

    RUN_ASCEND_FREQ_TEST=1 ASCEND_FREQ_TEST_DEVICE=0 \
    ASCEND_FREQ_TEST_MAX_MHZ=1800 ASCEND_FREQ_TEST_RESTORE_MHZ=1650 \
    python3 tests/test_npu_frequency_busy.py -v

The restore value is operator-supplied, never inferred from idle/rated clocks.
Normal discovery skips the hardware test before loading any driver or torch.
Native imports and calls run only in private, deadline-bounded helper processes.
Cleanup defers SIGINT/SIGTERM until restoration and owned-child shutdown finish.
SIGKILL, host failure or an uninterruptible kernel/driver cannot guarantee restore.
An unreaped setter may mutate the device later: restoration is then UNCERTAIN,
even if a subsequent busy read matches the explicit restore value.
"""

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock


def _config(environment):
    required = ("ASCEND_FREQ_TEST_DEVICE", "ASCEND_FREQ_TEST_RESTORE_MHZ")
    missing = [key for key in required if not environment.get(key, "").strip()]
    if missing:
        raise ValueError("Explicit device, targets and restore MHz are required: " + ", ".join(missing))
    device = int(environment[required[0]])
    targets_text = environment.get("ASCEND_FREQ_TEST_MHZ", "").strip()
    maximum_text = environment.get("ASCEND_FREQ_TEST_MAX_MHZ", "").strip()
    if bool(targets_text) == bool(maximum_text):
        raise ValueError("Set exactly one of ASCEND_FREQ_TEST_MHZ or ASCEND_FREQ_TEST_MAX_MHZ")
    targets = [int(value.strip()) for value in (targets_text or maximum_text).split(",")]
    restore = int(environment[required[1]])
    timeout = float(environment.get("ASCEND_FREQ_TEST_TIMEOUT_S", "30"))
    tolerance = float(environment.get("ASCEND_FREQ_TEST_TOLERANCE_MHZ", "0"))
    if (device < 0 or len(targets) not in (1, 2)
            or any(not 0 < value <= 65535 for value in [*targets, restore])):
        raise ValueError("Select one nonnegative physical device and one or two valid MHz targets plus restore MHz")
    if not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError("ASCEND_FREQ_TEST_TIMEOUT_S must be finite and within (0, 300]")
    if (not math.isfinite(tolerance) or tolerance < 0
            or (len(targets) == 2 and abs(targets[0] - targets[1]) <= 2 * tolerance)):
        raise ValueError("Targets must be distinct with disjoint tolerance bands")
    return device, targets, restore, timeout, tolerance


def _publish(path, data):
    path = Path(path)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def _role_result(operation, device, action):
    try:
        return {"operation": operation, "device": device, "ok": True, "data": action()}
    except BaseException as exc:
        data = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:512])}
        if isinstance(exc, SystemExit):
            data["code"] = exc.code if type(exc.code) in (int, str, type(None)) else repr(exc.code)
        return {"operation": operation, "device": device, "ok": False, "data": data}


def _busy_worker(device, stop, progress, lifetime):
    # Only the load role touches torch. Physical masking precedes both imports.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    deadline = time.monotonic() + lifetime
    import torch
    import torch_npu  # Registers the backend only in this private child role.

    if torch.npu.device_count() != 1:
        raise RuntimeError("The selected mask did not expose exactly one logical NPU")
    torch.npu.set_device(0)
    completed = 0
    with torch.inference_mode():
        left = torch.ones((4096, 4096), dtype=torch.float16, device="npu:0")
        right = torch.ones_like(left)
        output = torch.empty_like(left)
        while not Path(stop).exists() and time.monotonic() < deadline:
            for _ in range(8):
                torch.mm(left, right, out=output)
            torch.npu.synchronize()
            completed += 1
            _publish(progress, {"sequence": completed, "completed_at": time.monotonic()})
    return {"sequence": completed}


def _perform_role(operation, device, arguments):
    if os.environ.get("RUN_ASCEND_FREQ_TEST") != "1":
        raise ValueError("Child hardware roles require RUN_ASCEND_FREQ_TEST=1")
    configured, targets, restore, timeout, _ = _config(os.environ)
    if device != configured:
        raise ValueError("Child physical device differs from the explicit configuration")
    if operation == "load":
        stop, progress, lifetime = arguments
        lifetime = float(lifetime)
        if not math.isfinite(lifetime) or not 0 < lifetime <= 45 + 5 * timeout + 90:
            raise ValueError("Invalid load lifetime")
        return _busy_worker(device, stop, progress, lifetime)
    if operation not in ("check", "read", "set"):
        raise ValueError("Unknown private child role: " + operation)
    if operation == "set":
        if len(arguments) != 1:
            raise ValueError("SET requires exactly one MHz argument")
        mhz = int(arguments[0])
        if mhz not in [*targets, restore]:
            raise ValueError("SET requires an operator-supplied target or restore MHz")
    elif arguments:
        raise ValueError("Unexpected control-role arguments")

    # Import/DSMI initialization/get_info/set_freq all stay outside the parent.
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from src import energy_meter

    freq = energy_meter._load_freq_module()
    if freq.dsmi is None:
        raise RuntimeError("DSMI frequency-control driver unavailable")
    if device not in freq.dsmi.device_ids:
        raise RuntimeError("Selected physical device is not exposed by DSMI")
    if operation == "check":
        return {"device_ids": freq.dsmi.device_ids}
    if operation == "read":
        return freq.AscendML.get_info(device)
    energy_meter.set_freq([device], mhz)
    return {"mhz": mhz}


def _child_main(arguments):
    # The parent blocks signals across Popen + registration. Exec inherits that
    # mask, so undo it here; start_new_session keeps terminal/parent-group signals
    # away from the load while allowing explicit terminate/kill during shutdown.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
    operation, device, response, *role_arguments = arguments
    device = int(device)
    result = _role_result(operation, device, lambda: _perform_role(operation, device, role_arguments))
    _publish(response, result)
    if not result["ok"]:
        print(result, file=sys.stderr, flush=True)
    return 0 if result["ok"] else 1


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AssertionError("Frequency-test phase deadline expired")
    return remaining


def _validate_response(path, operation, device):
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AssertionError("Missing or malformed child response %s: %s" % (path, exc)) from exc
    if (not isinstance(result, dict) or result.get("operation") != operation
            or type(result.get("device")) is not int or result["device"] != device
            or type(result.get("ok")) is not bool or not isinstance(result.get("data"), dict)):
        raise AssertionError("Invalid child response for operation=%s device=%d: %r" % (operation, device, result))
    if not result["ok"]:
        raise AssertionError("Child %s device=%d failed: %s" % (operation, device, result["data"]))
    return result["data"]


def _diagnostic(child):
    try:
        with child["log"].open(encoding="utf-8", errors="replace") as log:
            output = log.read(4096)
    except OSError as exc:
        output = str(exc)
    response = ""
    if "response" in child:
        try:
            with child["response"].open(encoding="utf-8", errors="replace") as report:
                response = report.read(4096)
        except OSError as exc:
            response = str(exc)
    return "role=%s pid=%s log=%s: %s; response=%s" % (
        child.get("operation", "load"), child["process"].pid, child["log"], output, response)


def _stop_process(child, errors):
    process = child["process"]
    # Each wait is finite, including the final post-kill reap. Wait/signal errors
    # must not skip later escalation; a timeout is never an implicit join.
    for action in (None, process.terminate, process.kill):
        if action is not None:
            try:
                action()
            except ProcessLookupError:
                pass
            except BaseException as exc:
                errors.append("Shutdown signal failed for pid=%s: %s" % (process.pid, exc))
        try:
            process.wait(timeout=5.0)
            return True
        except subprocess.TimeoutExpired:
            continue
        except BaseException as exc:
            errors.append("Shutdown wait failed for pid=%s: %s" % (process.pid, exc))
    errors.append("UNREAPED owned child; " + _diagnostic(child))
    return False


class _Processes:
    """Ownership of this test's load and one-shot native-control children."""

    def __init__(self, directory, device):
        self.directory, self.device = directory, device
        self.children = []
        self.uncertain = False

    def spawn_role(self, operation, *arguments):
        stem = self.directory / (operation + "-" + uuid.uuid4().hex)
        child = {"operation": operation, "response": stem.with_suffix(".json"),
                 "log": stem.with_suffix(".log")}
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        try:
            with child["log"].open("xb") as output:
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--_ascend_freq_role",
                     operation, str(self.device), str(child["response"]), *map(str, arguments)],
                    shell=False, stdin=subprocess.DEVNULL, stdout=output,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
                child["process"] = process
                self.children.append(child)  # Own it before a pending signal can raise.
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
        return child

    def run_job(self, operation, deadline, *arguments):
        _remaining(deadline)
        child = self.spawn_role(operation, *arguments)
        try:
            code = child["process"].wait(timeout=_remaining(deadline))
        except subprocess.TimeoutExpired as exc:
            raise AssertionError("Native %s job exceeded phase deadline; %s" %
                                 (operation, _diagnostic(child))) from exc
        # Only a completed, reaped child is removed; interrupted/timed-out ones
        # stay owned for cancellation before the fresh restoration attempt.
        self.children.remove(child)
        try:
            data = _validate_response(child["response"], operation, self.device)
            if code != 0:
                raise AssertionError("Child exited with code %s despite an ok response" % code)
            _remaining(deadline)
            return data
        except AssertionError as exc:
            raise AssertionError("%s; %s" % (exc, _diagnostic(child))) from exc

    def _stop_owned(self, child, errors):
        try:
            reaped = _stop_process(child, errors)
        except BaseException as exc:
            errors.append("Child shutdown failed for pid=%s: %s" % (child["process"].pid, exc))
            reaped = False
        if reaped:
            self.children.remove(child)
        elif child["operation"] == "set":
            self.uncertain = True
            errors.append("Restoration UNCERTAIN: unreaped setter pid=%s may mutate hardware later" %
                          child["process"].pid)

    def cancel_controls(self, errors):
        for child in list(self.children):
            if child["operation"] != "load":
                self._stop_owned(child, errors)

    def shutdown(self, errors):
        for child in sorted(self.children, key=lambda item: item["operation"] == "load"):
            self._stop_owned(child, errors)


def _read_progress(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise AssertionError("Malformed/unreadable load progress: %s" % exc) from exc
    if (not isinstance(data, dict) or type(data.get("sequence")) is not int or data["sequence"] <= 0
            or type(data.get("completed_at")) is not float
            or not math.isfinite(data["completed_at"]) or data["completed_at"] <= 0):
        raise AssertionError("Invalid completed-matmul progress: %r" % data)
    return data


def _next_progress(process, progress, after, deadline):
    if process is None:
        raise AssertionError("NPU load process was not started")
    last = None
    while time.monotonic() < deadline:
        code = process["process"].poll()
        if code is not None:
            raise AssertionError("NPU load process exited (code %s); %s" % (code, _diagnostic(process)))
        last = _read_progress(progress)
        if last is not None and last["completed_at"] > after:
            if process["process"].poll() is not None:
                raise AssertionError("NPU load exited after publishing progress; " + _diagnostic(process))
            _remaining(deadline)
            return last
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    raise AssertionError("NPU workload did not complete new matrix multiplications before the deadline; progress=%r" % last)


def _read_busy_frequency(jobs, process, progress, deadline):
    _next_progress(process, progress, time.monotonic(), deadline)
    info = jobs.run_job("read", deadline)
    cutoff = time.monotonic()  # AFTER the read job has returned, not before get_info.
    value = info.get("aic_freq")
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise AssertionError("Invalid busy NPU frequency telemetry: %r; check src/freq.py driver/CANN library paths" % value)
    completed = _next_progress(process, progress, cutoff, deadline)
    return float(value), completed


def _verify_frequency(jobs, target, process, progress, deadline, tolerance):
    consecutive, observations = 0, []
    while time.monotonic() < deadline:
        try:
            mhz, completed = _read_busy_frequency(jobs, process, progress, deadline)
        except AssertionError as exc:
            raise AssertionError("device=%d target=%d MHz; observed=%s; %s" %
                                 (jobs.device, target, observations[-20:], exc)) from exc
        observations.append(mhz)
        consecutive = consecutive + 1 if abs(mhz - target) <= tolerance else 0
        print("device=%d target=%d MHz observed=%s progress=%s consecutive=%d" %
              (jobs.device, target, observations[-20:], completed, consecutive), flush=True)
        if consecutive == 3:
            return
        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
    raise AssertionError("device=%d target=%d MHz did not settle under load; observed=%s" %
                         (jobs.device, target, observations[-20:]))


def _cleanup_busy(jobs, load, progress, restore, timeout, tolerance, attempted, errors):
    try:
        try:
            jobs.cancel_controls(errors)
        except BaseException as exc:
            errors.append("Control cancellation failed: %s" % exc)
        if attempted:
            try:
                deadline = time.monotonic() + timeout
                jobs.run_job("set", deadline, restore)
                _verify_frequency(jobs, restore, load, progress, deadline, tolerance)
            except BaseException as exc:
                errors.append("Restoration to %d MHz failed or could not be verified under load: %s" % (restore, exc))
    finally:
        try:
            (jobs.directory / "stop").touch()
        except BaseException as exc:
            errors.append("Stop-marker creation failed: %s" % exc)
        finally:
            try:
                jobs.shutdown(errors)
            except BaseException as exc:
                errors.append("Owned-child shutdown failed: %s" % exc)


def _report_result(primary, deferred, errors):
    issues = list(errors)
    if deferred:
        issues.append("Deferred cleanup interruption: " + ", ".join(signal.Signals(value).name for value in deferred))
    if primary is not None:
        if issues:
            details = "; ".join(issues)
            if hasattr(primary, "add_note"):
                primary.add_note(details)
            try:
                print(details, file=sys.stderr, flush=True)
            except OSError:
                pass  # A broken diagnostic stream must not replace the primary error.
        raise primary
    if issues:
        raise AssertionError("; ".join(issues))


def _exercise_busy(jobs, targets, restore, timeout, tolerance, temporary=None, repeat_first=True):
    load, attempted, primary = None, False, None
    progress = jobs.directory / "progress.json"
    errors, deferred, previous = [], [], {}
    cleaning = False

    def interrupted(signum, frame):
        if cleaning:
            deferred.append(signum)
        else:
            raise KeyboardInterrupt("Frequency test received " + signal.Signals(signum).name)

    try:
        try:
            # Capture both dispositions before any new handler can interrupt us.
            previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
            for signum in previous:
                signal.signal(signum, interrupted)
            jobs.run_job("check", time.monotonic() + timeout)
            started = time.monotonic()
            load = jobs.spawn_role("load", jobs.directory / "stop", progress, 45 + 5 * timeout + 90)
            _next_progress(load, progress, started, started + 45)
            mhz, completed = _read_busy_frequency(jobs, load, progress, time.monotonic() + timeout)
            print("device=%d baseline=%s MHz progress=%s" % (jobs.device, mhz, completed), flush=True)
            for target in (*targets, targets[0]) if repeat_first else targets:
                deadline = time.monotonic() + timeout
                attempted = True  # Mark BEFORE launch: a native failure may already mutate.
                jobs.run_job("set", deadline, target)
                _verify_frequency(jobs, target, load, progress, deadline, tolerance)
        except BaseException as exc:
            cleaning = True
            primary = exc
        finally:
            cleaning = True  # Both handlers now record, never raise during cleanup.
            try:
                _cleanup_busy(jobs, load, progress, restore, timeout, tolerance, attempted, errors)
            except BaseException as exc:
                errors.append("Frequency-test cleanup failed: %s" % exc)
            finally:
                if temporary is not None:
                    try:
                        temporary.cleanup()
                    except BaseException as exc:
                        errors.append("Temporary-directory cleanup failed: %s" % exc)
    finally:
        # Restore both handlers even if one restoration itself fails.
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except BaseException as exc:
                errors.append("Signal-handler restoration failed for %s: %s" % (signum, exc))
    _report_result(primary, deferred, errors)


@unittest.skipUnless(os.environ.get("RUN_ASCEND_FREQ_TEST") == "1", "Opt-in hardware test: set RUN_ASCEND_FREQ_TEST=1")
class BusyNpuFrequencyTests(unittest.TestCase):
    def test_frequency_changes_while_npu_is_busy(self):
        if os.environ.get("ASCEND_FREQ_TEST_MAX_MHZ", "").strip():
            self.skipTest("ASCEND_FREQ_TEST_MAX_MHZ selects the 1800 MHz test")
        device, targets, restore, timeout, tolerance = _config(os.environ)
        temporary = tempfile.TemporaryDirectory(prefix="ascend-frequency-")
        jobs = _Processes(Path(temporary.name), device)
        # Retain ownership records if a kernel-blocked child cannot be reaped.
        self._owned_children = jobs.children
        _exercise_busy(jobs, targets, restore, timeout, tolerance, temporary)

    def test_frequency_reaches_1800_mhz_while_npu_is_busy(self):
        if not os.environ.get("ASCEND_FREQ_TEST_MAX_MHZ", "").strip():
            self.skipTest("Set ASCEND_FREQ_TEST_MAX_MHZ=1800 to run this hardware test")
        device, targets, restore, timeout, tolerance = _config(os.environ)
        if targets != [1800]:
            raise ValueError("ASCEND_FREQ_TEST_MAX_MHZ must be exactly 1800")
        temporary = tempfile.TemporaryDirectory(prefix="ascend-frequency-")
        jobs = _Processes(Path(temporary.name), device)
        self._owned_children = jobs.children
        _exercise_busy(jobs, targets, restore, timeout, tolerance, temporary, repeat_first=False)


class FrequencyTestConfigurationTests(unittest.TestCase):
    def test_max_frequency_uses_the_single_explicit_target(self):
        maximum = {"ASCEND_FREQ_TEST_DEVICE": "4", "ASCEND_FREQ_TEST_MAX_MHZ": "1800",
                   "ASCEND_FREQ_TEST_RESTORE_MHZ": "1650"}
        self.assertEqual(_config(maximum), (4, [1800], 1650, 30.0, 0.0))

    def test_explicit_inputs_and_nonoverlapping_targets_are_required(self):
        valid = {"ASCEND_FREQ_TEST_DEVICE": "4", "ASCEND_FREQ_TEST_MHZ": "900,1300",
                 "ASCEND_FREQ_TEST_RESTORE_MHZ": "1650"}
        self.assertEqual(_config(valid), (4, [900, 1300], 1650, 30.0, 0.0))
        for overrides in ({"ASCEND_FREQ_TEST_RESTORE_MHZ": ""}, {"ASCEND_FREQ_TEST_DEVICE": "-1"},
                          {"ASCEND_FREQ_TEST_MHZ": "900,900"}, {"ASCEND_FREQ_TEST_MHZ": "900,1300,1500"},
                          {"ASCEND_FREQ_TEST_MHZ": "900,65536"}, {"ASCEND_FREQ_TEST_TIMEOUT_S": "nan"},
                          {"ASCEND_FREQ_TEST_TOLERANCE_MHZ": "200"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                _config({**valid, **overrides})


class FrequencySafetyRegressionTests(unittest.TestCase):
    def test_single_busy_target_is_set_once_before_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = mock.Mock(directory=Path(directory), device=4)
            load = {"process": mock.Mock(), "log": Path(directory) / "load.log"}
            jobs.spawn_role.return_value = load
            with mock.patch.object(signal, "getsignal", return_value=signal.SIG_DFL), \
                    mock.patch.object(signal, "signal"), \
                    mock.patch(__name__ + "._next_progress", return_value={"sequence": 1, "completed_at": 1.0}), \
                    mock.patch(__name__ + "._read_busy_frequency", return_value=(900.0, {"sequence": 1})), \
                    mock.patch(__name__ + "._verify_frequency"):
                _exercise_busy(jobs, [1800], 1650, 30.0, 0.0, repeat_first=False)
        set_calls = [call for call in jobs.run_job.call_args_list if call.args[0] == "set"]
        self.assertEqual([call.args[2] for call in set_calls], [1800, 1650])

    def test_progress_must_complete_strictly_after_read_returns(self):
        now = [1.0]
        frames = iter(((2.0, 1, 2.0), (10.2, 2, 5.0), (10.3, 3, 10.0), (11.2, 4, 11.0)))
        load = {"process": mock.Mock(), "log": Path("load.log")}
        load["process"].poll.return_value = None
        jobs = mock.Mock(device=4)

        def read_progress(path):
            now[0], sequence, completed_at = next(frames)
            return {"sequence": sequence, "completed_at": completed_at}

        def read_job(operation, deadline):
            self.assertEqual((operation, deadline), ("read", 20.0))
            now[0] = 10.0
            return {"aic_freq": 900}

        jobs.run_job.side_effect = read_job
        with mock.patch.object(time, "monotonic", side_effect=lambda: now[0]), \
                mock.patch.object(time, "sleep"), \
                mock.patch(__name__ + "._read_progress", side_effect=read_progress):
            mhz, progress = _read_busy_frequency(jobs, load, Path("progress.json"), 20.0)
        self.assertEqual((mhz, progress), (900.0, {"sequence": 4, "completed_at": 11.0}))

    def test_stalled_progress_cannot_certify_a_busy_read(self):
        now = [1.0]
        load = {"process": mock.Mock(), "log": Path("load.log")}
        load["process"].poll.return_value = None
        jobs = mock.Mock(device=4)

        def read_job(operation, deadline):
            now[0] = 10.0
            return {"aic_freq": 900}

        def sleep(seconds):
            now[0] += seconds

        jobs.run_job.side_effect = read_job
        with mock.patch.object(time, "monotonic", side_effect=lambda: now[0]), \
                mock.patch.object(time, "sleep", side_effect=sleep), \
                mock.patch(__name__ + "._read_progress", side_effect=[
                    {"sequence": 1, "completed_at": 2.0},
                    *[{"sequence": 2, "completed_at": 10.0}] * 20,
                ]):
            with self.assertRaisesRegex(AssertionError, "did not complete new"):
                _read_busy_frequency(jobs, load, Path("progress.json"), 10.5)

    def test_matching_requires_three_consecutive_real_busy_readbacks(self):
        # Removing the reset on a mismatch would accept only the first four reads.
        now = [0.0]
        sequence = [0]
        load = {"process": mock.Mock(), "log": Path("load.log")}
        load["process"].poll.return_value = None

        def clock():
            now[0] += 0.01
            return now[0]

        def read_progress(path):
            sequence[0] += 1
            return {"sequence": sequence[0], "completed_at": clock()}

        for tolerance, values in ((0.0, (900, 901, 900, 900, 900)),
                                  (1.0, (901, 902, 901, 901, 901))):
            jobs = mock.Mock(device=4)
            jobs.run_job.side_effect = [{"aic_freq": value} for value in values]
            with self.subTest(tolerance=tolerance), \
                    mock.patch.object(time, "monotonic", side_effect=clock), \
                    mock.patch.object(time, "sleep"), mock.patch("builtins.print"), \
                    mock.patch(__name__ + "._read_progress", side_effect=read_progress):
                _verify_frequency(jobs, 900, load, Path("progress.json"), 30.0, tolerance)
            self.assertEqual(jobs.run_job.call_count, 5)
            self.assertTrue(all(call.args == ("read", 30.0) for call in jobs.run_job.call_args_list))

    def test_job_uses_remaining_deadline_and_keeps_timed_out_child_owned(self):
        now = [10.0]
        process = mock.Mock(pid=123)
        process.wait.side_effect = subprocess.TimeoutExpired("read", 1.0)

        def spawn(*args, **kwargs):
            now[0] = 11.0
            return process

        with tempfile.TemporaryDirectory() as directory:
            jobs = _Processes(Path(directory), 4)
            with mock.patch.object(time, "monotonic", side_effect=lambda: now[0]), \
                    mock.patch.object(signal, "pthread_sigmask"), \
                    mock.patch.object(subprocess, "Popen", side_effect=spawn) as popen:
                with self.assertRaisesRegex(AssertionError, "deadline"):
                    jobs.run_job("read", 12.0)
                now[0] = 12.0
                with self.assertRaisesRegex(AssertionError, "deadline"):
                    jobs.run_job("read", 12.0)
            process.wait.assert_called_once_with(timeout=1.0)
            self.assertEqual(len(jobs.children), 1)
            self.assertIs(jobs.children[0]["process"], process)
            self.assertEqual(popen.call_count, 1)
            args, kwargs = popen.call_args
            self.assertEqual(args[0][:4], [sys.executable, str(Path(__file__).resolve()),
                                          "--_ascend_freq_role", "read"])
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            self.assertFalse(kwargs["shell"])
            self.assertTrue(kwargs["start_new_session"])
            self.assertNotEqual(kwargs["stdout"], subprocess.PIPE)
            self.assertEqual(kwargs["stderr"], subprocess.STDOUT)

    def test_spawn_registers_ownership_before_delivering_pending_interrupt(self):
        process = mock.Mock(pid=123)
        with tempfile.TemporaryDirectory() as directory:
            jobs = _Processes(Path(directory), 4)
            with mock.patch.object(subprocess, "Popen", return_value=process), \
                    mock.patch.object(signal, "pthread_sigmask", side_effect=[set(), KeyboardInterrupt()]):
                with self.assertRaises(KeyboardInterrupt):
                    jobs.spawn_role("set", 900)
            self.assertIs(jobs.children[0]["process"], process)

    def test_shutdown_escalates_with_bounded_waits_even_after_wait_failure(self):
        process = mock.Mock(pid=123)
        process.wait.side_effect = [subprocess.TimeoutExpired("load", 5), OSError("wait failed"), 0]
        process.terminate.side_effect = OSError("terminate failed")
        errors = []
        child = {"process": process, "operation": "load", "log": Path("load.log")}
        self.assertTrue(_stop_process(child, errors))
        self.assertEqual(process.wait.call_args_list, [mock.call(timeout=5.0)] * 3)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertTrue(any("wait failed" in error for error in errors))
        self.assertTrue(any("terminate failed" in error for error in errors))

    def test_unreaped_setter_keeps_restoration_uncertain_after_a_matching_restore(self):
        process = mock.Mock(pid=123)
        process.wait.side_effect = subprocess.TimeoutExpired("set", 5)
        with tempfile.TemporaryDirectory() as directory:
            jobs = _Processes(Path(directory), 4)
            child = {"process": process, "operation": "set", "log": Path(directory) / "set.log"}
            jobs.children.append(child)
            errors = []
            with mock.patch.object(jobs, "run_job", return_value={"mhz": 1650}) as restore, \
                    mock.patch(__name__ + "._verify_frequency") as verify:
                _cleanup_busy(jobs, None, Path(directory) / "progress.json", 1650, 30.0, 0.0, True, errors)
            restore.assert_called_once()
            self.assertEqual(restore.call_args.args[0], "set")
            self.assertEqual(restore.call_args.args[2], 1650)
            verify.assert_called_once()
            self.assertTrue(jobs.uncertain)
            self.assertIn(child, jobs.children)
            with self.assertRaisesRegex(AssertionError, "UNCERTAIN"):
                _report_result(None, [], errors)

    def test_cleanup_defers_signals_and_preserves_primary_error_through_shutdown(self):
        now, sequence, events = [0.0], [0], []
        previous = {signal.SIGINT: object(), signal.SIGTERM: object()}
        handlers = dict(previous)
        process = mock.Mock(pid=123)
        process.poll.return_value = None
        primary = RuntimeError("first SET failed after possible mutation")

        def install(signum, handler):
            old = handlers[signum]
            handlers[signum] = handler
            return old

        def clock():
            now[0] += 0.01
            return now[0]

        def read_progress(path):
            sequence[0] += 1
            return {"sequence": sequence[0], "completed_at": clock()}

        def job(operation, deadline, *arguments):
            if operation == "set":
                if arguments[0] != 1650:
                    raise primary
                events.append("restore")
                handlers[signal.SIGINT](signal.SIGINT, None)
                handlers[signal.SIGTERM](signal.SIGTERM, None)
                return {"mhz": 1650}
            return {"aic_freq": 1650} if operation == "read" else {}

        def wait(timeout):
            events.append("shutdown")
            return 0

        process.wait.side_effect = wait
        with tempfile.TemporaryDirectory() as directory:
            jobs = _Processes(Path(directory), 4)
            with mock.patch.object(jobs, "run_job", side_effect=job), \
                    mock.patch.object(subprocess, "Popen", return_value=process), \
                    mock.patch.object(signal, "pthread_sigmask"), \
                    mock.patch.object(signal, "getsignal", side_effect=lambda signum: handlers[signum]), \
                    mock.patch.object(signal, "signal", side_effect=install), \
                    mock.patch.object(time, "monotonic", side_effect=clock), \
                    mock.patch.object(time, "sleep"), mock.patch("builtins.print"), \
                    mock.patch(__name__ + "._read_progress", side_effect=read_progress):
                with self.assertRaises(RuntimeError) as failure:
                    _exercise_busy(jobs, [900, 1300], 1650, 30.0, 0.0)
            self.assertIs(failure.exception, primary)
            self.assertEqual(events, ["restore", "shutdown"])
            self.assertEqual(handlers, previous)
            self.assertTrue((Path(directory) / "stop").exists())
            if hasattr(primary, "__notes__"):
                self.assertIn("SIGINT", " ".join(primary.__notes__))
                self.assertIn("SIGTERM", " ".join(primary.__notes__))
        with self.assertRaisesRegex(AssertionError, "Deferred"):
            _report_result(None, [signal.SIGTERM], [])

    def test_interruption_during_handler_installation_restores_original_dispositions(self):
        previous = {signal.SIGINT: signal.default_int_handler, signal.SIGTERM: signal.SIG_DFL}
        handlers = dict(previous)

        def install(signum, handler):
            old = handlers[signum]
            handlers[signum] = handler
            if signum == signal.SIGINT and handler is not previous[signum]:
                raise KeyboardInterrupt("interrupted after installing SIGINT")
            return old

        with tempfile.TemporaryDirectory() as directory:
            jobs = mock.Mock(directory=Path(directory), device=4)
            with mock.patch.object(signal, "getsignal", side_effect=lambda signum: handlers[signum]), \
                    mock.patch.object(signal, "signal", side_effect=install):
                with self.assertRaises(KeyboardInterrupt):
                    _exercise_busy(jobs, [900, 1300], 1650, 30.0, 0.0)
            self.assertEqual(handlers, previous)
            jobs.run_job.assert_not_called()
            jobs.shutdown.assert_called_once()

    def test_system_exit_is_reported_with_its_code(self):
        def fail():
            raise SystemExit(7)

        result = _role_result("set", 4, fail)
        self.assertFalse(result["ok"])
        self.assertEqual(result["data"]["code"], 7)
        self.assertIn("SystemExit", result["data"]["error"])
        with tempfile.TemporaryDirectory() as directory:
            response = Path(directory) / "response.json"
            _publish(response, result)
            with self.assertRaisesRegex(AssertionError, "SystemExit.*7"):
                _validate_response(response, "set", 4)

    def test_invalid_telemetry_cannot_count_as_a_matching_read(self):
        load = {"process": mock.Mock(), "log": Path("load.log")}
        load["process"].poll.return_value = None
        for value in (None, True, "900", 0, -1, float("nan"), float("inf")):
            jobs = mock.Mock(device=4)
            jobs.run_job.return_value = {"aic_freq": value}
            with self.subTest(value=value), mock.patch.object(time, "monotonic", return_value=1.0), \
                    mock.patch(__name__ + "._read_progress", return_value={"sequence": 1, "completed_at": 2.0}):
                with self.assertRaisesRegex(AssertionError, "Invalid busy NPU frequency"):
                    _read_busy_frequency(jobs, load, Path("progress.json"), 30.0)

    def test_missing_malformed_and_wrong_job_responses_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            response = Path(directory) / "response.json"
            with self.assertRaisesRegex(AssertionError, "response"):
                _validate_response(response, "read", 4)
            response.write_text("{not-json", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "response"):
                _validate_response(response, "read", 4)
            valid = {"operation": "read", "device": 4, "ok": True, "data": {"aic_freq": 900}}
            for changes in ({"operation": "set"}, {"device": 0}, {"device": True},
                            {"ok": 1}, {"data": None}):
                _publish(response, {**valid, **changes})
                with self.subTest(changes=changes), self.assertRaisesRegex(AssertionError, "response"):
                    _validate_response(response, "read", 4)
            _publish(response, valid)
            self.assertEqual(_validate_response(response, "read", 4), {"aic_freq": 900})


if __name__ == "__main__":
    if sys.argv[1:2] == ["--_ascend_freq_role"]:
        sys.exit(_child_main(sys.argv[2:]))
    unittest.main()
