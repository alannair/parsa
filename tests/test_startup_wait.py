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

"""Opt-in startup timing probe; never launches services during test discovery.

Run inside an otherwise idle serving container, with its usual runtime environment:

    MODEL=/models/qwen3-8b ROUTING_POLICY=random \
    VLLM_SERVE_EXTRA_ARGS=--no-enable-prefix-caching \
    python3 tests/test_startup_wait.py --run --devices 0,1 --presets PATH/presets.json

Use a generated single-lane preset matching the cap/MBT you want to measure.
No frequency setter or cache clearing is performed. Repeat representative cold
and warm launches before choosing a production wait; one sample is not a guarantee.
"""

import argparse
from contextlib import contextmanager
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


class _ProbeDeadline(Exception):
    pass


def _check_probe_context(router_args):
    if router_args:
        raise ValueError("ROUTER_EXTRA_ARGS is unsupported: it could override the probe's lane/ports")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Run the startup probe on the main thread")
    if any(signal.getitimer(signal.ITIMER_REAL)):
        raise RuntimeError("Refusing to replace an existing real-time alarm")


@contextmanager
def _http_deadline(seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise TimeoutError("Startup budget exhausted before HTTP probing")

    def expired(signum, frame):
        # Not OSError: urllib/health retries must not swallow the absolute deadline.
        raise _ProbeDeadline()

    previous = signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        except _ProbeDeadline:
            raise TimeoutError("Absolute HTTP readiness/inference deadline exceeded") from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def candidate_wait(health_ready_s, margin_s, *, completion_ok=False):
    if not completion_ok:
        raise ValueError("No wait recommendation without a successful completion")
    if any(not math.isfinite(value) or value < 0 for value in (health_ready_s, margin_s)):
        raise ValueError("Readiness time and margin must be finite and nonnegative")
    return math.ceil(health_ready_s + margin_s)


def stop_owned(processes):
    # Helpers create private sessions. Kill their groups, including leftover workers.
    failed = False
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for process in reversed(processes):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            except OSError:
                failed = True
        for process in reversed(processes):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if sig == signal.SIGKILL:
                    failed = True
    if failed:
        raise RuntimeError("An owned service group could not be stopped/reaped")


def measure(runner, cfg, rows, backend_url, timeout_s):
    processes = []
    real_popen = subprocess.Popen

    def tracked_popen(*args, **kwargs):
        if not kwargs.get("start_new_session"):
            raise RuntimeError("Probe launches must own their process groups")
        process = real_popen(*args, **kwargs)
        processes.append(process)  # Retain even if later PID/log bookkeeping fails.
        return process

    launch_start = time.monotonic()
    deadline = launch_start + timeout_s

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Startup/inference deadline exceeded")
        return seconds

    def check_alive():
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("A launched service exited; inspect the probe logs")

    try:
        with patch.object(subprocess, "Popen", tracked_popen):
            runner.start_vllm_lanes(cfg, rows)
            runner.start_router(cfg)
        wait_start = time.monotonic()  # Same position as run.py's WAIT_SECS sleep.
        # Arm only after process handles are captured; never interrupt Popen creation.
        with _http_deadline(remaining()):
            opener = build_opener(ProxyHandler({}))
            router_url = "http://127.0.0.1:%d" % cfg.router_port
            health_urls = (backend_url.rstrip("/") + "/health", router_url + "/health")
            while True:
                check_alive()
                healthy = True
                for url in health_urls:
                    timeout = min(1.0, remaining())
                    try:
                        with opener.open(url, timeout=timeout) as response:
                            healthy = healthy and response.status == 200
                    except (OSError, HTTPException):
                        healthy = False
                if healthy:
                    break
                time.sleep(min(1.0, remaining()))
            health_ready_s = time.monotonic() - wait_start

            # One bounded request, never retried: a timeout leaves its outcome unknown.
            request = Request(router_url + "/v1/completions", data=json.dumps({
                "model": cfg.model, "prompt": "Hello", "max_tokens": 1,
                "min_tokens": 1, "temperature": 0, "stream": False,
            }).encode(), headers={"Content-Type": "application/json"})
            completion_start = time.monotonic()
            with opener.open(request, timeout=remaining()) as response:
                body = response.read(65537)
                if response.status != 200 or len(body) > 65536:
                    raise ValueError("Completion response unsuccessful or too large")
                if response.headers.get("X-Energy-Aware-Preset-Id") != rows[0].preset_id:
                    raise ValueError("Completion did not identify the expected lane")
            data = json.loads(body)
            choices = data.get("choices") if isinstance(data, dict) else None
            if (not isinstance(choices, list) or not choices or not isinstance(choices[0], dict)
                    or choices[0].get("finish_reason") not in ("stop", "length")
                    or not isinstance(data.get("usage"), dict)
                    or type(data["usage"].get("completion_tokens")) is not int
                    or data["usage"].get("completion_tokens") != 1):
                raise ValueError("Response did not confirm a completed one-token inference")
            remaining()
            check_alive()
            finished = time.monotonic()
            result = {
                "launch_call_s": wait_start - launch_start,
                "health_ready_after_wait_start_s": health_ready_s,
                "first_completion_s": finished - completion_start,
                "successful_completion_after_wait_start_s": finished - wait_start,
            }
    finally:
        stop_owned(processes)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Opt in to launching on an idle NPU pair")
    parser.add_argument("--presets", required=True, type=Path)
    parser.add_argument("--devices", required=True, help="Confirm the free physical NPU pair, e.g. 0,1")
    parser.add_argument("--timeout-s", type=float, default=600,
                        help="Budget from launch start; launch calls are not interrupted. "
                             "HTTP uses the remaining absolute deadline; cleanup is additional.")
    parser.add_argument("--margin-s", type=float, default=30, help="Uncalibrated diagnostic safety buffer")
    args = parser.parse_args()
    if not args.run:
        parser.error("Use --run only inside an idle container with a free NPU pair")
    if not os.environ.get("MODEL"):
        parser.error("Set MODEL to the same model used by the benchmark")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be finite and positive")
    if not math.isfinite(args.margin_s) or args.margin_s < 0:
        parser.error("--margin-s must be finite and nonnegative")
    devices = [value.strip() for value in args.devices.split(",")]
    if len(devices) != 2 or not all(value.isdecimal() for value in devices) or len(set(map(int, devices))) != 2:
        parser.error("--devices must name two distinct nonnegative device IDs")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible and not set(devices).issubset(value.strip() for value in visible.split(",")):
        parser.error("The requested pair is outside ASCEND_RT_VISIBLE_DEVICES")

    presets_path = args.presets.resolve()
    presets = json.loads(presets_path.read_text(encoding="utf-8"))
    if not isinstance(presets, list) or len(presets) != 1 or not isinstance(presets[0], dict):
        parser.error("Use a generated preset file containing exactly one lane")
    backend_url = presets[0].get("backend_url", "")
    backend = urlsplit(backend_url)
    if (backend.scheme != "http" or backend.hostname not in ("127.0.0.1", "localhost")
            or not backend.port or backend.path not in ("", "/")
            or backend.username or backend.password or backend.query or backend.fragment):
        parser.error("The preset must use a plain localhost HTTP backend with an explicit port")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import run as runner
    from src.serving_profiles import runtime_package_context

    state = Path(tempfile.mkdtemp(prefix="parsa-startup-"))
    print("Probe logs (retained): %s" % state, file=sys.stderr)
    with patch.dict(os.environ, {"DEMO_STATE_DIR": str(state), "PRESETS_JSON": str(presets_path),
                                "ASCEND_DEVICES": ",".join(devices), "RESULT_JSON": str(state / "unused.json")}):
        cfg = runner.load_config()
        _check_probe_context(cfg.router_extra_args)
        rows = runner.plan_lanes(cfg, persist=False)
        runner.require_owned_lanes(cfg, rows)
        if (cfg.tp_size != 2 or cfg.skip_router or len(rows) != 1
                or set(rows[0].devices.split(",")) != set(devices)):
            raise ValueError("Probe requires one owned TP2 lane on exactly the confirmed pair, plus its router")
        anchor = runtime_package_context()
        cfg.serving_extra_args, cfg.serving_env_updates, cfg.serving_profile, cfg.serving_condition = runner.plan_serving_profile(
            cfg.model, cfg.tp_size, cfg.vllm_serve_extra_args, os.environ,
            versions=anchor["versions"], origins=anchor["origins"],
        )
        runner.preflight_serving_runtime(cfg, rows, anchor)
        runner.setup_process_env(cfg)
        ports = (backend.port, cfg.router_port)
        if len(set(ports)) != 2 or any(not 1 <= port <= 65535 for port in ports):
            raise ValueError("Backend and router need distinct valid ports")
        for port in ports:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", port))  # Refuse an occupied port; never kill its owner.
        result = measure(runner, cfg, rows, backend_url, args.timeout_s)
        result.update(
            candidate_wait_secs=candidate_wait(result["health_ready_after_wait_start_s"], args.margin_s, completion_ok=True),
            diagnostic_margin_s=args.margin_s,
            model=cfg.model, tp=cfg.tp_size, max_num_seqs=rows[0].max_seqs,
            max_num_batched_tokens=rows[0].max_tokens, serving_condition=cfg.serving_condition,
            frequency_control="unchanged; preset frequency is not applied by this probe",
            logs=str(state),
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    print("Candidate only, not a proven safe minimum. Repeat representative launches. "
          "Idle waiting does not reproduce inference-triggered compilation/warmup; "
          "the short completion does not establish workload-specific steady-state readiness.", file=sys.stderr)


class CandidateWaitTests(unittest.TestCase):
    def test_router_overrides_and_existing_alarms_are_refused(self):
        with self.assertRaises(ValueError):
            _check_probe_context(["--presets=other.json"])
        with patch.object(signal, "getitimer", return_value=(3.0, 0.0)):
            with self.assertRaises(RuntimeError):
                _check_probe_context([])

    def test_exhausted_budget_and_alarm_restore_before_cleanup(self):
        with patch.object(signal, "signal") as install:
            with self.assertRaises(TimeoutError):
                with _http_deadline(0):
                    self.fail("An exhausted budget must not enter the HTTP phase")
            install.assert_not_called()
        for expire in (False, True):
            events = []
            previous = object()

            def install(signum, handler):
                events.append(("handler", handler))
                return previous

            def phase():
                try:
                    with _http_deadline(1):
                        if expire:
                            try:
                                events[0][1](signal.SIGALRM, None)
                            except OSError:
                                self.fail("A health retry must not swallow the alarm")
                finally:
                    events.append(("cleanup", None))

            with self.subTest(expire=expire), patch.object(signal, "signal", side_effect=install), \
                    patch.object(signal, "setitimer", side_effect=lambda timer, delay: events.append(("timer", delay))):
                if expire:
                    with self.assertRaises(TimeoutError):
                        phase()
                else:
                    phase()
            self.assertEqual(events[-3:], [("timer", 0), ("handler", previous), ("cleanup", None)])

    def test_candidate_needs_success_and_a_finite_nonnegative_buffer(self):
        self.assertEqual(candidate_wait(12.2, 30, completion_ok=True), 43)
        with self.assertRaises(ValueError):
            candidate_wait(12.2, 30)
        for ready, margin in ((math.nan, 30), (12.2, math.inf), (-1, 30), (12.2, -1)):
            with self.assertRaises(ValueError):
                candidate_wait(ready, margin, completion_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Startup measurement failed: %s" % exc, file=sys.stderr)
        raise SystemExit(1)
