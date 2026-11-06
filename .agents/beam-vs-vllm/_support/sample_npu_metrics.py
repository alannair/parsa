#!/usr/bin/env python3
"""Sample power and AI-core frequency for selected NPUs until signalled."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import os
import signal
import sys
import time
from pathlib import Path


FIELDS = (
    "elapsed_s",
    "utc",
    "run_id",
    "device",
    "freq_mhz",
    "rated_mhz",
    "power_w",
    "error",
)


def load_controller():
    root = Path(__file__).resolve().parents[3]
    source = (
        root
        / "setup"
        / "vllm-beam-npu-0.23.0"
        / "vllm"
        / "v1"
        / "worker"
        / "beam_ascend.py"
    )
    spec = importlib.util.spec_from_file_location("beam_ascend_metrics", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load BEAM DSMI reader from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AscendClockController


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--devices", nargs="+", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    if args.interval <= 0 or len(set(args.devices)) != len(args.devices):
        parser.error("--interval must be positive and devices must be unique")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stopped = False

    def stop(_signal, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    controller_type = load_controller()
    controllers = {
        device: controller_type(device) for device in args.devices
    }
    start = time.monotonic()
    next_tick = start
    try:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o644)
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            while not stopped:
                for device, controller in controllers.items():
                    try:
                        controller.initialize()
                        values = {
                            "freq_mhz": controller.get_clock_mhz(),
                            "rated_mhz": controller._rated_clock_mhz,
                            "power_w": controller.get_power_w(),
                        }
                        error = ""
                    except Exception as issue:
                        values = {}
                        error = f"{type(issue).__name__}: {issue}"
                    row = {
                        "elapsed_s": f"{time.monotonic() - start:.3f}",
                        "utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                        "run_id": args.run_id,
                        "device": device,
                        "freq_mhz": values.get("freq_mhz"),
                        "rated_mhz": values.get("rated_mhz"),
                        "power_w": values.get("power_w"),
                        "error": error,
                    }
                    writer.writerow(row)
                    handle.flush()
                    if error:
                        print(
                            f"NPU {device} telemetry error: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                next_tick += args.interval
                delay = next_tick - time.monotonic()
                if delay > 0 and not stopped:
                    time.sleep(delay)
                elif delay <= 0:
                    next_tick = time.monotonic()
    finally:
        for controller in controllers.values():
            controller.close()


if __name__ == "__main__":
    main()
