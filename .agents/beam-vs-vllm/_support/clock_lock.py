#!/usr/bin/env python3
"""Hold explicitly catalogued Ascend clocks and restore rated clocks on stop."""

import argparse
import importlib.util
import signal
import threading
import time
from pathlib import Path


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
    spec = importlib.util.spec_from_file_location("beam_ascend_clock", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load BEAM DSMI controller from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AscendClockController


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device-clock",
        action="append",
        required=True,
        metavar="DEVICE=MHZ",
        help="Physical NPU ID and catalogued clock, for example 0=1300",
    )
    args = parser.parse_args()
    assignments = []
    for item in args.device_clock:
        try:
            device_text, clock_text = item.split("=", 1)
            device, clock = int(device_text), int(clock_text)
        except ValueError as error:
            parser.error(f"invalid --device-clock {item!r}: {error}")
        if device < 0:
            parser.error("device IDs must be nonnegative")
        assignments.append((device, clock))
    if len({device for device, _ in assignments}) != len(assignments):
        parser.error("each physical device may be specified only once")

    controller_type = load_controller()
    controllers = []
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())

    try:
        for device, clock in assignments:
            controller = controller_type(device)
            controllers.append(controller)
            observed = controller.set_clock(clock)
            print(
                f"NPU-{device}: requested={clock} MHz observed={observed} MHz",
                flush=True,
            )
        print("Clock lock active", flush=True)
        while not stopped.wait(1):
            pass
    finally:
        failures = []
        for controller in reversed(controllers):
            try:
                controller.close()
                print(
                    f"NPU-{controller.device_id}: rated clock restored",
                    flush=True,
                )
            except Exception as error:
                failures.append((controller.device_id, error))
        if failures:
            details = "; ".join(
                f"NPU-{device}: {error}" for device, error in failures
            )
            raise RuntimeError(f"Failed to restore rated NPU clocks: {details}")


if __name__ == "__main__":
    main()
