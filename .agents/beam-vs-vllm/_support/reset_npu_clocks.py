#!/usr/bin/env python3
"""Restore rated NPU clocks and verify the devices return to their idle clocks."""

import argparse
import importlib.util
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
    spec = importlib.util.spec_from_file_location("beam_ascend_reset", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load BEAM DSMI controller from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AscendClockController


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", required=True, type=int)
    parser.add_argument("--rated-mhz", type=int, default=1800)
    parser.add_argument("--idle-mhz", type=int, default=800)
    parser.add_argument("--timeout-seconds", type=float, default=10)
    args = parser.parse_args()
    if (
        len(set(args.devices)) != len(args.devices)
        or any(device < 0 for device in args.devices)
        or args.rated_mhz <= 0
        or args.idle_mhz <= 0
        or args.timeout_seconds <= 0
    ):
        parser.error("devices must be unique and all clock/timeout values positive")

    controller_type = load_controller()
    for device in args.devices:
        controller = controller_type(device)
        try:
            controller.initialize()
            if controller._rated_clock_mhz != args.rated_mhz:
                raise RuntimeError(
                    f"NPU-{device} rated clock is {controller._rated_clock_mhz} MHz, "
                    f"expected {args.rated_mhz} MHz"
                )
            controller.restore_rated_clock()
            print(f"NPU-{device}: rated clock restored to {args.rated_mhz} MHz", flush=True)
        finally:
            controller.close()

        deadline = time.monotonic() + args.timeout_seconds
        observed = None
        while time.monotonic() < deadline:
            verifier = controller_type(device)
            try:
                observed = verifier.get_clock_mhz()
            finally:
                verifier.close()
            if observed == args.idle_mhz:
                break
            time.sleep(0.5)
        if observed != args.idle_mhz:
            raise RuntimeError(
                f"NPU-{device} idle clock is {observed} MHz, expected {args.idle_mhz} MHz"
            )
        print(
            f"NPU-{device}: verified idle clock {observed} MHz "
            f"(rated {args.rated_mhz} MHz)",
            flush=True,
        )


if __name__ == "__main__":
    main()
