#!/usr/bin/env python3
"""Generate/check a deliberately SYNTHETIC BEAM NPU S1/S2 DVFS profile.

This is a functional actuator/scheduler test, NOT calibration data.
Requires only Python 3.9 stdlib to generate; use patched BEAM Python
in its container for --check-only.

Default frequencies: 900,1000,1100,1200,1300,1400,1500,1650 MHz.
Do not add 800 to the clock list unless the hardware and both BEAM catalogs
have explicitly been verified/changed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

CLOCKS = (900, 1000, 1100, 1200, 1300, 1400, 1500, 1650)
BATCHES = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
FIELDS = ("clock", "batch_size", "total_ctx_len", "time_taken", "energy_consumption", "valid")
DEFAULT_OUT = Path("setup/beam_profiles/910b3-qwen3-4b-wide-toy.csv")


def profile_values(batch: int, clock: int) -> tuple[float, float]:
    """Synthetic latency seconds and energy millijoules.

    Steep *fictional* slowdown at low frequency makes longer prefills require
    higher clocks under --beam-ttft-slo=0.6 and --beam-tbt-slo=0.12.
    Rising fictitious energy makes the lowest feasible clock attractive.
    These curves are intentionally NOT a physical performance/power model.
    """
    time_s = 0.006 + 0.00019 * batch * (1650 / clock) ** 8
    energy_mj = (0.05 + 0.01 * batch) * (clock / 900) ** 3
    return time_s, energy_mj


def generate(path: Path, clocks: tuple[int, ...] = CLOCKS) -> None:
    if not clocks or any(f not in CLOCKS for f in clocks):
        raise ValueError(f"Only the catalogued frequencies {CLOCKS} are allowed")
    if 1650 not in clocks:
        raise ValueError("BEAM's current NPU EnergySimulator requires batch=512 at 1650 MHz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        for freq in clocks:
            for batch in BATCHES:
                latency, energy = profile_values(batch, freq)
                writer.writerow({
                    "clock": freq,
                    "batch_size": batch,
                    "total_ctx_len": 4096,
                    "time_taken": f"{latency:.9f}",
                    "energy_consumption": f"{energy:.9f}",
                    "valid": 1,
                })
    print(f"Created {path} ({len(clocks)*len(BATCHES)} synthetic rows)")
    print(f"Catalogued clocks: {clocks} MHz")
    print("WARNING: Fake data. Use only for testing DVFS routing, never energy/SLO evaluation.")


def check(path: Path) -> None:
    """Use the real, patched BEAM policy modules installed in the container."""
    from vllm.v1.core.sched.beam_policy import choose_s1, choose_s2
    from vllm.v1.core.sched.energy_model import EnergySimulator

    model = EnergySimulator(
        str(path), 1, 1, gpu_name="910B3",
        model_name="/models/Qwen__Qwen3-4B", device_backend="npu",
    )
    print("Loaded clocks:", model.available_clocks)
    assert tuple(model.available_clocks) == CLOCKS, model.available_clocks

    results = []
    for tokens, target in ((64, 1100), (256, 1300), (512, 1400),
                           (1024, 1500), (2048, 1650)):
        dec = choose_s1(
            model, prefill_tokens=tokens, num_running=1,
            tbt_slo=0.12, ttft_slo=0.60, max_chunk_size=2048,
        )
        print(f"S1 prefill {tokens:4d} tokens: {dec.clock} MHz, "
              f"chunk={dec.chunk_size}, feasible={dec.feasible}, "
              f"estimated TTFT={dec.ttft:.3f}s")
        results.append((f"prefill {tokens}", dec.clock, target, dec.feasible))

    for running, target in ((1, 900), (4, 900)):
        dec = choose_s2(
            model, num_running=running,
            total_ctx_len=2304 * running, tbt_slo=0.12,
        )
        print(f"S2 decode {running:2d} running: {dec.clock} MHz, "
              f"microbatches={dec.num_microbatches}, feasible={dec.feasible}, "
              f"estimated TBT={dec.tbt:.3f}s")
        results.append((f"decode {running}", dec.clock, target, dec.feasible))

    failures = [(name, found, expected, feasible)
                for name, found, expected, feasible in results
                if found != expected or not feasible]
    if failures:
        raise AssertionError(f"Unexpected synthetic BEAM policy decisions: {failures}")
    print("PASS: actual BEAM policy selects 900 through 1650 MHz across test workloads.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-only", action="store_true", help="Use actual patched policy; do not overwrite CSV")
    args = parser.parse_args()
    if args.check_only:
        check(args.output)
    else:
        generate(args.output)


if __name__ == "__main__":
    main()
