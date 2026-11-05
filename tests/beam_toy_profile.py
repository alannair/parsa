#!/usr/bin/env python3
"""Generate or verify a SYNTHETIC BEAM S1/S2 NPU profile.

This is intentionally fake data for scheduler/clock-actuation testing ONLY.
It is not a performance model and must never be used to claim energy savings,
latency SLO compliance, or production DVFS performance.

On host: python3 tests/beam_toy_profile.py --output setup/beam_profiles/910b3-qwen3-4b-toy.csv
In running BEAM container: python /workspace/tests/beam_toy_profile.py --check-only --output /workspace/setup/beam_profiles/910b3-qwen3-4b-toy.csv
"""

import argparse
import csv
from pathlib import Path

NPU_CATALOG = (900, 1000, 1100, 1200, 1300, 1400, 1500, 1650)
BATCHES = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
FIELDS = ("clock", "batch_size", "total_ctx_len", "time_taken", "energy_consumption", "valid")


def generate(output: Path, low: int, high: int):
    if low not in NPU_CATALOG or high not in NPU_CATALOG or low >= high:
        raise ValueError(f"Require low < high from {NPU_CATALOG}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for clock in (low, high):
            for batch in BATCHES:
                # Fake low clock: cheap, but slow for sizable prefills.
                # Fake high clock: costlier, but fast enough for a tight TTFT.
                latency_seconds = (
                    0.012 + 0.0014 * batch if clock == low else
                    0.007 + 0.00013 * batch
                )
                energy_millijoules = (
                    0.9 + 0.018 * batch if clock == low else
                    3.0 + 0.026 * batch
                )
                writer.writerow(dict(clock=clock, batch_size=batch, total_ctx_len=4096,
                                     time_taken=f"{latency_seconds:.6f}",
                                     energy_consumption=f"{energy_millijoules:.6f}",
                                     valid=1))
    print(f"Wrote synthetic profile to {output} ({len(BATCHES) * 2} data rows)")
    print(f"Toy low/high clock: {low}/{high} MHz. These are UNVERIFIED hardware candidates.")


def check(output: Path, low: int, high: int):
    # Import the ACTUAL patched BEAM selector. This needs the BEAM Python
    # environment, usually the running vllm-ascend-beam container.
    from vllm.v1.core.sched.beam_policy import choose_s1, choose_s2
    from vllm.v1.core.sched.energy_model import EnergySimulator

    model = EnergySimulator(
        str(output), 1, 1, gpu_name="910B3",
        model_name="/models/Qwen__Qwen3-4B", device_backend="npu"
    )
    short = choose_s1(
        model, prefill_tokens=64, num_running=1,
        tbt_slo=0.12, ttft_slo=0.60,
        max_chunk_size=2048,
    )
    long = choose_s1(
        model, prefill_tokens=2048, num_running=1,
        tbt_slo=0.12, ttft_slo=0.60,
        max_chunk_size=2048,
    )
    decode = choose_s2(
        model, num_running=1, total_ctx_len=2304, tbt_slo=0.12
    )
    print("ACTUAL BEAM POLICY PREDICTIONS (toy data; not NPU measurements):")
    print(f"  64-token prefill:   {short}")
    print(f"  2048-token prefill: {long}")
    print(f"  decode (1 seq):    {decode}")
    assert short.feasible and short.clock == low, (
        f"Expected short prefill at {low} MHz, got {short}"
    )
    assert long.feasible and long.clock == high, (
        f"Expected long prefill at {high} MHz, got {long}"
    )
    assert decode.feasible and decode.clock == low, (
        f"Expected decode at {low} MHz, got {decode}"
    )
    print("PASS: patched BEAM policy chooses distinct low/high clocks.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("setup/beam_profiles/910b3-qwen3-4b-toy.csv"))
    p.add_argument("--low-clock", type=int, default=1500)
    p.add_argument("--high-clock", type=int, default=1650)
    p.add_argument("--check-only", action="store_true", help="Load CSV into actual BEAM policy; do NOT regenerate")
    args = p.parse_args()
    if args.check_only:
        check(args.output, args.low_clock, args.high_clock)
    else:
        generate(args.output, args.low_clock, args.high_clock)


if __name__ == "__main__":
    main()
