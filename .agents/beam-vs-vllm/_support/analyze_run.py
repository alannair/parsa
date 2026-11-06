#!/usr/bin/env python3
"""Summarize paired BEAM/vLLM request and sampled NPU telemetry results."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BACKENDS = ("beam", "vllm")
REQUEST_FIELDS = (
    "index",
    "ok",
    "input_tokens",
    "output_tokens",
    "ttft_s",
    "tpot_mean_s",
    "tpot_99.9th_s",
    "duration_s",
    "status_code",
    "stream_complete",
)


def percentile(values, fraction):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def number(row, field):
    value = row.get(field)
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def write_request_csv(path, requests):
    fields = (
        "index",
        "ok",
        "input_tokens",
        "output_tokens",
        "ttft_s",
        "tpot_mean_s",
        "tpot_99.9th_s",
        "duration_s",
        "status_code",
        "stream_complete",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for request in requests:
            writer.writerow(
                {
                    "index": request.get("prompt_idx"),
                    "ok": request.get("ok"),
                    "input_tokens": request.get("input_num_tokens"),
                    "output_tokens": request.get("output_num_tokens"),
                    "ttft_s": request.get("ttft_s"),
                    "tpot_mean_s": request.get("tpot_mean_s"),
                    "tpot_99.9th_s": request.get("tpot_99.9th_s"),
                    "duration_s": request.get("duration_s"),
                    "status_code": request.get("status_code"),
                    "stream_complete": request.get("stream_complete"),
                }
            )


def read_summary(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing client summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError(f"No request-level results in {path}")
    return payload, requests


def summarize_samples(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing NPU sample CSV: {path}")
    grouped = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            device = int(row["device"])
            elapsed = number(row, "elapsed_s")
            power = number(row, "power_w")
            freq = number(row, "freq_mhz")
            if row.get("error") or elapsed is None or power is None or freq is None:
                continue
            grouped.setdefault(device, []).append((elapsed, power, freq))
    result = {}
    for device, samples in grouped.items():
        samples.sort()
        energy = sum(
            (right[0] - left[0]) * (left[1] + right[1]) / 2
            for left, right in zip(samples, samples[1:])
        )
        powers = [sample[1] for sample in samples]
        clocks = [sample[2] for sample in samples]
        result[device] = {
            "samples": len(samples),
            "energy_j": energy,
            "mean_power_w": statistics.fmean(powers),
            "mean_freq_mhz": statistics.fmean(clocks),
            "min_freq_mhz": min(clocks),
            "max_freq_mhz": max(clocks),
        }
    if set(result) != {0, 1}:
        raise ValueError(f"Expected valid samples for NPUs 0 and 1 in {path}; got {sorted(result)}")
    return result


def summarize_rep(backend, repetition_dir):
    payload, requests = read_summary(repetition_dir / "summary.json")
    write_request_csv(repetition_dir / "request_metrics.csv", requests)
    successful = [request for request in requests if request.get("ok")]
    ttft = [number(request, "ttft_s") for request in successful]
    tpot = [number(request, "tpot_mean_s") for request in successful]
    ttft = [value for value in ttft if value is not None]
    tpot = [value for value in tpot if value is not None]
    devices = summarize_samples(repetition_dir / "npu_samples.csv")
    record = {
        "backend": backend,
        "repetition": repetition_dir.name,
        "requests": len(requests),
        "completed": len(successful),
        "failed": len(requests) - len(successful),
        "shape_valid": payload.get("shape_valid"),
        "elapsed_s": payload.get("total_elapsed_time"),
        "input_tokens": payload.get("total_input_tokens"),
        "output_tokens": payload.get("total_output_tokens"),
        "throughput_output_tokens_s": payload.get("tokens_per_second"),
        "mean_ttft_s": statistics.fmean(ttft) if ttft else None,
        "p95_ttft_s": percentile(ttft, 0.95),
        "p99_ttft_s": percentile(ttft, 0.99),
        "mean_tpot_s": statistics.fmean(tpot) if tpot else None,
        "p95_tpot_s": percentile(tpot, 0.95),
        "p99_tpot_s": percentile(tpot, 0.99),
        "energy_npu0_j": devices[0]["energy_j"],
        "energy_npu1_j": devices[1]["energy_j"],
        "energy_total_j": sum(device["energy_j"] for device in devices.values()),
        "mean_freq_npu0_mhz": devices[0]["mean_freq_mhz"],
        "mean_freq_npu1_mhz": devices[1]["mean_freq_mhz"],
        "min_freq_mhz": min(device["min_freq_mhz"] for device in devices.values()),
        "max_freq_mhz": max(device["max_freq_mhz"] for device in devices.values()),
    }
    return record, devices, requests


def write_csv(path, records, fields=None):
    if not records:
        raise ValueError(f"No rows to write to {path}")
    names = fields or list(records[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def plot_results(root, records, request_sets):
    backends = list(BACKENDS)
    colors = {"beam": "#2878b5", "vllm": "#ef8a17"}
    labels = {"beam": "BEAM", "vllm": "Vanilla vLLM"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, field, title in (
        (axes[0], "ttft_s", "TTFT"),
        (axes[1], "tpot_mean_s", "Mean TPOT"),
    ):
        for backend in backends:
            values = [
                number(request, field)
                for request in request_sets.get(backend, [])
                if request.get("ok")
            ]
            values = [value for value in values if value is not None]
            if values:
                axis.ecdf(values, label=labels[backend], color=colors[backend])
        if not any(line.get_label() in labels.values() for line in axis.lines):
            axis.text(
                0.5,
                0.5,
                "Not applicable for 1 output token",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set(xlabel="Seconds", ylabel="Cumulative fraction", title=title)
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()
    fig.tight_layout()
    fig.savefig(root / "latency_cdf.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for backend in backends:
        directories = sorted((root / backend).glob("rep*/npu_samples.csv"))
        if not directories:
            continue
        path = directories[0]
        series = {0: [], 1: []}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                device = int(row["device"])
                elapsed = number(row, "elapsed_s")
                frequency = number(row, "freq_mhz")
                power = number(row, "power_w")
                if elapsed is None or frequency is None or power is None:
                    continue
                series[device].append((elapsed, frequency, power))
        for device, points in series.items():
            if not points:
                continue
            axes[0].step(
                [point[0] for point in points],
                [point[1] for point in points],
                where="post",
                color=colors[backend],
                linestyle="-" if device == 0 else "--",
                label=f"{labels[backend]} NPU-{device}",
            )
            axes[1].plot(
                [point[0] for point in points],
                [point[2] for point in points],
                color=colors[backend],
                linestyle="-" if device == 0 else "--",
                label=f"{labels[backend]} NPU-{device}",
            )
    axes[0].set_ylabel("AI-core MHz")
    axes[1].set_ylabel("Power (W)")
    axes[1].set_xlabel("Seconds from capture start")
    axes[0].set_title("First repetition: sampled NPU telemetry")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(root / "frequency_power.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for backend in backends:
        selected = [record for record in records if record["backend"] == backend]
        if not selected:
            continue
        x = [record["repetition"] for record in selected]
        axes[0].bar(
            [f"{labels[backend]} {rep}" for rep in x],
            [record["energy_total_j"] for record in selected],
            color=colors[backend],
            alpha=0.85,
        )
        axes[1].bar(
            [f"{labels[backend]} {rep}" for rep in x],
            [record["throughput_output_tokens_s"] or 0 for record in selected],
            color=colors[backend],
            alpha=0.85,
        )
    axes[0].set_ylabel("Sampled energy (J)")
    axes[1].set_ylabel("Output tokens / s")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "energy_throughput.png", dpi=160)
    plt.close(fig)


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--prompt-tokens", required=True, type=int)
    parser.add_argument("--output-tokens", required=True, type=int)
    parser.add_argument("--concurrency-cap", required=True, type=int)
    parser.add_argument("--baseline-clock-mhz", required=True, type=int)
    parser.add_argument("--beam-ttft-slo", required=True, type=float)
    parser.add_argument("--beam-tbt-slo", required=True, type=float)
    args = parser.parse_args()
    root = args.run_dir.resolve()

    records = []
    request_sets = {}
    for backend in BACKENDS:
        request_sets[backend] = []
        rep_dirs = sorted((root / backend).glob("rep*"))
        if not rep_dirs:
            raise FileNotFoundError(f"No repetition directories found for {backend}")
        for repetition_dir in rep_dirs:
            record, _, requests = summarize_rep(backend, repetition_dir)
            records.append(record)
            request_sets[backend].extend(requests)

    profile_path = root / "beam" / "profile_synthetic.csv"
    config = {
        "model": "Qwen3-8B",
        "vllm_version": "0.23.0",
        "tensor_parallel_size": 2,
        "physical_npus": [0, 1],
        "request_shape": {
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "offered_requests_per_repetition": 200,
            "max_in_flight": args.concurrency_cap,
            "repetitions": 3,
        },
        "server": {
            "max_num_seqs": args.concurrency_cap,
            "max_num_batched_tokens": 16384,
            "max_model_len": 4096,
            "enforce_eager": True,
            "async_scheduling": False,
        },
        "beam": {
            "image": "vllm-ascend-beam:0.23.0",
            "use_s1": True,
            "use_s2": True,
            "ttft_slo_s": args.beam_ttft_slo,
            "tbt_slo_s": args.beam_tbt_slo,
            "profile": "beam/profile_synthetic.csv",
            "profile_calibrated_for_model_and_topology": False,
            "profile_sha256": (
                hashlib.sha256(profile_path.read_bytes()).hexdigest()
                if profile_path.is_file()
                else None
            ),
        },
        "vanilla_vllm": {
            "image": "quay.io/ascend/vllm-ascend:v0.23.0",
            "clock_policy": "host default dynamic DVFS; no static clock lock",
            "rated_clock_mhz": args.baseline_clock_mhz,
            "expected_idle_clock_mhz": 800,
        },
        "sampling": {
            "source": "read-only DSMI",
            "interval_s": 0.5,
            "power_units": "W",
            "energy_method": "trapezoidal integration of whole-device power",
        },
        "frequency_reset": {
            "rated_clock_mhz": args.baseline_clock_mhz,
            "verified_idle_clock_mhz": 800,
            "explicit_reset_before_and_after_each_backend_session": True,
            "vanilla_static_clock_lock": False,
        },
    }
    (root / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(root / "comparison.csv", records)
    energy_rows = []
    for record in records:
        for device in (0, 1):
            energy_rows.append(
                {
                    "backend": record["backend"],
                    "repetition": record["repetition"],
                    "device": device,
                    "energy_j": record[f"energy_npu{device}_j"],
                }
            )
    write_csv(root / "energy_by_device.csv", energy_rows)
    plot_results(root, records, request_sets)

    if args.prompt_tokens >= args.output_tokens:
        operating_point_note = (
            "- The requested BEAM prefill point of 1700 MHz is outside the checked-in BEAM NPU clock "
            "catalog (900–1650 MHz). Vanilla was intentionally run without a static clock lock at the "
            "host-default dynamic setting, rated at 1800 MHz."
        )
    else:
        operating_point_note = (
            f"- The requested 1300 MHz applies only to BEAM's synthetic-profile policy. Vanilla was "
            f"intentionally run without a static clock lock at the host-default dynamic setting, "
            f"rated at {args.baseline_clock_mhz} MHz."
        )

    lines = [
        f"# {root.name.replace('-', ' ').title()}",
        "",
        "## Workload and setup",
        "",
        f"- Model: Qwen3-8B, vLLM 0.23.0, tensor parallelism 2 on NPU 0 and 1.",
        "- Workload shape follows `experiments/req-shape-1.json`; synthetic prompts use the configured exact token counts.",
        f"- Synthetic request shape: {args.prompt_tokens} input tokens, "
        f"{args.output_tokens} output tokens; 200 offered requests per repetition.",
        f"- Server and client in-flight cap: {args.concurrency_cap}.",
        f"- Vanilla vLLM clock policy: host default dynamic DVFS without a static clock lock; "
        f"rated clock {args.baseline_clock_mhz} MHz, expected 800 MHz idle and up to "
        f"{args.baseline_clock_mhz} MHz under load.",
        f"- BEAM configured TTFT/TBT SLOs: {args.beam_ttft_slo:.3f}/{args.beam_tbt_slo:.3f} s.",
        "- BEAM S1/S2 used `tests/beam_wide_profile.py`'s synthetic profile; this was not calibrated for Qwen3-8B or TP2.",
        "- Read-only DSMI frequency and whole-device power samples were collected every 0.5 s for each NPU.",
        "- Sampled energy is trapezoidal integration of whole-device power; it includes idle/system power and is not an NPU energy-counter reading.",
        "- Both NPUs were explicitly reset to rated 1800 MHz before and after each backend session; each reset verified 800 MHz at idle. Vanilla vLLM ran without a static clock lock.",
        "- Warm-up traffic is retained under each backend's `warmup/` folder and excluded from the three measured repetitions.",
        "",
        "The existing `experiments/benchmarks.json` does not expand: its lane sweeps omit the required `max_num_batched_tokens`. "
        "`experiments/azure_mix.py` was not run because `/data1/datasets/azure-mix/scheduled_prompts.csv` is absent.",
        "",
        "## Results",
        "",
        "| Backend | Rep | Completed/200 | Shape valid | Mean TTFT (s) | P95 TTFT (s) | Mean TPOT (s) | P95 TPOT (s) | Output tok/s | NPU energy (J) | Mean MHz NPU0/1 |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| {backend} | {rep} | {completed}/{requests} | {shape} | {ttft} | {p95ttft} | {tpot} | {p95tpot} | {throughput} | {energy} | {freq0}/{freq1} |".format(
                backend=record["backend"],
                rep=record["repetition"],
                completed=record["completed"],
                requests=record["requests"],
                shape=record["shape_valid"],
                ttft=fmt(record["mean_ttft_s"]),
                p95ttft=fmt(record["p95_ttft_s"]),
                tpot=fmt(record["mean_tpot_s"]),
                p95tpot=fmt(record["p95_tpot_s"]),
                throughput=fmt(record["throughput_output_tokens_s"]),
                energy=fmt(record["energy_total_j"], 1),
                freq0=fmt(record["mean_freq_npu0_mhz"], 0),
                freq1=fmt(record["mean_freq_npu1_mhz"], 0),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "- Compare measured TTFT/TPOT, throughput, sampled clocks, and integrated device power; BEAM's profile-estimated energy is not a measured value.",
            operating_point_note,
            "- Confirm the expected 800 MHz idle and 1800 MHz loaded clocks from the raw per-device telemetry; idle and boundary samples can lower the reported mean.",
            "- TPOT is the request client's mean inter-content-chunk gap. It is not a hardware token-event counter.",
            "- Repetitions are reported separately; first-run effects and run-to-run variability are retained rather than discarded.",
            "- Inspect the saved request JSON/CSV and NPU sample CSVs for errors, token-shape validity, sampling gaps, and variability before drawing conclusions.",
            "",
            "## Plots",
            "",
            "![Request latency empirical CDF](latency_cdf.png)",
            "",
            "![NPU frequency and power](frequency_power.png)",
            "",
            "![Energy and throughput](energy_throughput.png)",
            "",
        ]
    )
    (root / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {root / 'analysis.md'} and plots; {len(records)} repetition rows.")


if __name__ == "__main__":
    main()
