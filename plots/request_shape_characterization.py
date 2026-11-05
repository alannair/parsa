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

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plots.style import CONCURRENT_REQUESTS_LABEL, global_legend, plot_color, set_size, tick_style
from plots.dataload import load_result_json_rows, trim_idle_edges


def _png_path(value):
    path = Path(value)
    if path.suffix.lower() != ".png":
        raise argparse.ArgumentTypeError("--output must end in .png")
    return path


def load_runs(root):
    return load_result_json_rows(root, "profile-request-shape-*/result.json")


def _power_samples(series):
    if not isinstance(series, list) or not series:
        raise ValueError("result row has no power_samples_series")
    samples = []
    for sample in series:
        timestamp = float(sample.get("timestamp"))
        per_npu = sample.get("per_npu")
        if not math.isfinite(timestamp) or not isinstance(per_npu, dict) or not per_npu:
            raise ValueError("power_samples_series contains an invalid sample")
        watts = 0.0
        devices = {}
        for device, values in per_npu.items():
            power = float(values.get("power_sample"))
            if not math.isfinite(power):
                raise ValueError("power_samples_series contains invalid wattage")
            watts += power
            frequency = values.get("frequency")
            if frequency is not None:
                frequency = float(frequency)
                if not math.isfinite(frequency):
                    raise ValueError("power_samples_series contains invalid frequency")
            devices[str(device)] = {"power": power, "frequency": frequency}
        samples.append((timestamp, watts, devices))
    return samples


def _power_duration_s(series):
    timestamps = [timestamp for timestamp, _, _ in _power_samples(series)]
    duration = max(timestamps) - min(timestamps)
    if duration <= 0:
        raise ValueError("power_samples_series must span a positive duration")
    return duration


def _static_power_watts(data):
    watts_by_device = {}
    devices = set()
    for series in data["power_samples_series"]:
        for _, _, sample in _power_samples(series):
            for device, values in sample.items():
                devices.add(device)
                if values["frequency"] == 800:
                    watts_by_device.setdefault(device, []).append(values["power"])
    missing = sorted(devices - set(watts_by_device))
    if missing:
        raise ValueError("no 800 MHz power samples for monitored devices: %s" % ", ".join(missing))
    return sum(sum(watts) / len(watts) for watts in watts_by_device.values())


def _legacy_npu_zero_idle_power_watts(data):
    if "serving_condition" in data.columns:
        conditions = data["serving_condition"].fillna("legacy")
        data = data[conditions == "legacy"]
    watts = []
    for series in data["power_samples_series"]:
        for _, _, sample in _power_samples(series):
            values = sample.get("0")
            if values is not None and values["frequency"] == 800:
                watts.append(values["power"])
    if not watts:
        raise ValueError("no legacy NPU 0 power samples at 800 MHz")
    return sum(watts) / len(watts)


def _power_npu_count(series):
    devices = {frozenset(sample) for _, _, sample in _power_samples(series)}
    if len(devices) != 1:
        raise ValueError("power_samples_series changes monitored NPUs")
    return len(devices.pop())


def summarize_runs(frame, total_energy=False, serving_condition=None, static_power_per_npu_watts=None):
    required = {"request_shape", "frequency_mhz", "max_num_seqs", "num_requests", "shape_valid", "power_samples_series"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("result rows missing: %s" % ", ".join(sorted(missing)))
    data = frame.copy()
    for column in ("frequency_mhz", "max_num_seqs", "num_requests"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[data["shape_valid"] == True].dropna(subset=["frequency_mhz", "max_num_seqs", "num_requests"])
    data = data[data["num_requests"] > 0]
    if data.empty:
        raise ValueError("no shape-valid positive-request runs")
    if "serving_condition" not in data.columns:
        data = data.assign(serving_condition="legacy")
    else:
        data = data.assign(serving_condition=data["serving_condition"].fillna("legacy"))
    if serving_condition is not None:
        data = data[data["serving_condition"] == serving_condition]
        if data.empty:
            raise ValueError("no shape-valid positive-request runs for serving condition: " + serving_condition)
    conditions = sorted(data["serving_condition"].unique())
    if len(conditions) > 1:
        raise ValueError("multiple serving conditions: %s; use --serving-condition to select one" % ", ".join(conditions))
    # Calibration must see the original 800 MHz samples before trimming.
    if not total_energy:
        if static_power_per_npu_watts is None:
            static_power_watts = _static_power_watts(data)
            data = data.assign(static_power_watts=static_power_watts)
        else:
            static_power_watts = float(static_power_per_npu_watts)
            if not math.isfinite(static_power_watts) or static_power_watts < 0:
                raise ValueError("static_power_per_npu_watts must be finite and nonnegative")
            data = data.assign(static_power_watts=data.power_samples_series.map(_power_npu_count) * static_power_watts)

    trimmed, energies = [], []
    for index, row in data.iterrows():
        try:
            series = trim_idle_edges(row["power_samples_series"])
        except ValueError as exc:
            source = row.get("_result_json_path", "result row %s" % index)
            raise ValueError("%s: %s" % (source, exc)) from exc
        samples = _power_samples(series)
        energies.append(sum(
            (left[1] + right[1]) * 0.5 * (right[0] - left[0])
            for left, right in zip(samples, samples[1:])
        ))
        trimmed.append(series)
    data = data.assign(energy_joules=energies, power_samples_series=trimmed)

    if total_energy:
        data = data.assign(energy_per_request=lambda d: d.energy_joules / d.num_requests)
    else:
        data = data.assign(energy_per_request=lambda d: (
            d.energy_joules - d.power_samples_series.map(_power_duration_s) * d.static_power_watts
        ) / d.num_requests)
    aggregations = {
        "energy_per_request_mean": ("energy_per_request", "mean"),
        "energy_per_request_std": ("energy_per_request", "std"),
        "runs": ("energy_per_request", "count"),
    }
    if not total_energy:
        aggregations["static_power_watts"] = ("static_power_watts", "first")
    return data.groupby(["request_shape", "frequency_mhz", "max_num_seqs"], as_index=False).agg(
        **aggregations,
    )


def plot_request_shapes(summary, total_energy=False, version=None):
    shapes = [("prefill-heavy", "Prefill-heavy (1000, 1)"), ("decode-heavy", "Decode-heavy (1, 1000)")]
    fig, axes = plt.subplots(1, 2, figsize=(set_size(350)[0] * 2, set_size(350)[1]))
    for ax, (shape, title) in zip(axes, shapes):
        data = summary[summary["request_shape"] == shape]
        frequencies = sorted(data["frequency_mhz"].unique())
        groups = list(data.groupby("max_num_seqs", sort=True))
        width = 0.8 / len(groups) if groups else 0.8
        frequency_positions = {frequency: index for index, frequency in enumerate(frequencies)}
        for index, (concurrency, group) in enumerate(groups):
            group = group.sort_values("frequency_mhz")
            positions = [frequency_positions[frequency] + (index - (len(groups) - 1) / 2) * width
                         for frequency in group["frequency_mhz"]]
            ax.bar(positions, group["energy_per_request_mean"], width=width,
                   yerr=group["energy_per_request_std"].fillna(0), capsize=3,
                   color=plot_color(index), label=str(int(concurrency)))
        ax.set_title(title)
        ax.set_xlabel("NPU frequency (MHz)")
        tick_style(ax)
        ax.set_xticks(range(len(frequencies)), [str(int(frequency)) for frequency in frequencies])
        if not data.empty:
            ax.set_ylim(0, data["energy_per_request_mean"].max() * 1.2)
    axes[0].set_ylabel("Trimmed total energy per request (J)" if total_energy else "Trimmed static-power-subtracted energy per request (J)")
    global_legend(axes[1], title=CONCURRENT_REQUESTS_LABEL, ncol=4)
    if version:
        fig.suptitle("vLLM " + version)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(prog="plots.request_shape_characterization")
    parser.add_argument("--runs-root", default="experiments/_runs")
    parser.add_argument("--output", required=True, type=_png_path)
    parser.add_argument("--total-energy", action="store_true", help="Plot trimmed measured energy without static-power deduction")
    parser.add_argument("--serving-condition", help="Select one serving condition (use legacy for rows without metadata)")
    parser.add_argument("--version", help="Version label for the plot title")
    args = parser.parse_args(argv)
    runs = load_runs(args.runs_root)
    summary = summarize_runs(runs, total_energy=args.total_energy, serving_condition=args.serving_condition,
                             static_power_per_npu_watts=None if args.total_energy else _legacy_npu_zero_idle_power_watts(runs))
    figure = plot_request_shapes(summary, total_energy=args.total_energy,
                                 version=args.version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
