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
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent


def _ensure_repo_on_path():
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_freq_module():
    _ensure_repo_on_path()
    try:
        from src import freq as freq_module

        return freq_module
    except ImportError:
        try:
            from . import freq as freq_module

            return freq_module
        except ImportError:
            src_dir = str(_SRC_DIR)
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            import freq as freq_module

            return freq_module


def _devices_from_arg(value):
    if value is None:
        return []
    devices = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        devices.append(int(item))
    return devices


def _run_command(command):
    return subprocess.run(command, capture_output=True, text=True)


def _extract_first_number(text, patterns):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                continue
    return None


def _npu_smi_power(device):
    if shutil.which("npu-smi") is None:
        return None
    commands = [
        ["npu-smi", "info", "-t", "power", "-i", str(device)],
        ["npu-smi", "info", "-t", "power", "-i", str(device), "-c", "0"],
    ]
    for command in commands:
        completed = _run_command(command)
        if completed.returncode == 0:
            return completed.stdout
    return None


def _metric_value(raw):
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        try:
            value = float(int(raw))
        except (TypeError, ValueError, OverflowError):
            return None
    if value != value:  # NaN
        return None
    return value


def _freq_info(device, retries=3, retry_delay_s=0.2):
    last = {"frequency": None, "volt": None}
    attempts = max(1, int(retries))
    for attempt in range(attempts):
        freq = _load_freq_module()
        info = freq.AscendML.get_info(int(device))
        last = {
            "frequency": _metric_value(info.get("aic_freq")),
            "volt": _metric_value(info.get("aic_volt")),
        }
        if last["frequency"] is not None or last["volt"] is not None:
            return last
        if attempt + 1 < attempts:
            time.sleep(retry_delay_s)
    return last


def _metric_value_legacy(raw):
    if raw is None:
        return None
    try:
        value = float(int(raw))
    except (TypeError, ValueError, OverflowError):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
    if value != value:  # NaN
        return None
    return value


def _normalize_device_sample(sample, metric_value=_metric_value):
    normalized = dict(sample)
    for key in ("power_sample", "frequency", "volt"):
        normalized[key] = metric_value(normalized.get(key))
    return normalized


def _warmup_freq_reads_legacy(devices):
    for device in devices:
        _freq_info(device, retries=2, retry_delay_s=0.1)


def _sample_device_legacy(device):
    """Committed sampling path: AscendML freq/volt, then ``npu-smi`` power."""
    freq_volt = _freq_info(device)
    text = _npu_smi_power(device)
    power = None
    if text:
        power = _extract_first_number(text, [r"([0-9]+(?:\.[0-9]+)?)\s*W", r"Power[^0-9]*([0-9]+(?:\.[0-9]+)?)"])
        if freq_volt.get("frequency") is None:
            freq_volt["frequency"] = _extract_first_number(
                text,
                [r"([0-9]+(?:\.[0-9]+)?)\s*MHz", r"freq[^0-9]*([0-9]+(?:\.[0-9]+)?)"],
            )
        if freq_volt.get("volt") is None:
            freq_volt["volt"] = _extract_first_number(
                text,
                [r"([0-9]+(?:\.[0-9]+)?)\s*V", r"volt[^0-9]*([0-9]+(?:\.[0-9]+)?)"],
            )
    return _normalize_device_sample(
        {
            "power_sample": power,
            "frequency": freq_volt.get("frequency"),
            "volt": freq_volt.get("volt"),
            "raw": text,
        },
        metric_value=_metric_value_legacy,
    )


def _warmup_freq_reads(devices):
    for device in devices:
        if _dsmi_metrics(device) is None:
            _freq_info(device, retries=2, retry_delay_s=0.1)


def _dsmi_metrics(device):
    try:
        freq = _load_freq_module()
    except Exception:
        return None
    getter = getattr(getattr(freq, "dsmi", None), "get_device_metrics", None)
    if getter is None:
        return None
    try:
        return getter(int(device))
    except Exception:
        return None


def _sample_device(device):
    """Default sampler: direct DSMI metrics with legacy fallbacks."""
    metrics = _dsmi_metrics(device) or {}
    power = metrics.get("power_w")
    freq_volt = {
        "frequency": metrics.get("frequency_mhz"),
        "volt": metrics.get("volt_mv"),
    }
    if freq_volt["frequency"] is None and freq_volt["volt"] is None:
        freq_volt = _freq_info(device)
    text = None
    if power is None:
        text = _npu_smi_power(device)
    if text:
        power = _extract_first_number(text, [r"([0-9]+(?:\.[0-9]+)?)\s*W", r"Power[^0-9]*([0-9]+(?:\.[0-9]+)?)"])
        if freq_volt.get("frequency") is None:
            freq_volt["frequency"] = _extract_first_number(
                text,
                [r"([0-9]+(?:\.[0-9]+)?)\s*MHz", r"freq[^0-9]*([0-9]+(?:\.[0-9]+)?)"],
            )
        if freq_volt.get("volt") is None:
            freq_volt["volt"] = _extract_first_number(
                text,
                [r"([0-9]+(?:\.[0-9]+)?)\s*V", r"volt[^0-9]*([0-9]+(?:\.[0-9]+)?)"],
            )
    return _normalize_device_sample(
        {
            "power_sample": power,
            "frequency": freq_volt.get("frequency"),
            "volt": freq_volt.get("volt"),
            "raw": text,
        }
    )


def snapshot(devices):
    timestamp = time.time()
    per_npu = {}
    powers = {}
    for device in devices:
        sample = _sample_device(device)
        per_npu[str(device)] = sample
        powers[str(device)] = sample.get("power_sample")
    return {
        "timestamp": timestamp,
        "devices": [int(device) for device in devices],
        "powers_watts": powers,
        "per_npu": per_npu,
    }


def delta(before, after):
    duration = max(0.0, float(after.get("timestamp", 0.0)) - float(before.get("timestamp", 0.0)))
    before_powers = before.get("powers_watts") or {}
    after_powers = after.get("powers_watts") or {}
    per_npu = {}
    total = 0.0
    for device in set(before_powers) | set(after_powers):
        p0 = float(before_powers.get(device) or 0.0)
        p1 = float(after_powers.get(device) or 0.0)
        joules = (p0 + p1) * 0.5 * duration
        per_npu[str(device)] = joules
        total += joules
    return {"per_npu": per_npu, "total_joules": total, "duration_s": duration}


def summarize_log(path):
    samples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    samples.sort(key=lambda item: float(item.get("timestamp", 0.0)))
    per_npu = {}
    sample_series = {}
    power_samples_series = []
    for sample in samples:
        timestamp = float(sample.get("timestamp", 0.0))
        current_per_npu = {}
        for device, values in (sample.get("per_npu") or {}).items():
            if device not in sample_series:
                sample_series[device] = []
            power = float(values.get("power_sample") or 0.0)
            freq = values.get("frequency")
            volt = values.get("volt")
            current_per_npu[device] = {"power_sample": power, "frequency": freq, "volt": volt}
            sample_series[device].append([timestamp, power, freq, volt])
        power_samples_series.append({"timestamp": timestamp, "per_npu": current_per_npu})
    for device, series in sample_series.items():
        joules = 0.0
        for left, right in zip(series, series[1:]):
            dt = max(0.0, float(right[0]) - float(left[0]))
            joules += (float(left[1] or 0.0) + float(right[1] or 0.0)) * 0.5 * dt
        per_npu[device] = joules
    total = sum(per_npu.values())
    duration = 0.0
    if len(samples) >= 2:
        duration = float(samples[-1].get("timestamp", 0.0)) - float(samples[0].get("timestamp", 0.0))
    return {
        "per_npu": per_npu,
        "total_joules": total,
        "samples": len(samples),
        "duration_s": duration,
        "sample_series": sample_series,
        "power_samples_series": power_samples_series,
    }


def set_freq(devices, mhz):
    try:
        freq = _load_freq_module()
    except Exception as exc:
        raise RuntimeError("freq module is required for frequency control") from exc
    if freq.dsmi is None:
        raise RuntimeError("freq module: DSMI library unavailable")
    for device in devices:
        freq.dsmi.set_aic_freq(int(device), int(mhz))


def build_parser():
    parser = argparse.ArgumentParser(prog="energy_meter.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check")

    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--output", required=True)
    snapshot_parser.add_argument("--devices", required=True)

    delta_parser = subparsers.add_parser("delta")
    delta_parser.add_argument("--before", required=True)
    delta_parser.add_argument("--after", required=True)

    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--output", required=True)
    monitor_parser.add_argument("--interval", default=1.0, type=float)
    monitor_parser.add_argument("--devices", required=True)
    monitor_parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the output log instead of truncating it on startup",
    )

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--log", required=True)

    freq_parser = subparsers.add_parser("freq")
    freq_parser.add_argument("--freq", required=True, type=int)
    freq_parser.add_argument("--devices", required=True)

    return parser


def _write_json(path, payload):
    target = os.path.expanduser(path)
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _read_json(path):
    with open(os.path.expanduser(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "check":
        return 0 if shutil.which("npu-smi") else 1
    if args.command == "snapshot":
        payload = snapshot(_devices_from_arg(args.devices))
        _write_json(args.output, payload)
        return 0
    if args.command == "delta":
        payload = delta(_read_json(args.before), _read_json(args.after))
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.command == "monitor":
        target = os.path.expanduser(args.output)
        directory = os.path.dirname(target)
        if directory:
            os.makedirs(directory, exist_ok=True)
        devices = _devices_from_arg(args.devices)
        _warmup_freq_reads(devices)
        period = max(0.01, float(args.interval))
        open_mode = "a" if args.append else "w"
        try:
            with open(target, open_mode, encoding="utf-8") as handle:
                while True:
                    started = time.monotonic()
                    payload = snapshot(devices)
                    handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    remaining = period - (time.monotonic() - started)
                    if remaining > 0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            return 0
    if args.command == "summarize":
        payload = summarize_log(args.log)
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.command == "freq":
        set_freq(_devices_from_arg(args.devices), args.freq)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
