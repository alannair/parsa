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
import sys
import time
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent


def _ensure_repo_on_path():
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_kv_projection_module():
    _ensure_repo_on_path()
    try:
        from src import kv_projection as kv_module

        return kv_module
    except ImportError:
        try:
            from . import kv_projection as kv_module

            return kv_module
        except ImportError:
            src_dir = str(_SRC_DIR)
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            import kv_projection as kv_module

            return kv_module


def _targets_from_arg(value):
    targets = {}
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("target %r must use preset_id=backend_url form" % item)
        preset_id, backend_url = item.split("=", 1)
        preset_id = preset_id.strip()
        backend_url = backend_url.strip().rstrip("/")
        if not preset_id or not backend_url:
            raise ValueError("target %r must use preset_id=backend_url form" % item)
        targets[preset_id] = backend_url
    return targets


def _percentile(values, quantile):
    cleaned = sorted(float(item) for item in values if item is not None)
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    index = int(round((len(cleaned) - 1) * float(quantile)))
    index = max(0, min(len(cleaned) - 1, index))
    return cleaned[index]


def _mean(values):
    cleaned = [float(item) for item in values if item is not None]
    if not cleaned:
        return None
    return sum(cleaned) / float(len(cleaned))


def _fetch_lane_metrics(kv_module, preset_id, backend_url, timeout_s):
    url = kv_module.admin_kv_projection_url(backend_url)
    snapshot = kv_module.fetch_kv_projection_snapshot_sync(url, timeout_s, expected_engine_id=preset_id)
    metrics = snapshot.to_metrics_dict()
    metrics["preset_id"] = str(preset_id)
    metrics["backend_url"] = str(backend_url)
    return metrics


def snapshot(targets, timeout_s=1.0):
    kv_module = _load_kv_projection_module()
    timestamp = time.time()
    per_lane = {}
    for preset_id, backend_url in targets.items():
        try:
            per_lane[str(preset_id)] = _fetch_lane_metrics(kv_module, preset_id, backend_url, timeout_s)
        except Exception as exc:
            per_lane[str(preset_id)] = {
                "preset_id": str(preset_id),
                "backend_url": str(backend_url),
                "current_utilization": None,
                "peak_util_p95": None,
                "peak_util_p90": None,
                "npu_used_blocks": None,
                "npu_blocks": None,
                "npu_free_blocks": None,
                "host_used_blocks": None,
                "host_blocks": None,
                "active_requests": None,
                "block_size_tokens": None,
                "error": str(exc),
            }
    return {
        "timestamp": timestamp,
        "per_lane": per_lane,
    }


def summarize_log(path):
    samples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    samples.sort(key=lambda item: float(item.get("timestamp", 0.0)))
    sample_series = {}
    kv_samples_series = []
    per_lane_summary = {}
    host_swap_samples = 0
    for sample in samples:
        timestamp = float(sample.get("timestamp", 0.0))
        current_per_lane = {}
        for preset_id, values in (sample.get("per_lane") or {}).items():
            if preset_id not in sample_series:
                sample_series[preset_id] = []
            current_util = values.get("current_utilization")
            peak_util = values.get("peak_util_p95")
            host_used = int(values.get("host_used_blocks") or 0)
            if host_used > 0:
                host_swap_samples += 1
            row = {
                "current_utilization": current_util,
                "peak_util_p95": peak_util,
                "host_used_blocks": host_used,
                "active_requests": values.get("active_requests"),
                "error": values.get("error"),
            }
            current_per_lane[preset_id] = row
            sample_series[preset_id].append([timestamp, current_util, peak_util, host_used])
            lane_stats = per_lane_summary.setdefault(
                preset_id,
                {
                    "current_utils": [],
                    "peak_utils": [],
                    "errors": 0,
                },
            )
            if row.get("error"):
                lane_stats["errors"] += 1
            if current_util is not None:
                lane_stats["current_utils"].append(current_util)
            if peak_util is not None:
                lane_stats["peak_utils"].append(peak_util)
        kv_samples_series.append({"timestamp": timestamp, "per_lane": current_per_lane})

    by_lane = {}
    all_current = []
    all_peak = []
    for preset_id, lane_stats in per_lane_summary.items():
        current_mean = _mean(lane_stats["current_utils"])
        current_p95 = _percentile(lane_stats["current_utils"], 0.95)
        current_peak = max(lane_stats["current_utils"]) if lane_stats["current_utils"] else None
        peak_mean = _mean(lane_stats["peak_utils"])
        by_lane[preset_id] = {
            "mean_utilization": current_mean,
            "p95_utilization": current_p95,
            "peak_utilization": current_peak,
            "mean_peak_util_p95": peak_mean,
            "samples": len(lane_stats["current_utils"]),
            "errors": lane_stats["errors"],
        }
        all_current.extend(lane_stats["current_utils"])
        all_peak.extend(lane_stats["peak_utils"])

    duration = 0.0
    if len(samples) >= 2:
        duration = float(samples[-1].get("timestamp", 0.0)) - float(samples[0].get("timestamp", 0.0))

    return {
        "samples": len(samples),
        "duration_s": duration,
        "host_swap_samples": host_swap_samples,
        "mean_utilization": _mean(all_current),
        "p95_utilization": _percentile(all_current, 0.95),
        "peak_utilization": max(all_current) if all_current else None,
        "mean_peak_util_p95": _mean(all_peak),
        "by_lane": by_lane,
        "sample_series": sample_series,
        "kv_samples_series": kv_samples_series,
    }


def build_parser():
    parser = argparse.ArgumentParser(prog="kv_meter.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--output", required=True)
    snapshot_parser.add_argument("--targets", required=True)
    snapshot_parser.add_argument("--timeout-s", default=1.0, type=float)

    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--output", required=True)
    monitor_parser.add_argument("--interval", default=1.0, type=float)
    monitor_parser.add_argument("--targets", required=True)
    monitor_parser.add_argument("--timeout-s", default=1.0, type=float)
    monitor_parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the output log instead of truncating it on startup",
    )

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--log", required=True)

    return parser


def _write_json(path, payload):
    target = os.path.expanduser(path)
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "snapshot":
        payload = snapshot(_targets_from_arg(args.targets), timeout_s=args.timeout_s)
        _write_json(args.output, payload)
        return 0
    if args.command == "monitor":
        target = os.path.expanduser(args.output)
        directory = os.path.dirname(target)
        if directory:
            os.makedirs(directory, exist_ok=True)
        targets = _targets_from_arg(args.targets)
        open_mode = "a" if args.append else "w"
        try:
            with open(target, open_mode, encoding="utf-8") as handle:
                while True:
                    payload = snapshot(targets, timeout_s=args.timeout_s)
                    handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    time.sleep(max(0.01, float(args.interval)))
        except KeyboardInterrupt:
            return 0
    if args.command == "summarize":
        payload = summarize_log(args.log)
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
