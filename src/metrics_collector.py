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

import asyncio
import json
import os
import time
import urllib.request


def resolve_collect_dir():
    path = os.environ.get("ENERGY_AWARE_COLLECT_DIR")
    if path:
        return os.path.expanduser(path)
    return os.path.expanduser("~/Projects/workspace/collected")


def prometheus_sum_count_mean(text, metric_prefix):
    total = 0.0
    count = 0.0
    saw_sum = False
    saw_count = False
    prefix_sum = str(metric_prefix) + "_sum"
    prefix_count = str(metric_prefix) + "_count"
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            value = float(parts[-1])
        except Exception:
            continue
        if name == prefix_sum:
            total += value
            saw_sum = True
        elif name == prefix_count:
            count += value
            saw_count = True
    if not saw_sum or not saw_count or count <= 0:
        return None
    return total / count


class BaselineCalibration:

    __slots__ = ("baseline_ttft_s", "baseline_tbt_s", "ewma_alpha")

    def __init__(self, baseline_ttft_s, baseline_tbt_s, ewma_alpha=0.15):
        self.baseline_ttft_s = float(baseline_ttft_s)
        self.baseline_tbt_s = float(baseline_tbt_s)
        self.ewma_alpha = float(ewma_alpha)
        if self.baseline_ttft_s <= 0 or self.baseline_tbt_s <= 0:
            raise ValueError("baseline metrics must be > 0")

    def to_dict(self):
        return {
            "baseline_ttft_s": self.baseline_ttft_s,
            "baseline_tbt_s": self.baseline_tbt_s,
            "ewma_alpha": self.ewma_alpha,
        }


def _clip(value, low, high):
    return max(float(low), min(float(high), float(value)))


def update_scales_from_observed(scales, cfg, obs_ttft_s=None, obs_tbt_s=None):
    current = scales.snapshot()
    ttft = current.get("ttft", 1.0)
    tbt = current.get("tbt", 1.0)
    alpha = cfg.ewma_alpha
    if obs_ttft_s is not None and float(obs_ttft_s) > 0:
        target = _clip(float(obs_ttft_s) / cfg.baseline_ttft_s, 0.5, 2.0)
        ttft = (1.0 - alpha) * ttft + alpha * target
    if obs_tbt_s is not None and float(obs_tbt_s) > 0:
        target = _clip(float(obs_tbt_s) / cfg.baseline_tbt_s, 0.5, 2.0)
        tbt = (1.0 - alpha) * tbt + alpha * target
    scales.update(ttft=ttft, tbt=tbt)
    return scales.snapshot()


def _fetch_metrics_text(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode("utf-8", "replace")


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
    os.replace(tmp_path, path)


async def calibration_loop(metrics_base_urls, scales, cfg, interval_s=1.0):
    collect_dir = resolve_collect_dir()
    os.makedirs(collect_dir, exist_ok=True)
    latest_path = os.path.join(collect_dir, "energy_aware_calibration_latest.json")
    append_path = os.path.join(collect_dir, "energy_aware_calibration.jsonl")
    interval_s = max(0.2, float(interval_s or 1.0))
    urls = list(metrics_base_urls or [])
    while True:
        await asyncio.sleep(interval_s)
        per_backend = []
        observed_ttft = []
        observed_tbt = []
        for url in urls:
            try:
                text = await asyncio.to_thread(_fetch_metrics_text, url)
                ttft = prometheus_sum_count_mean(text, "vllm:time_to_first_token_seconds")
                tbt = prometheus_sum_count_mean(text, "vllm:inter_token_latency_seconds")
                if ttft is not None:
                    observed_ttft.append(ttft)
                if tbt is not None:
                    observed_tbt.append(tbt)
                per_backend.append({"url": url, "ttft_s": ttft, "tbt_s": tbt})
            except Exception as exc:
                per_backend.append({"url": url, "error": str(exc)})
        obs_ttft = sum(observed_ttft) / len(observed_ttft) if observed_ttft else None
        obs_tbt = sum(observed_tbt) / len(observed_tbt) if observed_tbt else None
        snapshot = update_scales_from_observed(scales, cfg, obs_ttft_s=obs_ttft, obs_tbt_s=obs_tbt)
        payload = {
            "timestamp_s": time.time(),
            "scales": snapshot,
            "obs_ttft_s": obs_ttft,
            "obs_tbt_s": obs_tbt,
            "config": cfg.to_dict(),
            "per_backend": per_backend,
        }
        _atomic_write_json(latest_path, payload)
        with open(append_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
