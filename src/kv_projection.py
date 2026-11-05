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
import math
import threading
import time
import urllib.error
import urllib.request

SCHEMA_VERSION = "parsa.vllm.kv_projection.v1"
DEFAULT_QUANTILES = ("p50", "p90", "p95")


def _as_int(value, field):
    if isinstance(value, bool):
        raise TypeError("%s must be an integer" % field)
    return int(value)


def _as_float(value, field):
    if value is None:
        raise TypeError("%s is required" % field)
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise ValueError("%s must be finite" % field)
    return number


def _read_mapping(payload, key):
    value = payload.get(key) or {}
    if not isinstance(value, dict):
        raise TypeError("%s must be an object" % key)
    return value


def _quantile_field(mapping, preferred_key, legacy_key=None):
    value = mapping.get(preferred_key)
    if value is None and legacy_key is not None:
        value = mapping.get(legacy_key)
    return value


class KVProjectionSnapshot:

    __slots__ = (
        "schema_version",
        "ts_s",
        "engine_id",
        "block_size_tokens",
        "npu_blocks",
        "host_blocks",
        "npu_used_blocks",
        "npu_free_blocks",
        "host_used_blocks",
        "host_used_blocks_available",
        "active_requests",
        "horizon_iters",
        "peak_npu_blocks",
        "peak_utilization",
        "peak_batch_size",
        "time_to_below_utilization_ms",
        "projection",
    )

    def __init__(
        self,
        schema_version,
        ts_s,
        engine_id,
        block_size_tokens,
        npu_blocks,
        host_blocks,
        npu_used_blocks,
        npu_free_blocks,
        host_used_blocks,
        active_requests,
        horizon_iters,
        peak_npu_blocks,
        peak_utilization,
        peak_batch_size,
        time_to_below_utilization_ms,
        projection,
        host_used_blocks_available=True,
    ):
        self.schema_version = str(schema_version)
        self.ts_s = float(ts_s)
        self.engine_id = str(engine_id)
        self.block_size_tokens = int(block_size_tokens)
        self.npu_blocks = int(npu_blocks)
        self.host_blocks = int(host_blocks)
        self.npu_used_blocks = int(npu_used_blocks)
        self.npu_free_blocks = int(npu_free_blocks)
        self.host_used_blocks = int(host_used_blocks)
        self.host_used_blocks_available = bool(host_used_blocks_available)
        self.active_requests = int(active_requests)
        self.horizon_iters = int(horizon_iters)
        self.peak_npu_blocks = dict(peak_npu_blocks)
        self.peak_utilization = dict(peak_utilization)
        self.peak_batch_size = dict(peak_batch_size)
        self.time_to_below_utilization_ms = dict(time_to_below_utilization_ms)
        self.projection = dict(projection)

    @classmethod
    def from_payload(cls, payload, expected_engine_id=None):
        if not isinstance(payload, dict):
            raise TypeError("kv projection payload must be an object")
        schema_version = str(payload.get("schema_version") or "")
        if schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported kv projection schema version: %s" % schema_version)
        ts_s = _as_float(payload.get("ts_s"), "ts_s")
        engine_id = str(payload.get("engine_id") or "")
        if not engine_id:
            raise ValueError("engine_id must be non-empty")
        if expected_engine_id and str(expected_engine_id) != engine_id:
            raise ValueError("engine_id mismatch: expected %s got %s" % (expected_engine_id, engine_id))
        block_size_tokens = _as_int(payload.get("block_size_tokens"), "block_size_tokens")
        capacity = _read_mapping(payload, "capacity")
        current = _read_mapping(payload, "current")
        projection_basis = _read_mapping(payload, "projection_basis")
        summary = _read_mapping(payload, "summary")

        npu_blocks = _as_int(_quantile_field(capacity, "npu_blocks", "gpu_blocks"), "capacity.npu_blocks")
        host_blocks = _as_int(_quantile_field(capacity, "host_blocks", "cpu_blocks") or 0, "capacity.host_blocks")
        npu_used_blocks = _as_int(
            _quantile_field(current, "npu_used_blocks", "gpu_used_blocks"),
            "current.npu_used_blocks",
        )
        npu_free_blocks = _as_int(
            _quantile_field(current, "npu_free_blocks", "gpu_free_blocks"),
            "current.npu_free_blocks",
        )
        raw_host_used_blocks = _quantile_field(current, "host_used_blocks", "swap_used_blocks")
        host_used_blocks_available = (
            isinstance(raw_host_used_blocks, int) and not isinstance(raw_host_used_blocks, bool)
            and raw_host_used_blocks >= 0
        )
        host_used_blocks = _as_int(raw_host_used_blocks or 0, "current.host_used_blocks")
        active_requests = _as_int(current.get("active_requests"), "current.active_requests")
        horizon_iters = _as_int(projection_basis.get("horizon_iters"), "projection_basis.horizon_iters")

        peak_npu_blocks = {}
        raw_peak_blocks = _read_mapping(summary, "peak_npu_blocks" if "peak_npu_blocks" in summary else "peak_gpu_blocks")
        peak_utilization = _read_mapping(summary, "peak_utilization")
        peak_batch_size = _read_mapping(summary, "peak_batch_size")
        for quantile in DEFAULT_QUANTILES:
            peak_npu_blocks[quantile] = _as_int(raw_peak_blocks.get(quantile), "summary.peak_npu_blocks.%s" % quantile)
            peak_utilization[quantile] = _as_float(peak_utilization.get(quantile), "summary.peak_utilization.%s" % quantile)
            peak_batch_size[quantile] = _as_int(peak_batch_size.get(quantile), "summary.peak_batch_size.%s" % quantile)

        time_to_below_utilization_ms = {}
        raw_recovery = _read_mapping(summary, "time_to_below_utilization_ms")
        for threshold in ("0.80", "0.70", "0.60"):
            time_to_below_utilization_ms[threshold] = _as_int(
                raw_recovery.get(threshold, -1),
                "summary.time_to_below_utilization_ms.%s" % threshold,
            )

        raw_projection = _read_mapping(payload, "projection")
        projection = {}
        for quantile in DEFAULT_QUANTILES:
            items = raw_projection.get(quantile) or []
            if not isinstance(items, list):
                raise TypeError("projection.%s must be a list" % quantile)
            normalized = []
            for item in items:
                if not isinstance(item, dict):
                    raise TypeError("projection.%s entries must be objects" % quantile)
                normalized.append(
                    {
                        "iter_offset": _as_int(item.get("iter_offset"), "projection.%s.iter_offset" % quantile),
                        "batch_size": _as_int(item.get("batch_size"), "projection.%s.batch_size" % quantile),
                        "npu_kv_blocks": _as_int(
                            _quantile_field(item, "npu_kv_blocks", "gpu_kv_blocks"),
                            "projection.%s.npu_kv_blocks" % quantile,
                        ),
                    }
                )
            projection[quantile] = normalized

        return cls(
            schema_version=schema_version,
            ts_s=ts_s,
            engine_id=engine_id,
            block_size_tokens=block_size_tokens,
            npu_blocks=npu_blocks,
            host_blocks=host_blocks,
            npu_used_blocks=npu_used_blocks,
            npu_free_blocks=npu_free_blocks,
            host_used_blocks=host_used_blocks,
            active_requests=active_requests,
            horizon_iters=horizon_iters,
            peak_npu_blocks=peak_npu_blocks,
            peak_utilization=peak_utilization,
            peak_batch_size=peak_batch_size,
            time_to_below_utilization_ms=time_to_below_utilization_ms,
            projection=projection,
            host_used_blocks_available=host_used_blocks_available,
        )

    def current_utilization(self):
        if self.npu_blocks <= 0:
            return 0.0
        return max(0.0, min(1.0, float(self.npu_used_blocks) / float(self.npu_blocks)))

    def peak_blocks(self, quantile="p95"):
        return int(self.peak_npu_blocks.get(str(quantile), self.peak_npu_blocks.get("p95", 0)))

    def peak_util(self, quantile="p95"):
        return float(self.peak_utilization.get(str(quantile), self.peak_utilization.get("p95", 0.0)))

    def peak_batch(self, quantile="p95"):
        return int(self.peak_batch_size.get(str(quantile), self.peak_batch_size.get("p95", 0)))

    def recovery_seconds(self, threshold="0.70"):
        raw = int(self.time_to_below_utilization_ms.get(str(threshold), -1))
        if raw < 0:
            return None
        return float(raw) / 1000.0

    def delta_blocks_for_tokens(self, total_tokens):
        total_tokens = max(0.0, float(total_tokens or 0.0))
        if total_tokens <= 0 or self.block_size_tokens <= 0:
            return 0
        return int(math.ceil(total_tokens / float(self.block_size_tokens)))

    def peak_util_if_admitted(self, delta_blocks=0, quantile="p95"):
        if self.npu_blocks <= 0:
            return 1.0
        blocks = float(self.peak_blocks(quantile)) + float(delta_blocks or 0)
        util = blocks / float(self.npu_blocks)
        return max(0.0, min(2.0, util))

    def to_metrics_dict(self):
        return {
            "engine_id": self.engine_id,
            "snapshot_ts_s": self.ts_s,
            "current_utilization": self.current_utilization(),
            "peak_util_p95": self.peak_util("p95"),
            "peak_util_p90": self.peak_util("p90"),
            "npu_used_blocks": self.npu_used_blocks,
            "npu_blocks": self.npu_blocks,
            "npu_free_blocks": self.npu_free_blocks,
            "host_used_blocks": self.host_used_blocks,
            "host_blocks": self.host_blocks,
            "active_requests": self.active_requests,
            "block_size_tokens": self.block_size_tokens,
        }


def admin_kv_projection_url(backend_url):
    return str(backend_url).rstrip("/") + "/admin/parsa/kv-projection"


def fetch_kv_projection_snapshot_sync(url, timeout_s, expected_engine_id=None):
    timeout_s = max(0.01, float(timeout_s or 1.0))
    request = urllib.request.Request(str(url), headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = getattr(response, "status", None) or response.getcode()
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise RuntimeError("kv projection status %s from %s: %s" % (exc.code, url, text[:200])) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("kv projection request failed for %s: %s" % (url, exc)) from exc
    if int(status) != 200:
        raise RuntimeError("kv projection status %s from %s: %s" % (status, url, body[:200]))
    payload = json.loads(body)
    return KVProjectionSnapshot.from_payload(payload, expected_engine_id=expected_engine_id)


class _KVProjectionEntry:
    __slots__ = ("snapshot", "last_success_s", "last_error", "last_error_s")

    def __init__(self):
        self.snapshot = None
        self.last_success_s = 0.0
        self.last_error = ""
        self.last_error_s = 0.0


class KVProjectionStore:

    __slots__ = ("config", "_lock", "_entries")

    def __init__(self, config=None):
        self.config = dict(config or {})
        self._lock = threading.RLock()
        self._entries = {}

    def _entry(self, preset_id):
        key = str(preset_id)
        entry = self._entries.get(key)
        if entry is None:
            entry = _KVProjectionEntry()
            self._entries[key] = entry
        return entry

    def stale_after_s(self):
        return max(0.01, float(self.config.get("stale_after_ms", 250) or 250) / 1000.0)

    def update(self, preset_id, snapshot):
        with self._lock:
            entry = self._entry(preset_id)
            entry.snapshot = snapshot
            entry.last_success_s = time.time()
            entry.last_error = ""
            entry.last_error_s = 0.0

    def record_error(self, preset_id, error):
        with self._lock:
            entry = self._entry(preset_id)
            entry.last_error = str(error or "")
            entry.last_error_s = time.time()

    def get_snapshot(self, preset_id, now_s=None, allow_stale=False):
        with self._lock:
            entry = self._entries.get(str(preset_id))
            if entry is None or entry.snapshot is None:
                return None
            if allow_stale:
                return entry.snapshot
            now_value = time.time() if now_s is None else float(now_s)
            if now_value - float(entry.snapshot.ts_s) > self.stale_after_s():
                return None
            return entry.snapshot

    def status(self, preset_id, now_s=None):
        with self._lock:
            entry = self._entries.get(str(preset_id))
            if entry is None:
                return {"available": False, "stale": True, "last_error": "uninitialized"}
            snapshot = entry.snapshot
            stale = True
            if snapshot is not None:
                now_value = time.time() if now_s is None else float(now_s)
                stale = (now_value - float(snapshot.ts_s)) > self.stale_after_s()
            return {
                "available": snapshot is not None and not stale,
                "stale": stale,
                "last_error": entry.last_error,
                "last_error_s": entry.last_error_s,
                "snapshot_ts_s": None if snapshot is None else snapshot.ts_s,
            }

    def snapshot(self):
        with self._lock:
            result = {}
            now_s = time.time()
            for preset_id, entry in self._entries.items():
                result[preset_id] = self.status(preset_id, now_s=now_s)
            return result


async def fetch_kv_projection_snapshot(session, url, timeout_s, expected_engine_id=None):
    async def _fetch():
        async with session.get(url) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError("kv projection status %s from %s: %s" % (response.status, url, text[:200]))
            payload = await response.json(content_type=None)
            return KVProjectionSnapshot.from_payload(payload, expected_engine_id=expected_engine_id)

    return await asyncio.wait_for(_fetch(), timeout=max(0.01, float(timeout_s or 0.05)))
