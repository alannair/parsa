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

import json
import math
import threading
from types import MappingProxyType
from typing import NamedTuple


COEFF_KEYS = (
    "a_ttft",
    "b_ttft",
    "a_tbt",
    "b_tbt",
    "a_ep",
    "b_ep",
    "a_ed",
    "b_ed",
)


class ProfileEstimate:

    __slots__ = ("ttft_s", "tbt_s", "e_prefill_j", "e_decode_per_tok_j")

    def __init__(self, ttft_s=0.0, tbt_s=0.0, e_prefill_j=0.0, e_decode_per_tok_j=0.0):
        self.ttft_s = float(ttft_s)
        self.tbt_s = float(tbt_s)
        self.e_prefill_j = float(e_prefill_j)
        self.e_decode_per_tok_j = float(e_decode_per_tok_j)

    def latency_s(self, l_out, queue_delay_s=0.0):
        queue = max(0.0, float(queue_delay_s or 0.0))
        ttft = max(0.0, self.ttft_s)
        tbt = max(0.0, self.tbt_s)
        return queue + ttft + max(0.0, float(l_out or 0.0)) * tbt

    def energy_j(self, l_out):
        return max(0.0, self.e_prefill_j) + max(0.0, float(l_out or 0.0)) * max(0.0, self.e_decode_per_tok_j)

    def power_w(self, l_out, queue_delay_s=0.0):
        latency = self.latency_s(l_out=l_out, queue_delay_s=queue_delay_s)
        if latency <= 0:
            return 0.0
        return self.energy_j(l_out) / latency


class LinearProfile:

    __slots__ = COEFF_KEYS

    def __init__(self, a_ttft=0.0, b_ttft=0.0, a_tbt=0.0, b_tbt=0.0, a_ep=0.0, b_ep=0.0, a_ed=0.0, b_ed=0.0):
        self.a_ttft = float(a_ttft)
        self.b_ttft = float(b_ttft)
        self.a_tbt = float(a_tbt)
        self.b_tbt = float(b_tbt)
        self.a_ep = float(a_ep)
        self.b_ep = float(b_ep)
        self.a_ed = float(a_ed)
        self.b_ed = float(b_ed)

    def estimate(self, l_in, l_out):
        _ = l_out
        l_in = float(l_in or 0.0)
        return ProfileEstimate(
            ttft_s=self.a_ttft + self.b_ttft * l_in,
            tbt_s=self.a_tbt + self.b_tbt * l_in,
            e_prefill_j=self.a_ep + self.b_ep * l_in,
            e_decode_per_tok_j=self.a_ed + self.b_ed * l_in,
        )


class DictProfileTable:

    __slots__ = ("profiles",)

    def __init__(self, profiles):
        self.profiles = dict(profiles or {})

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, mapping):
        if not isinstance(mapping, dict):
            raise TypeError("profile JSON must be an object")
        profiles = {}
        for preset_id, coeffs in mapping.items():
            if not isinstance(coeffs, dict):
                raise TypeError("profile entry for %s must be an object" % preset_id)
            values = {}
            for key in COEFF_KEYS:
                values[key] = float(coeffs.get(key, 0.0) or 0.0)
            profiles[str(preset_id)] = LinearProfile(**values)
        return cls(profiles)

    def estimate(self, preset_id, l_in, l_out):
        profile = self.profiles.get(preset_id)
        if profile is None:
            raise KeyError(preset_id)
        return profile.estimate(l_in=l_in, l_out=l_out)


class ProfileBinding(NamedTuple):
    lane_id: str
    op_id: str
    generation: int
    coefficients: object


class OperatingPointProfileTable:
    """Generation-checked lane profile bindings backed by a validated catalog."""

    __slots__ = ("_bindings", "_lock", "_catalog")

    def __init__(self, catalog):
        self._lock = threading.RLock()
        self._catalog = catalog
        self._bindings = {
            lane: ProfileBinding(lane, op, 0, MappingProxyType(dict(catalog["operating_points"][op]["profile"])))
            for lane, op in catalog["initial_bank"].items()
        }

    def estimate(self, preset_id, l_in, l_out):
        with self._lock:
            binding = self._bindings.get(preset_id)
        if binding is None:
            raise KeyError(preset_id)
        return LinearProfile(**binding.coefficients).estimate(l_in=l_in, l_out=l_out)

    def prepare_binding(self, lane_id, op_id, generation):
        _integer = lambda x: isinstance(x, int) and not isinstance(x, bool)
        if not _integer(generation) or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        ops = self._catalog["operating_points"]
        if op_id not in ops:
            raise KeyError(op_id)
        profile = dict(ops[op_id]["profile"])
        if set(profile) != set(COEFF_KEYS) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in profile.values()):
            raise ValueError("invalid profile coefficients")
        with self._lock:
            if lane_id not in self._bindings:
                raise KeyError(lane_id)
        return ProfileBinding(lane_id, op_id, generation, MappingProxyType(profile))

    def activate_binding(self, binding, expected_generation):
        if not isinstance(binding, ProfileBinding) or isinstance(expected_generation, bool) or not isinstance(expected_generation, int):
            raise ValueError("invalid profile binding or expected generation")
        if isinstance(binding.generation, bool) or not isinstance(binding.generation, int) or binding.generation <= 0:
            raise ValueError("prepared generation must be a positive integer")
        if not isinstance(binding.coefficients, dict) and not hasattr(binding.coefficients, "items"):
            raise ValueError("profile coefficients must be a mapping")
        coefficients = dict(binding.coefficients)
        if set(coefficients) != set(COEFF_KEYS) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in coefficients.values()
        ):
            raise ValueError("invalid profile coefficients")
        with self._lock:
            current = self._bindings.get(binding.lane_id)
            if current is None:
                raise KeyError(binding.lane_id)
            op = self._catalog["operating_points"].get(binding.op_id)
            if op is None:
                raise KeyError(binding.op_id)
            if coefficients != op["profile"]:
                raise ValueError("binding coefficients do not match catalog operating point")
            if current.generation != expected_generation:
                raise ValueError("stale expected generation")
            if binding.generation <= current.generation:
                raise ValueError("prepared generation must advance")
            activated = ProfileBinding(binding.lane_id, binding.op_id, binding.generation, MappingProxyType(coefficients))
            self._bindings[binding.lane_id] = activated
            return activated


class LatencyScaleFactors:

    __slots__ = ("ttft", "tbt", "_lock")

    def __init__(self, ttft=1.0, tbt=1.0):
        self.ttft = float(ttft)
        self.tbt = float(tbt)
        self._lock = threading.RLock()

    def update(self, ttft=None, tbt=None):
        with self._lock:
            if ttft is not None:
                self.ttft = float(ttft)
            if tbt is not None:
                self.tbt = float(tbt)

    def snapshot(self):
        with self._lock:
            return {"ttft": self.ttft, "tbt": self.tbt}


class ScaledProfileTable:

    __slots__ = ("base", "scales")

    def __init__(self, base, scales):
        self.base = base
        self.scales = scales

    def estimate(self, preset_id, l_in, l_out):
        estimate = self.base.estimate(preset_id=preset_id, l_in=l_in, l_out=l_out)
        current = self.scales.snapshot()
        return ProfileEstimate(
            ttft_s=estimate.ttft_s * current.get("ttft", 1.0),
            tbt_s=estimate.tbt_s * current.get("tbt", 1.0),
            e_prefill_j=estimate.e_prefill_j,
            e_decode_per_tok_j=estimate.e_decode_per_tok_j,
        )


class CallableProfileTable:

    __slots__ = ("predict",)

    def __init__(self, predict):
        self.predict = predict

    def estimate(self, preset_id, l_in, l_out):
        sample = self.predict({"preset_id": preset_id, "l_in": l_in, "l_out": l_out})
        if not isinstance(sample, dict):
            raise TypeError("callable profile predictor must return an object")
        return ProfileEstimate(
            ttft_s=float(sample.get("ttft_s", 0.0) or 0.0),
            tbt_s=float(sample.get("tbt_s", 0.0) or 0.0),
            e_prefill_j=float(sample.get("e_prefill_j", 0.0) or 0.0),
            e_decode_per_tok_j=float(sample.get("e_decode_per_tok_j", 0.0) or 0.0),
        )
