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

from .backends import build_backend_adapter


KNOWN_KEYS = {
    "preset_id",
    "backend_url",
    "max_num_batched_tokens",
    "max_num_seqs",
    "frequency_mhz",
    "lane_type",
    "policy_weights",
    "capabilities",
    "metadata",
    "adapter",
}


def _as_positive_int(value, name, allow_none=False):
    if value is None:
        if allow_none:
            return None
        raise ValueError("%s is required" % name)
    if isinstance(value, bool):
        raise TypeError("%s must be an integer" % name)
    ivalue = int(value)
    if ivalue <= 0:
        raise ValueError("%s must be > 0" % name)
    return ivalue


class Preset:

    __slots__ = (
        "preset_id",
        "backend_url",
        "max_num_batched_tokens",
        "max_num_seqs",
        "frequency_mhz",
        "lane_type",
        "policy_weights",
        "capabilities",
        "metadata",
        "adapter_kind",
        "adapter",
    )

    def __init__(
        self,
        preset_id,
        backend_url,
        max_num_batched_tokens,
        max_num_seqs=None,
        frequency_mhz=None,
        lane_type="generic",
        policy_weights=None,
        capabilities=None,
        metadata=None,
        adapter_kind="vllm",
        adapter=None,
    ):
        self.preset_id = str(preset_id)
        self.backend_url = str(backend_url).rstrip("/")
        self.max_num_batched_tokens = int(max_num_batched_tokens)
        self.max_num_seqs = None if max_num_seqs is None else int(max_num_seqs)
        self.frequency_mhz = None if frequency_mhz is None else int(frequency_mhz)
        self.lane_type = str(lane_type or "generic").strip().lower() or "generic"
        self.policy_weights = dict(policy_weights or {})
        self.capabilities = dict(capabilities or {})
        self.metadata = dict(metadata or {})
        self.adapter_kind = str(adapter_kind or "vllm")
        self.adapter = adapter or build_backend_adapter(self.adapter_kind, self.backend_url, self.capabilities)

    def to_dict(self):
        data = {
            "preset_id": self.preset_id,
            "backend_url": self.backend_url,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "lane_type": self.lane_type,
        }
        if self.max_num_seqs is not None:
            data["max_num_seqs"] = self.max_num_seqs
        if self.frequency_mhz is not None:
            data["frequency_mhz"] = self.frequency_mhz
        if self.policy_weights:
            data["policy_weights"] = dict(self.policy_weights)
        if self.capabilities:
            data["capabilities"] = dict(self.capabilities)
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        if self.adapter_kind and self.adapter_kind != "vllm":
            data["adapter"] = self.adapter_kind
        return data


def preset_from_dict(data):
    if not isinstance(data, dict):
        raise TypeError("preset entries must be objects")
    preset_id = str(data.get("preset_id") or "").strip()
    if not preset_id:
        raise ValueError("preset_id must be non-empty")
    backend_url = str(data.get("backend_url") or "").strip().rstrip("/")
    if not backend_url:
        raise ValueError("backend_url must be non-empty")
    max_num_batched_tokens = _as_positive_int(data.get("max_num_batched_tokens"), "max_num_batched_tokens")
    max_num_seqs = _as_positive_int(data.get("max_num_seqs"), "max_num_seqs", allow_none=True)
    frequency_mhz = _as_positive_int(data.get("frequency_mhz"), "frequency_mhz", allow_none=True)
    lane_type = str(data.get("lane_type") or "generic").strip().lower() or "generic"
    policy_weights = data.get("policy_weights") or {}
    if not isinstance(policy_weights, dict):
        raise TypeError("policy_weights must be an object")
    capabilities = data.get("capabilities") or {}
    if not isinstance(capabilities, dict):
        raise TypeError("capabilities must be an object")
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be an object")
    for key, value in data.items():
        if key not in KNOWN_KEYS:
            metadata[key] = value
    adapter_kind = str(data.get("adapter") or "vllm")
    adapter = build_backend_adapter(adapter_kind, backend_url, capabilities)
    return Preset(
        preset_id=preset_id,
        backend_url=backend_url,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        frequency_mhz=frequency_mhz,
        lane_type=lane_type,
        policy_weights=policy_weights,
        capabilities=capabilities,
        metadata=metadata,
        adapter_kind=adapter_kind,
        adapter=adapter,
    )


def load_presets_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError("presets JSON must be a list")
    presets = []
    seen = set()
    for item in data:
        preset = preset_from_dict(item)
        if preset.preset_id in seen:
            raise ValueError("duplicate preset_id: %s" % preset.preset_id)
        seen.add(preset.preset_id)
        presets.append(preset)
    return presets
