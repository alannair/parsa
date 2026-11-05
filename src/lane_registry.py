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

import threading


class LaneRegistry:

    __slots__ = ("_presets", "_order", "_lock", "_rr_index")

    def __init__(self, presets=None):
        self._presets = {}
        self._order = []
        self._lock = threading.RLock()
        self._rr_index = 0
        for preset in list(presets or []):
            self.add(preset)

    def add(self, preset):
        with self._lock:
            if preset.preset_id not in self._presets:
                self._order.append(preset.preset_id)
            self._presets[preset.preset_id] = preset

    def remove(self, preset_id):
        with self._lock:
            self._presets.pop(preset_id, None)
            self._order = [item for item in self._order if item != preset_id]
            self._rr_index = 0 if not self._order else self._rr_index % len(self._order)

    def replace(self, presets):
        with self._lock:
            self._presets = {}
            self._order = []
            self._rr_index = 0
            for preset in list(presets or []):
                self.add(preset)

    def get(self, preset_id):
        with self._lock:
            return self._presets.get(preset_id)

    def ids(self):
        with self._lock:
            return list(self._order)

    def snapshot(self):
        with self._lock:
            return [self._presets[preset_id] for preset_id in self._order if preset_id in self._presets]

    def rescue_preset(self, rescue_preset_id=None):
        with self._lock:
            if rescue_preset_id and rescue_preset_id in self._presets:
                return self._presets[rescue_preset_id]
            for preset_id in self._order:
                preset = self._presets[preset_id]
                if getattr(preset, "lane_type", "") == "rescue":
                    return preset
            return None

    def next_round_robin(self, candidate_ids=None):
        with self._lock:
            allowed = None if candidate_ids is None else set(candidate_ids)
            ordered = []
            for preset_id in self._order:
                if preset_id not in self._presets:
                    continue
                if allowed is not None and preset_id not in allowed:
                    continue
                ordered.append(preset_id)
            if not ordered:
                return None
            preset_id = ordered[self._rr_index % len(ordered)]
            self._rr_index += 1
            return self._presets[preset_id]
