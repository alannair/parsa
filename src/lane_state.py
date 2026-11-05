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
import time


class LaneQueueState:

    __slots__ = ("in_flight", "ewma_service_s", "last_update_s", "_lock")

    def __init__(self):
        self.in_flight = 0
        self.ewma_service_s = 0.0
        self.last_update_s = 0.0
        self._lock = threading.RLock()

    def on_start(self, count=1):
        with self._lock:
            self.in_flight += int(count or 1)
            self.last_update_s = time.time()

    def on_finish(self, observed_service_s, alpha=0.2):
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)
            observed_service_s = max(0.0, float(observed_service_s or 0.0))
            if observed_service_s > 0:
                if self.ewma_service_s <= 0:
                    self.ewma_service_s = observed_service_s
                else:
                    alpha = min(1.0, max(0.0, float(alpha)))
                    self.ewma_service_s = (1.0 - alpha) * self.ewma_service_s + alpha * observed_service_s
            self.last_update_s = time.time()

    def estimate_queue_delay_s(self, fallback_service_s):
        with self._lock:
            service_s = self.ewma_service_s if self.ewma_service_s > 0 else max(0.0, float(fallback_service_s or 0.0))
            return max(0, self.in_flight) * service_s

    def snapshot(self):
        with self._lock:
            return {
                "in_flight": self.in_flight,
                "ewma_service_s": self.ewma_service_s,
                "last_update_s": self.last_update_s,
            }


class QueueManager:

    __slots__ = ("_states", "_lock")

    def __init__(self, preset_ids=None):
        self._states = {}
        self._lock = threading.RLock()
        for preset_id in list(preset_ids or []):
            self.ensure_lane(preset_id)

    def ensure_lane(self, preset_id):
        with self._lock:
            if preset_id not in self._states:
                self._states[preset_id] = LaneQueueState()
            return self._states[preset_id]

    def remove_lane(self, preset_id):
        with self._lock:
            self._states.pop(preset_id, None)

    def ensure_from_presets(self, presets):
        active = []
        for preset in list(presets or []):
            active.append(preset.preset_id)
            self.ensure_lane(preset.preset_id)
        with self._lock:
            for preset_id in list(self._states):
                if preset_id not in active:
                    del self._states[preset_id]

    def get(self, preset_id):
        return self.ensure_lane(preset_id)

    def snapshot(self):
        with self._lock:
            return {preset_id: state.snapshot() for preset_id, state in self._states.items()}

    def total_in_flight(self):
        with self._lock:
            return sum(state.snapshot().get("in_flight", 0) for state in self._states.values())
