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
import logging
import threading
import time
from collections import deque


class MetricsPipeline:

    __slots__ = (
        "_lock",
        "_route_counts",
        "_events",
        "_rejections",
        "_completions",
        "_admission_tiers",
        "_planner_runs",
        "_reroutes",
        "_queue_events",
        "_feedback_events",
    )

    def __init__(self, max_events=512):
        self._lock = threading.RLock()
        self._route_counts = {}
        self._events = deque(maxlen=max(32, int(max_events or 512)))
        self._rejections = 0
        self._completions = 0
        self._admission_tiers = {}
        self._planner_runs = 0
        self._reroutes = 0
        self._queue_events = 0
        self._feedback_events = 0

    def record_choice(self, choice):
        with self._lock:
            preset_id = choice.preset.preset_id
            self._route_counts[preset_id] = self._route_counts.get(preset_id, 0) + 1
            self._events.append(
                {
                    "ts": time.time(),
                    "event": "choice",
                    "preset_id": preset_id,
                    "rescue": bool(choice.used_rescue),
                    "admission_tier": getattr(choice, "admission_tier", ""),
                    "route_attempts": getattr(choice, "route_attempts", 0),
                    "latency_s": choice.predicted_latency_s,
                    "energy_j": choice.predicted_energy_j,
                }
            )

    def record_completion(self, choice, elapsed_s, status_code=None):
        with self._lock:
            self._completions += 1
            self._events.append(
                {
                    "ts": time.time(),
                    "event": "completion",
                    "preset_id": choice.preset.preset_id,
                    "admission_tier": getattr(choice, "admission_tier", ""),
                    "elapsed_s": float(elapsed_s or 0.0),
                    "status_code": status_code,
                }
            )

    def record_rejection(self, reason, tier="rejected"):
        with self._lock:
            self._rejections += 1
            self._events.append(
                {
                    "ts": time.time(),
                    "event": "rejection",
                    "tier": str(tier or "rejected"),
                    "reason": str(reason or "unknown"),
                }
            )

    def record_admission_decision(self, request_id, tier, reason="", route_attempts=0, preset_id=None, planner_epoch=0):
        with self._lock:
            tier = str(tier or "")
            self._admission_tiers[tier] = self._admission_tiers.get(tier, 0) + 1
            route_attempts = int(route_attempts or 0)
            if route_attempts > 0:
                self._reroutes += 1
            self._events.append(
                {
                    "ts": time.time(),
                    "event": "admission",
                    "request_id": str(request_id or ""),
                    "tier": tier,
                    "reason": str(reason or ""),
                    "route_attempts": route_attempts,
                    "preset_id": str(preset_id or ""),
                    "planner_epoch": int(planner_epoch or 0),
                }
            )

    def record_planner_run(self, planner_epoch, pending_count, guaranteed_count, best_effort_count, duration_s):
        with self._lock:
            self._planner_runs += 1
            self._events.append(
                {
                    "ts": time.time(),
                    "event": "planner_run",
                    "planner_epoch": int(planner_epoch or 0),
                    "pending_count": int(pending_count or 0),
                    "guaranteed_count": int(guaranteed_count or 0),
                    "best_effort_count": int(best_effort_count or 0),
                    "duration_s": float(duration_s or 0.0),
                }
            )

    def record_queue_event(self, request_id, state, reason="", blocked_on=None):
        with self._lock:
            self._queue_events += 1
            self._events.append(
                {
                    "ts": time.time(),
                    "event": "queue",
                    "request_id": str(request_id or ""),
                    "state": str(state or ""),
                    "reason": str(reason or ""),
                    "blocked_on": list(blocked_on or []),
                }
            )

    def record_feedback_observation(self, request_id, observation):
        with self._lock:
            self._feedback_events += 1
            payload = dict(observation or {})
            payload.update(
                {
                    "ts": time.time(),
                    "event": "feedback",
                    "request_id": str(request_id or ""),
                }
            )
            self._events.append(payload)

    def record_registry_change(self, action, preset_id):
        with self._lock:
            self._events.append(
                {"ts": time.time(), "event": "registry", "action": str(action), "preset_id": str(preset_id)}
            )

    def record_shadow_event(self, evidence):
        # Only constructed shadow evidence, never a Choice, request or policy state.
        allowed = ("identity_source", "runtime_context", "catalog_hash", "config_hash",
                   "observation", "estimates", "decision", "stage", "error_type")
        payload = {key: evidence[key] for key in allowed if key in evidence}
        payload.update(ts=time.time(), event="op_drift", mode="shadow", executed=False)
        text = json.dumps(payload, sort_keys=True, allow_nan=False)
        with self._lock:
            self._events.append(json.loads(text))
        logging.getLogger("src.router_app").info("op_drift %s", text)

    def snapshot(self):
        with self._lock:
            return {
                "route_counts": dict(self._route_counts),
                "rejections": self._rejections,
                "completions": self._completions,
                "admission_tiers": dict(self._admission_tiers),
                "planner_runs": self._planner_runs,
                "reroutes": self._reroutes,
                "queue_events": self._queue_events,
                "feedback_events": self._feedback_events,
                "events": list(self._events),
            }
